import jax.numpy as jnp
import jax
import chex
import flashbax as fbx

from typing import Callable, Any, Optional

from experior.envs import Environment, MetaParam
from .utils import RewardModel


def make_thompson_sampling(
    env: Environment,
    reward_model: RewardModel,
    num_envs: int,
    total_steps: int,
    langevin_batch_size: int,
    langevin_updates_per_step: int,
    prior_log_pdf: Callable[[Any, Optional[MetaParam]], chex.Array] = None,
):
    if prior_log_pdf is None:
        prior_log_pdf = jax.tree_util.Partial(lambda p, _: jnp.zeros((1,)))

    def train(
        rng,
        langevin_learning_rate: float,
        meta_params: chex.Array = None,
        langevin_grad_clip: float = 50.0,
    ):
        def langevin_posterior_sampling(
            rng: chex.PRNGKey,
            reward_params: Any,
            observations: chex.Array,
            actions: chex.Array,
            rewards: chex.Array,
            step_i: int,
        ):
            # using https://proceedings.mlr.press/v119/mazumdar20a/mazumdar20a.pdf
            def _update_parameters(runner_state, _):
                rng, params = runner_state
                # Compute the gradient of the unnormalized log prior
                grad_log_prior = jax.grad(
                    lambda p: prior_log_pdf(p, meta_params).sum()
                )(params)

                # Compute the gradient of the log likelihood
                def log_likelihood_fn(p):
                    log_like = jax.vmap(
                        lambda p, o, a, r: reward_model.apply(
                            p, o, a, r, method=reward_model.log_pdf
                        ),
                        in_axes=(None, 0, 0, 0),
                    )(p, observations, actions, rewards).sum()
                    return log_like

                grad_log_likelihood = jax.grad(log_likelihood_fn)(params)
                # Sum the gradients to get the gradient of the log posterior
                grad_log_posterior = jax.tree_util.tree_map(
                    lambda prior, likelihood: prior
                    + step_i / langevin_batch_size * likelihood,
                    grad_log_prior,
                    grad_log_likelihood,
                )

                # write a clip the gradient # TODO
                grad_log_posterior = jax.tree_util.tree_map(
                    lambda p: jnp.clip(p, -langevin_grad_clip, langevin_grad_clip),
                    grad_log_posterior,
                )

                step_size = langevin_learning_rate * (1 - step_i / total_steps)
                # Langevin dynamics update rule
                rng, rng_ = jax.random.split(rng)
                num_vars = len(jax.tree_util.tree_leaves(grad_log_posterior))
                treedef = jax.tree_util.tree_structure(grad_log_posterior)
                updated_param = jax.tree_util.tree_map(
                    lambda p, g, k: p
                    + step_size * g
                    + jnp.sqrt(2 * step_size)
                    * jax.random.normal(k, shape=g.shape, dtype=g.dtype),
                    params,
                    grad_log_posterior,
                    jax.tree_util.tree_unflatten(
                        treedef, jax.random.split(rng_, num_vars)
                    ),
                )
                runner_state = (rng, updated_param)
                return runner_state, None

            runner_state = (rng, reward_params)
            runner_state, _ = jax.lax.scan(
                _update_parameters, runner_state, None, langevin_updates_per_step
            )
            return runner_state[1]

        # init env
        rng, rng_ = jax.random.split(rng)
        env_params = jax.vmap(env.init_env, in_axes=(0, None, None))(
            jax.random.split(rng_, num_envs), env.default_params, meta_params
        )
        rng, rng_ = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(
            jax.random.split(rng_, num_envs), env_params
        )

        # init reward model
        rng, rng_ = jax.random.split(rng)
        reward_params = jax.vmap(reward_model.init)(
            jax.random.split(rng_, num_envs), obs
        )

        # replay buffer
        rng, rng_ = jax.random.split(rng)
        buffer = fbx.make_item_buffer(
            max_length=total_steps, min_length=1, sample_batch_size=langevin_batch_size
        )

        rng, rng_ = jax.random.split(rng)
        buffer_state = buffer.init(
            {
                "obs": obs,
                "action": jax.vmap(lambda k, p: env.action_space(p).sample(k))(
                    jax.random.split(rng_, num_envs), env_params
                ),
                "reward": jnp.zeros((num_envs,)),
            }
        )

        def _env_step(runner_state, i):
            obs, env_state, reward_params, buffer_state, rng = runner_state

            # select (Thompson Sampling)
            rng, rng_ = jax.random.split(rng)
            action = jax.vmap(
                lambda p, x: reward_model.apply(p, x, method=reward_model.best_action)
            )(reward_params, obs).reshape((num_envs,))

            # step env
            rng, rng_ = jax.random.split(rng)
            new_obs, env_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(rng_, num_envs), env_state, action, env_params
            )

            # add to buffer
            buffer_state = buffer.add(
                buffer_state,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                },
            )

            # update reward model
            rng, rng_ = jax.random.split(rng)
            batch = buffer.sample(buffer_state, rng_)
            rng, rng_ = jax.random.split(rng)
            params = jax.vmap(
                langevin_posterior_sampling, in_axes=(0, 0, 1, 1, 1, None)
            )(
                jax.random.split(rng_, num_envs),
                reward_params,
                batch.experience["obs"],
                batch.experience["action"],
                batch.experience["reward"],
                i,
            )
            runner_state = (new_obs, env_state, params, buffer_state, rng)
            return (
                runner_state,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    **info,
                },
            )

        runner_state = (obs, env_state, reward_params, buffer_state, rng)
        runner_state, metrics = jax.lax.scan(
            _env_step, runner_state, jnp.arange(total_steps)
        )

        return runner_state, metrics

    return train


def make_bernoulli_thompson_sampling(
    env: Environment,
    num_envs: int,
    total_steps: int,
    prior_alpha_betas: Callable[[Optional[MetaParam]], chex.Array],
):
    def train(rng, meta_params: chex.Array = None):
        # init env
        rng, rng_ = jax.random.split(rng)
        env_params = jax.vmap(env.init_env, in_axes=(0, None, None))(
            jax.random.split(rng_, num_envs), env.default_params, meta_params
        )
        rng, rng_ = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(
            jax.random.split(rng_, num_envs), env_params
        )

        alpha_beta_prior = prior_alpha_betas(meta_params)  # shape: (num_actions, 2)
        # success counts
        sum_rewards = jnp.zeros((num_envs, alpha_beta_prior.shape[0]))
        total_action_counts = jnp.zeros((num_envs, alpha_beta_prior.shape[0]))

        def _env_step(runner_state, i):
            obs, env_state, sum_rewards, total_action_counts, rng = runner_state

            # select (Thompson Sampling)
            rng, rng_ = jax.random.split(rng)
            action = jax.random.beta(
                rng_,
                sum_rewards + alpha_beta_prior[:, 0],
                alpha_beta_prior[:, 1] + total_action_counts - sum_rewards,
            ).argmax(
                axis=-1
            )  # shape: (num_envs,)

            # step env
            rng, rng_ = jax.random.split(rng)
            new_obs, env_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(rng_, num_envs), env_state, action, env_params
            )
            # update total_action_counts and sum_rewards
            total_action_counts = total_action_counts.at[
                jnp.arange(num_envs), action
            ].add(1)
            sum_rewards = sum_rewards.at[jnp.arange(num_envs), action].add(reward)

            runner_state = (new_obs, env_state, sum_rewards, total_action_counts, rng)
            return (
                runner_state,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    **info,
                },
            )

        runner_state = (obs, env_state, sum_rewards, total_action_counts, rng)
        runner_state, metrics = jax.lax.scan(
            _env_step, runner_state, jnp.arange(total_steps)
        )

        return runner_state, metrics

    return train
