import jax.numpy as jnp
import jax
import chex

from experior.envs import Environment


def make_multi_armed_ucb(env: Environment, num_envs: int, total_steps: int):
    def train(rng, rho, meta_params: chex.Array = None):
        # init env
        rng, rng_ = jax.random.split(rng)
        env_params = jax.vmap(env.init_env, in_axes=(0, None, None))(
            jax.random.split(rng_, num_envs), env.default_params, meta_params
        )
        rng, rng_ = jax.random.split(rng)
        obs, env_state = jax.vmap(env.reset)(
            jax.random.split(rng_, num_envs), env_params
        )
        n_actions = env.num_actions

        # success counts
        sum_rewards = jnp.zeros((num_envs, n_actions))
        action_pulls = jnp.ones((num_envs, n_actions)) * 1e-6

        def _env_step(runner_state, i):
            obs, env_state, sum_rewards, action_pulls, rng = runner_state

            # select (UCB)
            rng, rng_ = jax.random.split(rng)
            ciw = jnp.sqrt(rho * jnp.log(i + 1))
            ucb = sum_rewards / action_pulls + ciw * jnp.sqrt(1 / action_pulls)
            action = jnp.argmax(ucb, axis=-1)

            # step env
            rng, rng_ = jax.random.split(rng)
            new_obs, env_state, reward, done, info = jax.vmap(env.step)(
                jax.random.split(rng_, num_envs), env_state, action, env_params
            )
            # update total_action_counts and sum_rewards
            action_pulls = action_pulls.at[jnp.arange(num_envs), action].add(1)
            sum_rewards = sum_rewards.at[jnp.arange(num_envs), action].add(reward)

            runner_state = (new_obs, env_state, sum_rewards, action_pulls, rng)
            return (
                runner_state,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    **info,
                },
            )

        runner_state = (obs, env_state, sum_rewards, action_pulls, rng)
        runner_state, metrics = jax.lax.scan(
            _env_step, runner_state, jnp.arange(total_steps)
        )

        return runner_state, metrics

    return train
