"""Code addapted from https://raw.githubusercontent.com/google-deepmind/bsuite/main/bsuite/baselines/jax/boot_dqn/agent.py"""

import jax.numpy as jnp
import jax
import flax
import rlax
import chex
import optax
import flashbax as fbx
import flax.linen as nn

from typing import Callable, Union, Sequence, Dict

from experior.envs import Environment, EnvParams
from experior.utils import VecTrainState

from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


class Q_TrainState(VecTrainState):
    target_params: flax.core.FrozenDict


def make_boot_dqn_train(
    env: Environment,
    q_network: nn.Module,
    buffer_size: int,
    batch_size: int,
    steps: int,
    learning_starts: int,
    num_ensemble: int,
    optimizer: optax.GradientTransformation,
    q_init_fn: Callable[[chex.PRNGKey, chex.Array], flax.core.FrozenDict],
    epsilon_fn: Callable[[int], float] = lambda _: 0.0,
    discount_factor: float = 1.0,
):
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    assert learning_starts <= buffer_size

    def train(
        rng,
        env_params: EnvParams,
        mask_prob: float,
        noise_scale: float,
        train_frequency: int,
        target_network_frequency: int,
    ):
        # Initialize parameters and optimizer state for an ensemble of Q-networks.
        rng, rng_ = jax.random.split(rng)
        obs, _ = env.reset(rng_, env_params)
        rng, rng_ = jax.random.split(rng)
        q_init_params = jax.vmap(q_init_fn, (0, None))(
            jax.random.split(rng_, num_ensemble), obs
        )

        ensemble_q_state = Q_TrainState.create(
            apply_fn=q_network.apply,
            params=q_init_params,
            target_params=q_init_params,
            tx=optimizer,
        )

        active_head = jnp.array(0, dtype=jnp.int32)

        rng, rng_ = jax.random.split(rng)
        obs, env_state = env.reset(rng_, env_params)

        # replay buffer
        rng, rng_ = jax.random.split(rng)
        buffer = fbx.make_item_buffer(
            max_length=buffer_size,
            min_length=learning_starts,
            sample_batch_size=batch_size,
        )

        # TODO only for int action types
        buffer_state = buffer.init(
            {
                "obs": obs,
                "action": jnp.array(0),
                "reward": jnp.array(0.0),
                "done": jnp.array(False),
                "next_obs": obs,
                "mask": jnp.zeros(num_ensemble, dtype=jnp.bool_),
                "noise": jnp.zeros(num_ensemble),
            }
        )

        # Define update function for each member of ensemble..
        def update_q_network(q_state: Q_TrainState, batch: Dict) -> Q_TrainState:
            """Does a step of SGD for the whole ensemble over `transitions`."""

            obs, action, next_obs, reward, done, mask, noise = (
                batch["obs"],
                batch["action"],
                batch["next_obs"],
                batch["reward"],  # (batch_size,)
                batch["done"],
                batch["mask"],  # (batch_size, num_ensemble)
                batch["noise"],  # (batch_size, num_ensemble)
            )
            mask = mask.T  # (num_ensemble, batch_size)
            noise = noise.T  # (num_ensemble, batch_size)

            q_next_target = jax.vmap(q_state.apply_fn, (0, None))(
                q_state.target_params, next_obs
            )  # (num_ensemble, batch_size, n_actions)

            # Define loss function, including bootstrap mask `mask` & reward noise `noise`.
            def loss(params) -> chex.Array:
                """Q-learning loss with added reward noise + half-in bootstrap."""
                q_value = jax.vmap(q_state.apply_fn, (0, None))(
                    params, obs
                )  # (num_ensemble, batch_size, n_actions)
                r_t = reward + noise_scale * noise  # (num_ensemble, batch_size)
                batch_q_learning = jax.vmap(
                    jax.vmap(rlax.q_learning), (0, None, 0, None, 0)
                )
                td_error = batch_q_learning(
                    q_value, action, r_t, discount_factor * (1 - done), q_next_target
                )  # (num_ensemble, batch_size)
                return jnp.sum(jnp.mean(mask * td_error**2, axis=1))

            loss_value, grad = jax.value_and_grad(loss)(q_state.params)
            q_state = q_state.apply_gradients(grads=grad)
            return loss_value, q_state

        def _env_step(runner_state, i):
            obs, env_state, q_states, active_head, rng, buffer_state = runner_state
            epsilon = epsilon_fn(i)
            # action from q network
            rng, rng1, rng2 = jax.random.split(rng, 3)
            active_params = jax.tree_map(lambda x: x[active_head], q_states.params)
            action = jnp.where(
                jax.random.uniform(rng1) < epsilon,
                env.action_space().sample(rng2),
                jnp.argmax(
                    q_states.apply_fn(active_params, obs),
                    axis=-1,
                ),
            )
            rng, rng_ = jax.random.split(rng)
            next_obs, env_state, reward, done, info = env.step(
                rng_, env_state, action, env_params
            )

            active_head = jnp.where(
                done, jax.random.randint(rng_, (), 0, num_ensemble), active_head
            )  # only resample active head at the end of episode

            mask = jax.random.bernoulli(rng_, mask_prob, (num_ensemble,))
            noise = jax.random.normal(rng_, (num_ensemble,))

            # add to replay buffer
            buffer_state = buffer.add(
                buffer_state,
                {
                    "obs": obs,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "next_obs": next_obs,
                    "mask": mask,
                    "noise": noise,
                },
            )

            def _train():
                # sample from replay buffer
                k, rng_ = jax.random.split(rng)
                batch = buffer.sample(buffer_state, rng_)
                # train q network
                l, qs = update_q_network(q_states, batch.experience)
                return k, l, qs

            def _no_train():
                return (
                    rng,
                    0.0,
                    q_states,
                )

            # training
            rng, loss, q_states = jax.lax.cond(
                (i > learning_starts) * (i % train_frequency == 0),
                _train,
                _no_train,
            )

            q_states = jax.lax.cond(
                (i > learning_starts) * (i % target_network_frequency == 0),
                lambda: q_states.replace(target_params=q_states.params),
                lambda: q_states,
            )

            runner_state = (
                next_obs,
                env_state,
                q_states,
                active_head,
                rng,
                buffer_state,
            )
            return runner_state, {
                "loss": loss,
                "reward": reward,
                "done": done,
                "action": action,
                "info": info,
            }

        runner_state = (
            obs,
            env_state,
            ensemble_q_state,
            active_head.astype(jnp.int64),
            rng,
            buffer_state,
        )
        runner_state, output = jax.lax.scan(_env_step, runner_state, jnp.arange(steps))

        return runner_state, output

    return train
