import jax
import optax
import chex

import flax.linen as nn
import flashbax as fbx
import jax.numpy as jnp

from typing import Dict

from experior.experts import Trajectory
from experior.utils import VecTrainState


def make_ensemble_prior_train(
    q_network: nn.Module,
    steps: int,
    num_ensemble: int,
    batch_size: int,
    optimizer: optax.GradientTransformation,
):

    def ensemble_prior_train(rng, expert_trajectories: Trajectory, expert_beta: float):
        rng, rng_ = jax.random.split(rng)
        q_init_params = jax.vmap(q_network.init, (0, None))(
            jax.random.split(rng_, num_ensemble), expert_trajectories.obs[0]
        )
        ensemble_q_state = VecTrainState.create(
            apply_fn=q_network.apply,
            params=q_init_params,
            tx=optimizer,
        )
        n_trajectories = expert_trajectories.obs.shape[0]
        horizon = expert_trajectories.obs.shape[1]

        expert_buffer = fbx.make_trajectory_buffer(
            max_length_time_axis=horizon,
            min_length_time_axis=horizon,
            sample_batch_size=batch_size,
            period=1,
            add_batch_size=n_trajectories,
            sample_sequence_length=1,
        )

        buffer_state = expert_buffer.init(
            {
                "obs": expert_trajectories.obs[0][0],
                "action": expert_trajectories.action[0][0],
            }
        )
        buffer_state = expert_buffer.add(
            buffer_state,
            {
                "obs": expert_trajectories.obs,
                "action": expert_trajectories.action,
            },
        )

        def pretrain_q_network(
            q_state: VecTrainState, expert_batch: Dict
        ) -> VecTrainState:
            """minimizes the loss in Proposition (Ensemble Marginal Likelihood)"""
            expert_obs, expert_action = (
                expert_batch["obs"][:, 0],
                expert_batch["action"][:, 0],
            )

            def pretrain_loss(params) -> chex.Array:
                def loss(q_t: chex.Array, a_t: chex.Numeric) -> chex.Numeric:
                    chex.assert_rank([q_t, a_t], [1, 0])
                    chex.assert_type([q_t, a_t], [float, int])
                    target_t = (1.0 / expert_beta) * jnp.log(
                        jnp.sum(jnp.exp(expert_beta * q_t))
                    )
                    return target_t - q_t[a_t]

                batch_q_value = jax.vmap(q_state.apply_fn, (0, None))(
                    params, expert_obs
                )  # (num_ensemble, batch_size, n_actions)
                batch_error = jax.vmap(jax.vmap(loss), (0, None))(
                    batch_q_value, expert_action
                )  # (num_ensemble, batch_size)
                return jnp.mean(batch_error)

            loss_value, grad = jax.value_and_grad(pretrain_loss)(q_state.params)
            q_state = q_state.apply_gradients(grads=grad)
            return q_state, loss_value

        def pretrain_step(runner_state, _):
            q_states, rng = runner_state

            # action from q network
            rng, rng_ = jax.random.split(rng)
            expert_batch = expert_buffer.sample(buffer_state, rng_)
            new_q_states, loss = pretrain_q_network(q_states, expert_batch.experience)

            runner_state = (
                new_q_states,
                rng,
            )
            return runner_state, {
                "loss": loss,
            }

        runner_state = (ensemble_q_state, rng)
        new_runner_state, loss = jax.lax.scan(
            pretrain_step, runner_state, jnp.arange(steps)
        )

        return new_runner_state[0], loss

    return ensemble_prior_train
