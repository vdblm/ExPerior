from typing import Any, Callable

from flax import core
from flax import struct

import optax
import chex
import jax

import jax.numpy as jnp

import flax.linen as nn

from flax.training.train_state import TrainState
from experior.experts import Trajectory, expert_log_likelihood_fn
from jax.flatten_util import ravel_pytree


class VectMaxEntTrainState(struct.PyTreeNode):
    step: int
    lambda_: float
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    log_prior_fn: Callable = struct.field(pytree_node=False)
    max_param_value: float = 5.0

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = jax.vmap(self.tx.update)(
            grads, self.opt_state, self.params
        )
        new_params = jax.vmap(optax.apply_updates)(self.params, updates)

        # TODO clip the new params
        new_params = jax.tree_util.tree_map(
            lambda p: jnp.clip(p, -self.max_param_value, self.max_param_value),
            new_params,
        )
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    # def reset_opt_state(self):
    #     return self.replace(opt_state=jax.vmap(self.tx.init)(self.params))

    @classmethod
    def create(cls, *, rng, n_trajectory, lambda_, tx, num_envs, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        params = jax.random.normal(rng, (num_envs, n_trajectory))
        opt_state = jax.vmap(tx.init)(params)

        n_horizon = kwargs["n_horizon"] if "n_horizon" in kwargs else 1

        def log_prior_fn(params, traj_log_likelihoods):
            log_prior = jnp.exp(params - jnp.log(n_trajectory * n_horizon)) @ (
                traj_log_likelihoods
                * jax.lax.stop_gradient(jnp.exp(traj_log_likelihoods))
            )

            return log_prior

        return cls(
            step=0,
            params=params,
            tx=tx,
            opt_state=opt_state,
            lambda_=lambda_,
            log_prior_fn=log_prior_fn,
            **kwargs,
        )

    def make_max_ent_update_step(self, sampled_log_likelihoods: chex.Array):
        # sampled_log_likelihoods shape: (num_envs, prior_n_samples, n_trajectory)
        def max_ent_update_step(state, _):
            def single_loss_fn(params, log_likelihoods):
                # log_likelihoods shape: (prior_n_samples, n_trajectory)
                # params shape: (n_trajectory,)
                alphas = jnp.exp(params)
                m_alpha = jnp.exp(log_likelihoods) @ alphas
                loss = (
                    -jax.scipy.special.logsumexp(
                        m_alpha, axis=0, b=1.0 / m_alpha.shape[0]
                    )
                    + self.lambda_ * jnp.log(alphas / self.lambda_).sum()
                )
                return -loss.mean()

            multi_loss_fn = lambda params: jax.vmap(single_loss_fn)(
                params, sampled_log_likelihoods
            ).mean()
            loss, grad = jax.value_and_grad(multi_loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grad)
            return new_state, {"loss": loss}

        return max_ent_update_step


class MaxEntTrainState(struct.PyTreeNode):
    step: int
    lambda_: float
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)
    log_prior_fn: Callable = struct.field(pytree_node=False)
    max_param_value: float = 10.0

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def reset_opt_state(self):
        return self.replace(opt_state=self.tx.init(self.params))

    @classmethod
    def create(cls, *, rng, n_trajectory, lambda_, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        params = jax.random.normal(rng, (n_trajectory,))
        opt_state = tx.init(params)

        def log_prior_fn(params, traj_log_likelihoods):
            log_prior = jnp.exp(params) @ (traj_log_likelihoods)

            return log_prior

        return cls(
            step=0,
            params=params,
            tx=tx,
            opt_state=opt_state,
            lambda_=lambda_,
            log_prior_fn=log_prior_fn,
            **kwargs,
        )

    def make_max_ent_update_step(self, sampled_log_likelihoods: chex.Array):
        # sampled_log_likelihoods shape: (prior_n_samples, n_trajectory)
        def max_ent_update_step(state, _):
            def loss_fn(params):
                # log_likelihoods shape: (prior_n_samples, n_trajectory)
                # params shape: (n_trajectory,)
                alphas = jnp.exp(params)
                m_alpha = jnp.exp(sampled_log_likelihoods) @ alphas
                loss = (
                    -jax.scipy.special.logsumexp(
                        m_alpha, axis=0, b=1.0 / m_alpha.shape[0]
                    )
                    + self.lambda_ * jnp.log(alphas / self.lambda_).sum()
                )
                return -loss.mean()

            loss, grad = jax.value_and_grad(loss_fn)(state.params)
            new_state = state.apply_gradients(grads=grad)
            return new_state, {"loss": loss}

        return max_ent_update_step


def get_expert_log_likelihood(
    param, q_network, expert_trajectories, traj_i, horizon_j, expert_beta, reg=0.0
):
    # expert_trajectories.obs shape: (n_trajectory, n_horizon, obs_dim)
    # expert_q_values shape: (n_trajectory, n_horizon, n_action)
    expert_q_values = q_network.apply(param, expert_trajectories.obs)
    expert_q_values_taken = expert_q_values[
        traj_i, horizon_j, expert_trajectories.action.squeeze()
    ]
    # shape: (n_trajectory,)
    likelihoods = expert_log_likelihood_fn(
        expert_beta, expert_q_values, expert_q_values_taken
    )
    flatten_param, _ = ravel_pytree(param)
    regularizer = reg * (jnp.linalg.norm(flatten_param) + (expert_q_values**2).mean())
    return likelihoods - regularizer


def make_max_ent_log_pdf(
    prior_state: MaxEntTrainState,
    q_network: nn.Module,
    expert_trajectories: Trajectory,
    expert_beta: float,
    reg: float = 0.0,
):
    traj_i, horizon_j = jnp.meshgrid(
        jnp.arange(expert_trajectories.obs.shape[0]),
        jnp.arange(expert_trajectories.obs.shape[1]),
        indexing="ij",
    )

    def max_ent_log_pdf(q_params):
        likelihood = get_expert_log_likelihood(
            q_params,
            q_network,
            expert_trajectories,
            traj_i,
            horizon_j,
            expert_beta,
            reg,
        )

        return prior_state.log_prior_fn(prior_state.params, likelihood)

    return max_ent_log_pdf


def make_max_ent_prior_train(
    q_network: nn.Module,
    epochs: int,
    batch_size: int,
):
    def max_ent_prior_trian(
        rng,
        expert_trajectories: Trajectory,
        expert_beta: float,
        learning_rate: float,
        lambda_: float = 1.0,
        reg: float = 0.0,
    ):
        n_trajectory = expert_trajectories.obs.shape[0]
        horizon = expert_trajectories.obs.shape[1]

        rng, rng_ = jax.random.split(rng)
        tx = optax.adamw(learning_rate)
        prior_state = MaxEntTrainState.create(
            rng=rng_,
            n_trajectory=n_trajectory,
            lambda_=lambda_,
            tx=tx,
        )

        traj_i, horizon_j = jnp.meshgrid(
            jnp.arange(n_trajectory), jnp.arange(horizon), indexing="ij"
        )

        rng, rng_ = jax.random.split(rng)
        sampled_q_params = jax.vmap(q_network.init, (0, None))(
            jax.random.split(rng_, batch_size), expert_trajectories.obs[0][0]
        )
        expert_log_like = lambda param: get_expert_log_likelihood(
            param,
            q_network,
            expert_trajectories,
            traj_i,
            horizon_j,
            expert_beta,
            reg,
        )
        sampled_log_likelihoods = jax.vmap(expert_log_like)(sampled_q_params)
        update_step = prior_state.make_max_ent_update_step(sampled_log_likelihoods)

        prior_state, loss = jax.lax.scan(update_step, prior_state, jnp.arange(epochs))

        return prior_state, loss

    return max_ent_prior_trian
