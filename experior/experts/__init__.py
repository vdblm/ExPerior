import jax

import chex
from typing import NamedTuple

import jax.numpy as jnp

from experior.envs import Environment
from gymnax.environments import spaces
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


class Trajectory(NamedTuple):
    action: chex.Array
    reward: chex.Array
    obs: chex.Array
    value: chex.Array = None
    log_prob: chex.Array = None
    done: chex.Array = None
    info: chex.Array = None


def generate_optimal_trajectories(
    rng: chex.PRNGKey,
    env: Environment,
    num_trajectories: int,
    horizon: int,
    meta_params: chex.Array = None,
):
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    rng, rng_ = jax.random.split(rng)
    params = env.default_params
    params = jax.vmap(env.init_env, in_axes=(0, None, None))(
        jax.random.split(rng_, num_trajectories), params, meta_params
    )

    def optimal_rollout(key, params):
        obs, state = env.reset(key, params)

        def _env_step(runner_state, _):
            state, rng, obs = runner_state
            rng, rng_ = jax.random.split(rng)
            action = env.optimal_policy(rng_, state, params)
            rng, rng_ = jax.random.split(rng)
            next_obs, next_state, reward, done, info = env.step(
                rng_, state, action, params
            )
            runner_state = (next_state, rng, next_obs)
            return runner_state, Trajectory(action, reward, obs)

        runner_state = (state, key, obs)
        runner_state, trajectory = jax.lax.scan(_env_step, runner_state, None, horizon)
        return trajectory

    expert_trajectories = jax.vmap(optimal_rollout, in_axes=(0, 0))(
        jax.random.split(rng, num_trajectories), params
    )

    return expert_trajectories


def generate_noisy_optimal_trajectories(
    rng: chex.PRNGKey,
    env: Environment,
    num_trajectories: int,
    horizon: int,
    gamma: float,
    meta_params: chex.Array = None,
):
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    rng, rng_ = jax.random.split(rng)
    params = env.default_params
    params = jax.vmap(env.init_env, in_axes=(0, None, None))(
        jax.random.split(rng_, num_trajectories), params, meta_params
    )

    def optimal_noisy_rollout(key, params):
        obs, state = env.reset(key, params)

        def _env_step(runner_state, _):
            state, rng, obs = runner_state
            rng, rng1, rng2 = jax.random.split(rng, 3)
            action = jnp.where(
                jax.random.uniform(rng1) < gamma,
                env.optimal_policy(rng2, state, params),
                env.action_space(params).sample(rng2),
            )
            rng, rng_ = jax.random.split(rng)
            next_obs, next_state, reward, done, info = env.step(
                rng_, state, action, params
            )
            runner_state = (next_state, rng, next_obs)
            return runner_state, Trajectory(action, reward, obs)

        runner_state = (state, key, obs)
        runner_state, trajectory = jax.lax.scan(_env_step, runner_state, None, horizon)
        return trajectory

    expert_trajectories = jax.vmap(optimal_noisy_rollout, in_axes=(0, 0))(
        jax.random.split(rng, num_trajectories), params
    )

    return expert_trajectories


def generate_discrete_noisy_rational_trajectories(
    rng: chex.PRNGKey,
    env: Environment,
    num_trajectories: int,
    horizon: int,
    beta: float,
    meta_params: chex.Array = None,
):
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    rng, rng_ = jax.random.split(rng)
    params = env.default_params
    if not isinstance(env.action_space(params), spaces.Discrete):
        raise NotImplementedError
    params = jax.vmap(env.init_env, in_axes=(0, None, None))(
        jax.random.split(rng_, num_trajectories), params, meta_params
    )

    action_space = jnp.arange(env.action_space(params).n)

    def noisy_rational_rollout(key, params):
        obs, state = env.reset(key, params)

        def _env_step(runner_state, _):
            state, rng, obs = runner_state
            q_functions = jax.vmap(lambda a: env.q_function(state, params, a))(
                action_space
            )
            probs = jax.nn.softmax(beta * q_functions)
            rng, rng_ = jax.random.split(rng)
            action = jax.random.choice(rng_, action_space, p=probs)
            rng, rng_ = jax.random.split(rng)
            next_obs, next_state, reward, done, info = env.step(
                rng_, state, action, params
            )
            runner_state = (next_state, rng, next_obs)
            return runner_state, Trajectory(action, reward, obs)

        runner_state = (state, key, obs)
        runner_state, trajectory = jax.lax.scan(_env_step, runner_state, None, horizon)
        return trajectory

    expert_trajectories = jax.vmap(noisy_rational_rollout, in_axes=(0, 0))(
        jax.random.split(rng, num_trajectories), params
    )

    return expert_trajectories


def expert_log_likelihood_fn(
    beta: float,
    # shape: (n_trajectory, n_horizon, n_action), for continuous actions,
    # n_action is the number of sampled actions
    q_values: chex.Array,
    taken_q_values: chex.Array,  # shape: (n_trajectory, n_horizon)
):
    max_value = jnp.max(beta * q_values, axis=2)  # n_trajectory x n_horizon
    shifted_q_values = (
        beta * q_values - max_value[..., None]
    )  # n_trajectory x n_horizon x n_action
    shifted_taken_q_values = (
        beta * taken_q_values - max_value
    )  # n_trajectory x n_horizon
    # n_trajectory x n_horizon
    denominator = jnp.log(jnp.sum(jnp.exp(shifted_q_values), axis=2) + 1e-6)
    numerator = shifted_taken_q_values
    return (numerator - denominator).sum(axis=1)  # n_trajectory
