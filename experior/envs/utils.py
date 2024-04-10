import chex
import jax

import jax.numpy as jnp

from gymnax.environments import spaces, environment, EnvParams, EnvState
from typing import Union


def uniform_sample_ball(rng_key, size: int, d: int) -> chex.Array:
    """Samples uniformly from the unit ball in R^{d}.
    Source: https://stats.stackexchange.com/questions/481715/generating-uniform-points-inside-an-m-dimensional-ball

    Args:
        rng_key: A JAX random key.
        size: The size of the sample.
        d: The dimension of the ball.

    Returns:
        A sample from the unit ball in R^{d}, shape (size, d).
    """
    key1, key2 = jax.random.split(rng_key)
    norm = jax.random.normal(key1, (size, d))
    sphere = norm / jnp.linalg.norm(norm, axis=-1, keepdims=True)

    # Generate random radii
    random_radii = jax.random.uniform(key2, (size, 1)) ** (1 / d)
    return sphere * random_radii


class UnitBall(spaces.Space):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.d = d

    def sample(self, rng: chex.PRNGKey, size: int = 1) -> chex.Array:
        return uniform_sample_ball(rng, size, self.d)

    def contains(self, x: chex.Array) -> bool:
        return jnp.all(jnp.linalg.norm(x, axis=-1) <= 1.0)


class Environment(environment.Environment):
    def init_env(
        self, key: chex.PRNGKey, params: EnvParams, meta_params: chex.Array = None
    ) -> EnvParams:
        """Initialize environment state."""
        raise NotImplementedError

    def optimal_policy(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ) -> Union[int, float, chex.Array]:
        raise NotImplementedError

    def optimal_value(self, state: EnvState, params: EnvParams) -> float:
        raise NotImplementedError

    def q_function(
        self, state: EnvState, param: EnvParams, action: Union[int, float, chex.Array]
    ) -> float:
        raise NotImplementedError
