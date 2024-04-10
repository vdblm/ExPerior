import chex
import abc
import jax

import jax.numpy as jnp
import flax.linen as nn

from typing import Callable, Literal, Union


class RewardModel(abc.ABC, nn.Module):
    def log_pdf(
        self, obs: chex.Array, action: Union[chex.Array, int, float], reward: float
    ):
        raise NotImplementedError

    def best_action(self, obs: chex.Array):
        raise NotImplementedError

    def reward_mean(self, obs: chex.Array, action: Union[chex.Array, int, float]):
        raise NotImplementedError


class LinearDiscreteRewardModel(RewardModel):
    n_actions: int
    params_dim: int
    feature_fn: Callable[[chex.Array, int], chex.Array]
    dist: Literal["bernoulli", "normal"]
    eps: float = 1e-6

    @nn.compact
    def __call__(self, obs: chex.Array):
        # return the means vector
        if len(obs.shape) == 1:
            obs = obs.reshape((1, -1))
        obs = obs.reshape((obs.shape[0], -1))
        params = self.param("param", nn.initializers.uniform(), (self.params_dim,))
        means = (
            jax.vmap(jax.vmap(self.feature_fn, in_axes=(None, 0)), in_axes=(0, None))(
                obs, jnp.arange(self.n_actions)
            )
            @ params
        )
        if self.dist == "bernoulli":
            return jax.nn.sigmoid(means)
        elif self.dist == "normal":
            return means
        else:
            raise NotImplementedError

    def log_pdf(self, obs: chex.Array, action: int, reward: float) -> float:
        if self.dist == "bernoulli":
            mean_action_one = self(obs)[:, action]
            return jnp.log(
                mean_action_one * reward
                + (1.0 - mean_action_one) * (1.0 - reward)
                + self.eps
            ).sum()
        elif self.dist == "normal":
            return jax.scipy.stats.norm.logpdf(reward, self(obs)[:, action]).sum()
        else:
            raise NotImplementedError

    def reward_mean(self, obs: chex.Array, action: int):
        return self(obs)[:, action]

    def reward_mean_all(self, obs: chex.Array):
        return self(obs)

    def best_action(self, obs: chex.Array):
        return jnp.argmax(self(obs), axis=1)
