"""
Code adapted from
https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/misc/bernoulli_bandit.py
"""

import chex
from chex._src.pytypes import Array, PRNGKey
import jax
import jax.numpy as jnp

from jax import lax
from typing import Tuple

from typing import Tuple, Callable, Union, Optional
from flax import struct
from gymnax.environments import EnvParams, EnvState, spaces
from gymnax.environments.environment import EnvParams
from .utils import Environment


Param = chex.Array
MetaParam = chex.Array
Context = chex.Array
Action = Union[int, float, chex.Array]


@struct.dataclass
class EnvState:
    current_context: Context
    time: float


@struct.dataclass
class EnvParams:
    reward_param: Param = None


class BayesStochasticBandit(Environment):
    def __init__(
        self,
        action_space: spaces.Space,
        prior_fn: Callable[[chex.PRNGKey, Optional[MetaParam]], Param],
        reward_dist_fn: Callable[[chex.PRNGKey, Param, Context, Action], float],
        reward_mean_fn: Callable[[Param, Context, Action], float],
        best_action_value_fn: Callable[[Param, Context], Tuple[Action, float]],
        init_context_dist_fn: Callable[[chex.PRNGKey], Context] = jax.tree_util.Partial(
            lambda k: jnp.array([0.0])
        ),
    ):
        super().__init__()
        self.this_action_space = action_space
        self.best_action_value_fn = best_action_value_fn
        self.prior_fn = prior_fn
        self.reward_dist_fn = reward_dist_fn
        self.reward_mean_fn = reward_mean_fn
        self.init_context_dist_fn = init_context_dist_fn

    @property
    def default_params(self) -> EnvParams:
        # Default environment parameters
        return EnvParams(reward_param=self.prior_fn(jax.random.PRNGKey(42), None))

    def init_env(
        self, key: PRNGKey, params: EnvParams, meta_params: MetaParam = None
    ) -> EnvParams:
        reward_param = self.prior_fn(key, meta_params)
        return params.replace(reward_param=reward_param)

    def optimal_value(self, state: EnvState, params: EnvParams) -> float:
        return self.best_action_value_fn(params.reward_param, state.current_context)[1]

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: Action, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, dict]:
        """Sample context, reward, increase counter, construct input."""
        key, k1, k2 = jax.random.split(key, 3)
        context = self.init_context_dist_fn(k1)
        reward = self.reward_dist_fn(k2, params.reward_param, context, action)
        state = EnvState(context, state.time + 1)
        done = self.is_terminal(state, params)
        return (
            lax.stop_gradient(self.get_obs(state, params)),
            lax.stop_gradient(state),
            reward,
            done,
            {
                "reward_mean": self.reward_mean_fn(
                    params.reward_param, context, action
                ),
                "optimal_value": self.optimal_value(state, params),
                "optimal_action": self.optimal_policy(key, state, params),
            },
        )

    def optimal_policy(
        self, key: PRNGKey, state: EnvState, params: EnvParams
    ) -> Action:
        return self.best_action_value_fn(params.reward_param, state.current_context)[0]

    def q_function(self, state: EnvState, param: EnvParams, action: Action) -> float:
        return self.reward_mean_fn(param.reward_param, state.current_context, action)

    @property
    def num_actions(self) -> int:
        if isinstance(self.this_action_space, spaces.Discrete):
            return self.this_action_space.n
        else:
            raise NotImplementedError

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset environment state by sampling initial position."""
        context = self.init_context_dist_fn(key)
        state = EnvState(context, 0.0)
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params: EnvParams) -> chex.Array:
        """Concatenate context, reward, action and time stamp."""
        return state.current_context

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        """Check whether state is terminal."""
        # Horizon is 1 in bandits
        done = state.time >= 1
        return done

    @property
    def name(self) -> str:
        """Environment name."""
        return "BayesMultiArmedBandit"

    def action_space(self, params: EnvParams) -> spaces.Space:
        return self.this_action_space

    def observation_space(self, params: EnvParams) -> spaces.Space:
        raise NotImplementedError

    def state_space(self, params: EnvParams) -> spaces.Space:
        raise NotImplementedError
