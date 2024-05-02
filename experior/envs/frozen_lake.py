from gymnax.environments.environment import EnvParams, EnvState
import jax
import jax.numpy as jnp
import chex

from jax import lax
from gymnax.environments import EnvParams, EnvState, spaces
from typing import Tuple, Optional, Callable
from flax import struct

from .utils import Environment

Param = chex.Array
MetaParam = chex.Array


@struct.dataclass
class EnvState:
    row: int
    column: int
    hazard_row: int
    hazard_column: int
    time: int


@struct.dataclass
class EnvParams:
    hazard_param: Param
    max_steps_in_episode: int = 100


GOAL_LOC = lambda size: (size // 2, size - 1)
START_LOC = lambda size: (size // 2, 0)

# TODO add image-based observation
# TODO stochastic version of the environment (might change the q-values)


class FrozenLake(Environment):
    """
    JAX Compatible version of FrozenLake environment in https://arxiv.org/pdf/2012.15566.
    The agent receives a reward of -2, 20, and -100 for taking a step, reaching the goal, a
    nd falling into the hazard. Actions are 0, 1, 2, 3 for left, down, right, up.
    The (0, 0) coord is at the top left corner of the grid. The starting position of the
    agent is (size//2, 0) and the goal is at (size//2, size-1).
    """

    def __init__(
        self,
        size: int,
        partial_observe: bool,
        hazard_prior_fn: Callable[
            [chex.PRNGKey, Optional[MetaParam]], Param
        ],  # only used in init_env (run once)
        hazard_dist_fn: Callable[
            [chex.PRNGKey, Param], Tuple[int, int]
        ],  # used in reset_env (for any new episode)
    ):
        assert size >= 5, "Size of the grid should be at least 5x5"
        super().__init__()
        self.size = size
        self.partial_observe = partial_observe
        self.hazard_prior_fn = hazard_prior_fn
        self.hazad_dist_fn = hazard_dist_fn

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(
            hazard_param=self.hazard_prior_fn(jax.random.PRNGKey(42), None)
        )

    def init_env(
        self, key: chex.PRNGKey, params: EnvParams, meta_params: chex.Array = None
    ) -> EnvParams:
        hazard_param = self.hazard_prior_fn(key, meta_params)
        return params.replace(hazard_param=hazard_param)

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        hazard_row, hazard_column = self.hazad_dist_fn(key, params.hazard_param)
        agent_row, agent_column = START_LOC(self.size)
        state = EnvState(
            row=agent_row,
            column=agent_column,
            hazard_row=hazard_row,
            hazard_column=hazard_column,
            time=0,
        )
        return self.get_obs(state), state

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ):
        # making the transition
        row, column = step_transition(state, action, self.size)
        state = state.replace(row=row, column=column, time=state.time + 1)

        # calculating the reward after taking the action to the new state
        reward = step_reward(state, self.size)

        # done condition
        done = self.is_terminal(state, params)

        info = {}

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            reward,
            done,
            info,
        )

    def is_terminal(self, state: EnvState, params: EnvParams) -> bool:
        hazard_reach = hazard_reached(state)
        goal_reach = goal_reached(state, self.size)

        done_step = state.time >= params.max_steps_in_episode
        return hazard_reach | goal_reach | done_step

    def get_obs(self, state: EnvState) -> chex.Array:
        obs = jnp.zeros((self.size, self.size, 2))
        obs.at[state.row, state.column, 0].set(1.0)
        obs = jax.lax.select(
            self.partial_observe,
            obs,
            obs.at[state.hazard_row, state.hazard_column, 1].set(1.0),
        )
        return obs

    def optimal_value(self, state: EnvState, params: EnvParams) -> float:
        # TODO assumes there is only one hazard cell and it is in the middle square
        goal_row, goal_column = GOAL_LOC(self.size)
        row_diff = jnp.abs(state.row - goal_row)
        col_diff = jnp.abs(state.column - goal_column)

        hazard_in_row = (row_diff == 0) & (state.hazard_row == goal_row)
        hazard_in_col = (col_diff == 0) & (state.hazard_column == goal_column)

        num_steps = row_diff + col_diff
        num_steps += jax.lax.select(hazard_in_row | hazard_in_col, 2, 0)
        value = -2.0 * num_steps + 20.0
        is_done = self.is_terminal(state, params)
        return jax.lax.select(is_done, 0.0, value)

    def q_values(self, state: EnvState, params: EnvParams) -> chex.Array:
        # TODO can be more efficient
        q_values = jnp.zeros(self.num_actions)
        is_done = self.is_terminal(state, params)
        for action in range(self.num_actions):
            row, column = step_transition(state, action, self.size)
            new_state = state.replace(row=row, column=column)
            q_values = q_values.at[action].set(
                step_reward(new_state, self.size)
                + self.optimal_value(new_state, params)
            )
        return jax.lax.select(is_done, jnp.zeros(self.num_actions), q_values)

    def q_function(self, state: EnvState, param: EnvParams, action: int) -> float:
        return self.q_values(state, param)[action]

    def optimal_policy(
        self, key: chex.PRNGKey, state: EnvState, params: EnvParams
    ) -> int:
        q_values = self.q_values(state, params)
        return jnp.argmax(q_values)

    @property
    def name(self) -> str:
        return "Frozen Lake"

    @property
    def num_actions(self) -> int:
        return 4

    def action_space(self, params: Optional[EnvParams]) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # agent position, hazard position
        return spaces.Box(0, 1, (self.size, self.size, 2), jnp.float32)


def step_reward(state: EnvState, size: int) -> float:
    """Get the reward for the selected action."""
    # Reward calculation.
    hazard_reach = hazard_reached(state)
    goal_reach = goal_reached(state, size)
    reward = -2.0  # taking action
    reward += 20.0 * goal_reach
    reward += -100.0 * hazard_reach
    return reward


def step_transition(state: EnvState, action: int, size: int) -> Tuple[int, int]:
    column_change = (2 * (action // 2) - 1) * (1 - action % 2)
    row_change = (2 * (1 - action // 2) - 1) * (action % 2)
    new_row, new_column = state.row + row_change, state.column + column_change
    in_map_cond = (
        (new_row >= 0) & (new_row < size) & (new_column >= 0) & (new_column < size)
    )
    row, column = jax.lax.cond(
        in_map_cond,
        lambda _: (new_row, new_column),
        lambda _: (state.row, state.column),
        None,
    )

    return row, column


def goal_reached(state: EnvState, size) -> bool:
    goal_row, goal_col = GOAL_LOC(size)
    return (state.row == goal_row) & (state.column == goal_col)


def hazard_reached(state: EnvState) -> bool:
    return (state.row == state.hazard_row) & (state.column == state.hazard_column)
