from gymnax.environments.environment import EnvParams, EnvState
import jax
import jax.numpy as jnp
import chex

import numpy as np

from jax import lax
from gymnax.environments import EnvParams, EnvState, spaces
from typing import Tuple, Optional, Callable
from flax import struct
from functools import partial

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

NUM_ACTIONS = 4

W = 6
WHITE = np.array([[241, 240, 255.0]]) / 255.0
GREY = np.array([[90, 90, 90.0]]) / 255.0
WHITE_GREY = np.array([[185, 185, 185.0]]) / 255.0


def gen_grid_image(size: int):
    width = (size + 2) * W
    img = np.zeros((width, width, 3)) + GREY
    img[W:-W, W:-W] = WHITE
    img[W:-W:W, W:-W] = WHITE_GREY
    img[W:-W, W:-W:W] = WHITE_GREY

    # plot the goal
    goal_row, goal_col = GOAL_LOC(size)
    img[
        (goal_row + 1) * W : (goal_row + 2) * W, (goal_col + 1) * W : (goal_col + 2) * W
    ] = [0, 1, 0]
    img = np.clip(img + 0.05 * np.random.normal(size=img.shape), 0, 1)
    return jnp.array(img)


def gen_agent_locs(size: int):
    width = (size + 2) * W
    agent_locs = np.zeros((size, size, width, width, 3))
    for i in range(size):
        for j in range(size):
            red_agent = np.zeros((width, width, 3))
            red_agent[
                (i + 1) * W + 1 : (i + 2) * W - 1, (j + 1) * W : (j + 1) * W + W // 3
            ] = (np.array([0.9, 0, 0]) - WHITE)
            red_agent[
                (i + 1) * W + 2 : (i + 2) * W - 2,
                (j + 1) * W + W // 3 : (j + 1) * W + 2 * W // 3,
            ] = (
                np.array([0.9, 0, 0]) - WHITE
            )
            agent_locs[i, j] = red_agent

    return jnp.array(agent_locs)


def gen_hazard_locs(size: int):
    width = (size + 2) * W
    hazard_locs = np.zeros((size, size, width, width, 3))
    for i in range(size):
        for j in range(size):
            blue_hazard = np.zeros((width, width, 3))
            blue_hazard[(i + 1) * W : (i + 2) * W, (j + 1) * W : (j + 2) * W] = (
                np.array([0.0, 0, 0.9]) - WHITE
            )
            hazard_locs[i, j] = blue_hazard

    return jnp.array(hazard_locs)


GOAL_REWARD = 40.0
HAZARD_REWARD = -100.0
BORDER_REWARD = -3.0
NON_CENTER_LAKE_REWARD = -5.0
CENTER_LAKE_REWARD = -10.0

# TODO refactor the code


class FrozenLake(Environment):
    """
    JAX Compatible version of FrozenLake environment in https://arxiv.org/pdf/2012.15566.
    Actions are 0, 1, 2, 3 for left, down, right, up. The (0, 0) coord is at the
    top left corner of the grid. The starting position of the agent is (size//2, 0)
    and the goal is at (size//2, size-1). The agent receives a reward of 40 for reaching the goal,
    and a reward of -100 for falling into the hazard. Moreover, the agent will receive a reward
    of -3 for moving onto the non-lake squares (border), -5 for moving onto the non-central
    lake squares, and -10 for moving onto the central lake square. The rewards are similar
    to the implementation in https://arxiv.org/pdf/2402.08733.
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
        self.background = gen_grid_image(size)
        self.agent_locs = gen_agent_locs(size)
        self.hazard_locs = gen_hazard_locs(size)

        value_iter_fn = lambda hazard_row, hazard_column: value_iteration(
            size=size, steps=10000, hazard_row=hazard_row, hazard_column=hazard_column
        )
        self.values = jax.vmap(jax.vmap(value_iter_fn, (None, 0)), (0, None))(
            jnp.arange(size), jnp.arange(size)
        )

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
        row, column = step_transition(
            state.row,
            state.column,
            action,
            state.hazard_row,
            state.hazard_column,
            self.size,
        )

        # calculating the reward
        reward = reward_fn(
            state.row,
            state.column,
            action,
            state.hazard_row,
            state.hazard_column,
            self.size,
        )

        state = state.replace(row=row, column=column, time=state.time + 1)

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
        hazard_reach = hazard_reached(
            state.row, state.column, state.hazard_row, state.hazard_column
        )
        goal_reach = goal_reached(state.row, state.column, self.size)

        done_step = state.time >= params.max_steps_in_episode
        return hazard_reach | goal_reach | done_step

    def get_obs(self, state: EnvState) -> chex.Array:
        obs = jnp.zeros((self.size, self.size, 2))
        obs = obs.at[state.row, state.column, 0].set(1.0)
        obs = jax.lax.select(
            self.partial_observe,
            obs,
            obs.at[state.hazard_row, state.hazard_column, 1].set(1.0),
        )
        return obs

    def render(self, obs: chex.Array) -> chex.Array:
        try:
            obs = obs.reshape((self.size, self.size, 2))
        except:
            raise ValueError("Invalid observation shape")
        # this is to remove the first index, which has the size of 1
        agent_img = self.agent_locs[jnp.where(obs[:, :, 0] == 1.0, size=1)][0]
        hazard_img = jax.lax.select(
            self.partial_observe,
            jnp.zeros(shape=agent_img.shape),
            self.hazard_locs[jnp.where(obs[:, :, 1] == 1.0, size=1)][0],
        )
        return self.background + agent_img + hazard_img

    def optimal_value(self, state: EnvState, params: EnvParams) -> float:
        return self.values[state.hazard_row, state.hazard_column][
            state.row, state.column
        ]

    def q_values(self, state: EnvState, params: EnvParams) -> chex.Array:
        this_reward_fn = lambda action: reward_fn(
            state.row,
            state.column,
            action,
            state.hazard_row,
            state.hazard_column,
            self.size,
        )

        value_fn = lambda action: self.values[state.hazard_row, state.hazard_column][
            step_transition(
                state.row,
                state.column,
                action,
                state.hazard_row,
                state.hazard_column,
                self.size,
            )
        ]
        rewards = jax.vmap(this_reward_fn)(jnp.arange(NUM_ACTIONS))
        next_values = jax.vmap(value_fn)(jnp.arange(NUM_ACTIONS))
        return rewards + next_values

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
        return NUM_ACTIONS

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        # agent position, hazard position
        return spaces.Box(0, 1, (self.size, self.size, 2), jnp.float32)


def step_reward(
    state_row: int, state_column: int, hazard_row: int, hazard_column: int, size: int
) -> float:
    """Get the reward for the selected action."""
    # Reward calculation.
    hazard_reach = hazard_reached(state_row, state_column, hazard_row, hazard_column)
    goal_reach = goal_reached(state_row, state_column, size)
    is_center = is_center_lake(state_row, state_column, size)
    step_reward = jax.lax.select(
        is_non_lake(state_row, state_column, size),
        BORDER_REWARD,
        NON_CENTER_LAKE_REWARD * (1 - is_center) + CENTER_LAKE_REWARD * is_center,
    )
    reward = GOAL_REWARD * goal_reach + (1 - goal_reach) * step_reward
    reward = HAZARD_REWARD * hazard_reach + (1 - hazard_reach) * reward
    return reward


def reward_fn(row, column, action, hazard_row, hazard_column, size):
    next_row, next_col = step_transition(
        row, column, action, hazard_row, hazard_column, size
    )
    next_reward = step_reward(next_row, next_col, hazard_row, hazard_column, size)
    terminal = goal_reached(row, column, size) | hazard_reached(
        row, column, hazard_row, hazard_column
    )
    return jax.lax.select(
        terminal,
        0.0,
        next_reward,
    )


def step_transition(
    state_row: int,
    state_column: int,
    action: int,
    hazard_row: int,
    hazard_column: int,
    size: int,
) -> Tuple[int, int]:
    hazard_reach = hazard_reached(state_row, state_column, hazard_row, hazard_column)
    goal_reach = goal_reached(state_row, state_column, size)
    column_change = (2 * (action // 2) - 1) * (1 - action % 2)
    row_change = (2 * (1 - action // 2) - 1) * (action % 2)
    new_row, new_column = state_row + row_change, state_column + column_change
    in_map_cond = (
        (new_row >= 0) & (new_row < size) & (new_column >= 0) & (new_column < size)
    )
    not_terminal_cond = jnp.logical_not(hazard_reach | goal_reach)
    row, column = jax.lax.cond(
        in_map_cond & not_terminal_cond,
        lambda _: (new_row, new_column),
        lambda _: (state_row, state_column),
        None,
    )

    return row, column


def goal_reached(state_row: int, state_column: int, size) -> bool:
    goal_row, goal_col = GOAL_LOC(size)
    return (state_row == goal_row) & (state_column == goal_col)


def hazard_reached(
    state_row: int, state_column: int, hazard_row: int, hazard_column: int
) -> bool:
    return (state_row == hazard_row) & (state_column == hazard_column)


def is_non_lake(state_row: int, state_column: int, size: int) -> bool:
    return (
        (state_row == 0)
        | (state_row == size - 1)
        | (state_column == 0)
        | (state_column == size - 1)
    )


def is_center_lake(state_row: int, state_column: int, size: int) -> bool:
    center_row, center_col = size // 2, size // 2
    return (state_row == center_row) & (state_column == center_col)


@partial(jax.jit, static_argnums=(0, 1))
def value_iteration(
    size: int, steps: int, hazard_row: int, hazard_column: int, discount: float = 1.0
):

    this_reward_fn = lambda row, column, action: reward_fn(
        row, column, action, hazard_row, hazard_column, size
    )
    # size x size x NUM_ACTIONS
    rewards = jax.vmap(
        jax.vmap(
            jax.vmap(this_reward_fn, in_axes=(None, None, 0)), in_axes=(None, 0, None)
        ),
        in_axes=(0, None, None),
    )(jnp.arange(size), jnp.arange(size), jnp.arange(NUM_ACTIONS))

    next_state_fn = lambda row, column, action: jnp.array(
        step_transition(row, column, action, hazard_row, hazard_column, size)
    )

    # size x size x NUM_ACTIONS x 2
    next_states = jax.vmap(
        jax.vmap(
            jax.vmap(next_state_fn, in_axes=(None, None, 0)),
            in_axes=(None, 0, None),
        ),
        in_axes=(0, None, None),
    )(jnp.arange(size), jnp.arange(size), jnp.arange(NUM_ACTIONS))

    def value_iteration_step(values: chex.Array, _):
        def target_value_fn(row, col, action):
            next_row, next_col = next_states[row, col, action]
            return rewards[row, col, action] + discount * values[next_row, next_col]

        # size x size x NUM_ACTIONS
        next_q_values = jax.vmap(
            jax.vmap(
                jax.vmap(target_value_fn, in_axes=(None, None, 0)),
                in_axes=(None, 0, None),
            ),
            in_axes=(0, None, None),
        )(jnp.arange(size), jnp.arange(size), jnp.arange(NUM_ACTIONS))
        return jnp.max(next_q_values, axis=-1), None  # size x size

    values = jnp.zeros((size, size))
    new_values, _ = jax.lax.scan(value_iteration_step, values, None, steps)
    return new_values
