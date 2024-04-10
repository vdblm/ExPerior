import optax
import jax
import jax.numpy as jnp
import numpy as np

from typing import Any, Callable

from flax import core
from flax import struct


def linear_schedule_eps(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return jnp.maximum(slope * t + start_e, end_e)


# adapted from https://github.com/unstable-zeros/tasil
class PRNGSequence:
    def __init__(self, key_or_seed):
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        elif (
            hasattr(key_or_seed, "shape")
            and (not key_or_seed.shape)
            and hasattr(key_or_seed, "dtype")
            and key_or_seed.dtype == jnp.int32
        ):
            key_or_seed = jax.random.PRNGKey(key_or_seed)
        self._key = key_or_seed

    def __next__(self):
        k, n = jax.random.split(self._key)
        self._key = k
        return n


def moving_average(data: jnp.array, window_size: int):
    """Smooth data by calculating the moving average over a specified window size."""
    return jnp.convolve(data, jnp.ones(window_size) / window_size, mode="valid")


class VecTrainState(struct.PyTreeNode):
    """Train state to handle parallel updates."""

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    def apply_gradients(self, *, grads, **kwargs):
        """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

        Note that internally this function calls `.tx.update()` followed by a call
        to `optax.apply_updates()` to update `params` and `opt_state`.

        Args:
          grads: Gradients that have the same pytree structure as `.params`.
          **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

        Returns:
          An updated instance of `self` with `step` incremented by one, `params`
          and `opt_state` updated by applying `grads`, and additional attributes
          replaced as specified by `kwargs`.
        """
        updates, new_opt_state = jax.vmap(self.tx.update)(
            grads, self.opt_state, self.params
        )
        new_params = jax.vmap(optax.apply_updates)(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = jax.vmap(tx.init)(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


import matplotlib as mpl
import matplotlib.pyplot as plt


def latexify(fig_width, fig_height, font_size=7, legend_size=5, labelsize=7):
    """Set up matplotlib's RC params for LaTeX plotting."""
    params = {
        "backend": "ps",
        "text.latex.preamble": "\\usepackage{amsmath,amsfonts,amssymb,amsthm, mathtools,times}",
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "legend.fontsize": legend_size,
        "xtick.labelsize": labelsize,
        "ytick.labelsize": labelsize,
        "text.usetex": True,
        "figure.figsize": [fig_width, fig_height],
        "font.family": "serif",
        "xtick.minor.size": 0.5,
        "xtick.major.pad": 3,
        "xtick.minor.pad": 3,
        "xtick.major.size": 1,
        "ytick.minor.size": 0.5,
        "ytick.major.pad": 1.5,
        "ytick.major.size": 1,
    }

    mpl.rcParams.update(params)
    plt.rcParams.update(params)


COLORS = {
    "green": "#12f913",
    "blue": "#0000ff",
    "red": "#ff0000",
    "pink": "#fb87c4",
    "black": "#000000",
}

LIGHT_COLORS = {
    "blue": (0.237808, 0.688745, 1.0),
    "red": (1.0, 0.519599, 0.309677),
    "green": (0.0, 0.790412, 0.705117),
    "pink": (0.936386, 0.506537, 0.981107),
    "yellow": (0.686959, 0.690574, 0.0577502),
    "black": "#535154",
}

DARK_COLORS = {
    "green": "#3E9651",
    "red": "#CC2529",
    "blue": "#396AB1",
    "black": "#535154",
}

GOLDEN_RATIO = (np.sqrt(5) - 1.0) / 2

cm = 1 / 2.54
FIG_WIDTH = 17 * cm
FONT_SIZE = 10
LEGEND_SIZE = 8
