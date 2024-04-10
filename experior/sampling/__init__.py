import jax
import chex

import jax.numpy as jnp

from typing import Callable, Any


def langevin_sampling(
    rng: chex.PRNGKey,
    init_param: Any,
    log_pdf: Callable[[Any], chex.Array],
    step_size: float,
    num_steps: int,
    grad_opt: Callable[[Any], Any],
):
    """Langevin dynamics sampling of a distribution.

    Args:
        rng: Jax random key.
        init_param: Initial parameter.
        log_pdf: A function that returns the (unnormalized) log pdf of a parameter.
        step_size: The initial step size of the Langevin dynamics.
        num_steps: The number of steps of the Langevin dynamics.
        grad_opt: A function that transforms the gradient, e.g., to clip it.

    Returns:
        The updated parameter.
    """

    # From https://icml.cc/2011/papers/398_icmlpaper.pdf
    def _update_parameters(state, i):
        param, rng = state
        # Compute the gradient of the unnormalized log prior
        grad_log_prior = jax.grad(lambda p: log_pdf(p).sum())(param)

        # write a clip the gradient
        grad_log_prior = grad_opt(grad_log_prior)
        step = step_size / i

        # Langevin dynamics update rule
        rng, rng_ = jax.random.split(rng)
        rng, rng_ = jax.random.split(rng)
        num_vars = len(jax.tree_util.tree_leaves(grad_log_prior))
        treedef = jax.tree_util.tree_structure(grad_log_prior)
        updated_param = jax.tree_util.tree_map(
            lambda p, g, k: p
            + step * g
            + jnp.sqrt(2 * step) * jax.random.normal(k, shape=g.shape, dtype=g.dtype),
            param,
            grad_log_prior,
            jax.tree_util.tree_unflatten(treedef, jax.random.split(rng_, num_vars)),
        )
        state = (updated_param, rng)
        return state, updated_param

    last_updated_param, all_params = jax.lax.scan(
        _update_parameters,
        (init_param, rng),
        jnp.arange(1, num_steps + 1),
    )
    return last_updated_param[0], all_params
