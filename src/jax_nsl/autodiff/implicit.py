# File location: src/jax_nsl/autodiff/implicit.py

"""
Implicit differentiation: gradients through solvers without unrolling them.

If ``x*`` is defined implicitly by ``x* = f(params, x*)`` (a fixed point) then
by the implicit function theorem

    dx*/dparams = (I - df/dx)^{-1} df/dparams          (evaluated at x*)

so the VJP of ``params -> x*`` with cotangent ``g`` is

    u = (I - df/dx)^{-T} g,      dL/dparams = u^T df/dparams

and ``u`` is itself the fixed point of the *linear* map ``u -> g + (df/dx)^T u``,
which we solve with the same iteration.  Memory and compute are independent
of how many forward iterations were needed, and the forward solver may use
any stopping rule (``while_loop``) because nothing is differentiated through it.

This is the pattern behind deep equilibrium models, differentiable
optimisation layers and the implicit CG gradient in
:mod:`jax_nsl.linalg.solvers`.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


def _tree_norm(t: Any) -> Array:
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in jax.tree_util.tree_leaves(t)))


def _iterate_to_fixed_point(step: Callable[[Any], Any], x0: Any, tolerance: float,
                            max_iterations: int) -> Tuple[Any, Array]:
    """Run ``x <- step(x)`` until ``||x_new - x|| < tolerance``; returns ``(x, iters)``."""
    def cond(state):
        x, x_prev, k = state
        return jnp.logical_and(_tree_norm(jax.tree_util.tree_map(jnp.subtract, x, x_prev)) >= tolerance,
                               k < max_iterations)

    def body(state):
        x, _, k = state
        return step(x), x, k + 1

    x1 = step(x0)
    x, _, k = lax.while_loop(cond, body, (x1, x0, jnp.int32(1)))
    return x, k


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 3, 4))
def fixed_point(f: Callable[[Any, Any], Any], params: Any, x_init: Any,
                tolerance: float = 1e-6, max_iterations: int = 1000) -> Any:
    """Solve ``x = f(params, x)`` and differentiate the solution implicitly.

    Args:
        f: Contraction map ``(params, x) -> x`` (any pytree ``x``).
        params: Differentiable parameters of ``f``.
        x_init: Starting point (not differentiated).
        tolerance: Stop when successive iterates differ by less than this.
        max_iterations: Iteration cap for both forward and backward solves.

    Returns:
        The fixed point ``x*`` with a custom VJP w.r.t. ``params``.
    """
    x_star, _ = _iterate_to_fixed_point(lambda x: f(params, x), x_init, tolerance, max_iterations)
    return x_star


def _fixed_point_fwd(f, params, x_init, tolerance, max_iterations):
    x_star, _ = _iterate_to_fixed_point(lambda x: f(params, x), x_init, tolerance, max_iterations)
    return x_star, (params, x_star)


def _fixed_point_bwd(f, tolerance, max_iterations, residuals, g):
    params, x_star = residuals
    _, vjp_x = jax.vjp(lambda x: f(params, x), x_star)
    _, vjp_params = jax.vjp(lambda p: f(p, x_star), params)

    # u = g + (df/dx)^T u  - a linear fixed point solved by the same iteration.
    def linear_step(u):
        return jax.tree_util.tree_map(jnp.add, g, vjp_x(u)[0])

    u, _ = _iterate_to_fixed_point(linear_step, g, tolerance, max_iterations)
    (g_params,) = vjp_params(u)
    return g_params, jax.tree_util.tree_map(jnp.zeros_like, x_star)


fixed_point.defvjp(_fixed_point_fwd, _fixed_point_bwd)


def fixed_point_unrolled(f: Callable[[Any, Any], Any], params: Any, x_init: Any,
                         num_iterations: int) -> Any:
    """Same solve with a fixed iteration count via ``scan`` - differentiable by unrolling.

    Provided for comparison: the gradient is exact for the *truncated*
    iteration (not the true fixed point) and memory grows with
    ``num_iterations`` because every iterate is saved for the backward pass.
    """
    def body(x, _):
        return f(params, x), None

    x, _ = lax.scan(body, x_init, None, length=num_iterations)
    return x


def implicit_newton_solve(residual: Callable[[Any, Array], Array], params: Any, x_init: Array,
                          tolerance: float = 1e-8, max_iterations: int = 50) -> Array:
    """Root of ``residual(params, x) = 0`` by Newton's method with an implicit gradient.

    Uses :func:`fixed_point` on the Newton map ``x -> x - J^{-1} r(x)``; since
    the map's fixed point is the root, the implicit VJP is exactly the
    implicit-function-theorem gradient ``-J_x^{-1} J_params``.  ``x`` must be
    a 1-D array (dense Jacobian solve).
    """
    def newton_step(p, x):
        r = residual(p, x)
        J = jax.jacfwd(lambda y: residual(p, y))(x)
        return x - jnp.linalg.solve(J, r)

    return fixed_point(newton_step, params, x_init, tolerance, max_iterations)
