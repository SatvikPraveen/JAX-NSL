# File location: src/jax_nsl/transforms/control_flow.py

"""
Structured control flow: ``cond``, ``switch``, ``while_loop``, ``fori_loop``.

Rules that trip people up:

* Both branches of ``lax.cond`` must return the same pytree *structure*,
  shapes and dtypes - :func:`safe_cond` checks this eagerly and reports the
  mismatch instead of the generic tracer error.
* ``lax.cond`` under ``vmap`` becomes a ``select`` that evaluates *both*
  branches; do not rely on it to skip expensive or unsafe work.
* ``while_loop`` is not reverse-mode differentiable (dynamic trip count);
  use ``fori_loop`` with static bounds or ``scan`` when you need gradients,
  or implicit differentiation (see :mod:`jax_nsl.autodiff.implicit`).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax

from jax_nsl.core.numerics import safe_divide  # noqa: F401  (re-export)
from jax_nsl.core.numerics import softmax_stable as stable_softmax  # noqa: F401  (re-export)

Array = jax.Array


def _describe(tree: Any) -> str:
    return str(jax.tree_util.tree_map(lambda a: f"{a.dtype}{list(a.shape)}", tree))


def _check_same_structure(outputs: Sequence[Any], names: Sequence[str]) -> None:
    ref = outputs[0]
    ref_def = jax.tree_util.tree_structure(ref)
    ref_leaves = jax.tree_util.tree_leaves(ref)
    for out, name in zip(outputs[1:], names[1:]):
        if jax.tree_util.tree_structure(out) != ref_def:
            raise TypeError(f"Branch '{names[0]}' returns {_describe(ref)} but '{name}' returns "
                            f"{_describe(out)}: pytree structures differ.")
        for a, b in zip(ref_leaves, jax.tree_util.tree_leaves(out)):
            if a.shape != b.shape or a.dtype != b.dtype:
                raise TypeError(f"Branch '{names[0]}' returns {_describe(ref)} but '{name}' returns "
                                f"{_describe(out)}: shapes/dtypes differ (a common cause is a "
                                f"Python float in one branch and an int in the other).")


def safe_cond(pred: Union[bool, Array], true_fun: Callable, false_fun: Callable, *operands) -> Any:
    """``lax.cond`` that first checks the two branches agree in shape and dtype."""
    outs = [jax.eval_shape(true_fun, *operands), jax.eval_shape(false_fun, *operands)]
    _check_same_structure(outs, ["true_fun", "false_fun"])
    return lax.cond(pred, true_fun, false_fun, *operands)


def switch_case(index: Union[int, Array], branches: List[Callable], *operands) -> Any:
    """``lax.switch`` with the same eager structure check as :func:`safe_cond`.

    The index is clamped into range by ``lax.switch`` itself.
    """
    outs = [jax.eval_shape(b, *operands) for b in branches]
    _check_same_structure(outs, [f"branch[{i}]" for i in range(len(branches))])
    return lax.switch(index, branches, *operands)


def while_loop_safe(cond_fun: Callable, body_fun: Callable, init_val: Any,
                    max_iterations: Optional[int] = None) -> Any:
    """``while_loop`` with an optional iteration cap (guards against non-termination)."""
    if max_iterations is None:
        return lax.while_loop(cond_fun, body_fun, init_val)

    def cond(state):
        val, count = state
        return jnp.logical_and(cond_fun(val), count < max_iterations)

    def body(state):
        val, count = state
        return body_fun(val), count + 1

    final, _ = lax.while_loop(cond, body, (init_val, jnp.int32(0)))
    return final


bounded_while_loop = while_loop_safe


def for_loop(lower: int, upper: int, body_fun: Callable, init_val: Any, unroll: int = 1) -> Any:
    """``lax.fori_loop``; with static bounds it lowers to ``scan`` and is differentiable."""
    return lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)


def dynamic_slice_safe(operand: Array, start_indices: Sequence[Any], slice_sizes: Sequence[int]) -> Array:
    """``lax.dynamic_slice`` with start indices clamped so the slice stays in bounds.

    ``lax.dynamic_slice`` already clamps, silently; this version makes the
    behaviour explicit and works with a Python list of traced starts.
    """
    starts = [jnp.clip(jnp.asarray(s), 0, d - n)
              for s, d, n in zip(start_indices, operand.shape, slice_sizes)]
    return lax.dynamic_slice(operand, starts, slice_sizes)


def conditional_update(condition: Array, x: Array, update_fun: Callable, *args) -> Array:
    """``where(condition, update_fun(x, *args), x)`` - elementwise masked update.

    Both sides are always computed (there is no elementwise short-circuit on
    accelerators); make sure ``update_fun`` is finite on the unselected
    elements or the *gradient* will pick up NaNs through ``where``.
    """
    return jnp.where(condition, update_fun(x, *args), x)


def binary_search(f: Callable[[Array], Array], target: float, low: float, high: float,
                  tolerance: float = 1e-6, max_iterations: int = 100) -> Array:
    """Bisection for ``f(x) = target`` on a monotone ``f`` using ``while_loop``."""
    def cond(state):
        lo, hi, k = state
        return jnp.logical_and(jnp.abs(hi - lo) >= tolerance, k < max_iterations)

    def body(state):
        lo, hi, k = state
        mid = (lo + hi) / 2
        below = f(mid) < target
        return jnp.where(below, mid, lo), jnp.where(below, hi, mid), k + 1

    lo, hi, _ = lax.while_loop(cond, body, (jnp.asarray(low, jnp.float32),
                                            jnp.asarray(high, jnp.float32), jnp.int32(0)))
    return (lo + hi) / 2


def iterative_solver(f: Callable[[Array], Array], x0: Array, tolerance: float = 1e-6,
                     max_iterations: int = 100, damping: float = 1.0) -> Tuple[Array, Array]:
    """Damped fixed-point iteration ``x <- x + damping * (f(x) - x)``; returns ``(x, converged)``."""
    def cond(state):
        _, diff, k = state
        return jnp.logical_and(diff >= tolerance, k < max_iterations)

    def body(state):
        x, _, k = state
        x_new = x + damping * (f(x) - x)
        return x_new, jnp.linalg.norm(x_new - x), k + 1

    x, diff, _ = lax.while_loop(cond, body, (x0, jnp.asarray(jnp.inf, x0.dtype), jnp.int32(0)))
    return x, diff < tolerance


def select_n(pred: Array, on_true: Array, on_false: Array) -> Array:
    """``lax.select`` (no broadcasting, unlike ``jnp.where``)."""
    return lax.select(pred, on_true, on_false)


def gather_nd(params: Array, indices: Array) -> Array:
    """TensorFlow-style ``gather_nd``: ``indices[..., k]`` index the first ``k`` axes of ``params``."""
    k = indices.shape[-1]
    return params[tuple(jnp.moveaxis(indices, -1, 0))] if k > 0 else params


def scatter_add_nd(operand: Array, indices: Array, updates: Array) -> Array:
    """``operand.at[indices].add(updates)`` for ``indices`` of shape ``(n, k)``."""
    return operand.at[tuple(jnp.moveaxis(indices, -1, 0))].add(updates)


# ---------------------------------------------------------------------------
# Gradient shaping
# ---------------------------------------------------------------------------

def clip_gradient(x: Array, min_val: float = -1.0, max_val: float = 1.0) -> Array:
    """Identity forward; clips the incoming gradient elementwise in the backward pass."""
    from jax_nsl.autodiff.custom_vjp import clip_gradient_vjp

    return clip_gradient_vjp(x, min_val, max_val)


def clip_gradient_norm(fun: Callable, max_norm: float) -> Callable:
    """Transform ``fun`` so that ``grad(fun)`` has global norm at most ``max_norm``.

    Implemented as an identity on the *inputs* with a custom VJP that rescales
    the cotangent - so it composes with ``jit``, ``vmap`` and any optimiser.
    """
    @jax.custom_vjp
    def clipped_identity(x):
        return x

    def fwd(x):
        return x, None

    def bwd(_, g):
        leaves = jax.tree_util.tree_leaves(g)
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
        factor = jnp.minimum(1.0, max_norm / (norm + 1e-12))
        return (jax.tree_util.tree_map(lambda leaf: leaf * factor, g),)

    clipped_identity.defvjp(fwd, bwd)

    def wrapped(x, *args, **kwargs):
        return fun(clipped_identity(x), *args, **kwargs)

    return wrapped
