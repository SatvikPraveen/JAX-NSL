# File location: src/jax_nsl/core/numerics.py

"""
Numerically stable operations: logsumexp, softmax, clipping, and safe math.

Every function here exists because the naive formula is wrong in floating
point for *some* input range.  The docstrings explain which range and why the
stable version works, so this module doubles as a reference for the
techniques (max-shifting, the "double where" trick, dtype-aware step sizes).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal, overload

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array
Axis = int | tuple[int, ...] | None


# ---------------------------------------------------------------------------
# Elementwise safe math
# ---------------------------------------------------------------------------


def safe_log(x: Array, eps: float = 1e-8) -> Array:
    """``log(max(x, eps))`` so that zeros (and small negatives) never give -inf/NaN.

    Note that the gradient is exactly zero wherever ``x < eps``; if you need a
    non-zero gradient there, see :func:`jax_nsl.autodiff.custom_vjp.safe_log_vjp`.
    """
    return jnp.log(jnp.maximum(x, eps))


def safe_exp(x: Array, max_val: float | None = None) -> Array:
    """``exp(min(x, max_val))`` - clips the argument so the result never overflows.

    The default clip is ``log(finfo.max) - 1`` for the input dtype.
    """
    if max_val is None:
        max_val = float(jnp.log(jnp.finfo(jnp.result_type(x)).max)) - 1.0
    return jnp.exp(jnp.minimum(x, max_val))


def safe_sqrt(x: Array, eps: float = 0.0) -> Array:
    """``sqrt(max(x, eps))`` - clamps tiny negatives produced by round-off.

    With ``eps=0`` the gradient at 0 is still infinite (``1/(2*sqrt(0))``);
    pass a small positive ``eps`` when the value is used inside a loss.
    """
    return jnp.sqrt(jnp.maximum(x, eps))


def safe_divide(x: Array, y: Array, eps: float = 1e-8, replace_nan: bool = True) -> Array:
    """``x / (y + eps)`` with non-finite results replaced by 0.

    Args:
        x: Numerator.
        y: Denominator.
        eps: Added to the denominator (use a *signed* eps yourself if ``y`` can
            be negative and close to zero).
        replace_nan: Replace ``inf``/``nan`` results with 0.
    """
    result = x / (y + eps)
    if replace_nan:
        result = jnp.where(jnp.isfinite(result), result, 0.0)
    return result


def stable_sigmoid(x: Array) -> Array:
    """Sigmoid that is finite *and has finite gradients* for every float input.

    The textbook piecewise form::

        where(x >= 0, 1 / (1 + exp(-x)), exp(x) / (1 + exp(x)))

    is finite in the forward pass, but ``jnp.where`` still evaluates *both*
    branches, so for ``x = -1000`` the unused ``exp(-x)`` overflows to ``inf``
    and its derivative becomes ``inf * 0 = nan``.  The "double where" trick
    feeds each branch an argument that is always safe::

        z = where(x >= 0, -x, x)     # z <= 0, so exp(z) <= 1
        e = exp(z)
        where(x >= 0, 1 / (1 + e), e / (1 + e))
    """
    z = jnp.where(x >= 0, -x, x)
    e = jnp.exp(z)
    return jnp.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def stable_tanh(x: Array) -> Array:
    """``tanh`` via ``2 * sigmoid(2x) - 1`` using :func:`stable_sigmoid`."""
    return 2.0 * stable_sigmoid(2.0 * x) - 1.0


# ---------------------------------------------------------------------------
# Log-sum-exp family
# ---------------------------------------------------------------------------


@overload
def logsumexp_stable(
    x: Array, axis: Axis = ..., keepdims: bool = ..., return_max: Literal[False] = ...
) -> Array: ...


@overload
def logsumexp_stable(
    x: Array, axis: Axis = ..., keepdims: bool = ..., return_max: Literal[True] = ...
) -> tuple[Array, Array]: ...


def logsumexp_stable(
    x: Array, axis: Axis = None, keepdims: bool = False, return_max: bool = False
) -> Array | tuple[Array, Array]:
    """``log(sum(exp(x)))`` computed as ``m + log(sum(exp(x - m)))`` with ``m = max(x)``.

    Subtracting the max guarantees the largest exponent is ``exp(0) = 1``, so
    nothing overflows, and at least one term is not underflowed.  The max is
    wrapped in ``stop_gradient``: mathematically the shift cancels out of the
    derivative, so we avoid tracing a useless gradient path through ``max``.

    Rows that are entirely ``-inf`` return ``-inf`` (not NaN).

    Args:
        x: Input array.
        axis: Axis or axes to reduce over.
        keepdims: Keep reduced axes as size-1 dimensions.
        return_max: Also return the shift ``m`` (with the same ``keepdims``).
    """
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_max = lax.stop_gradient(jnp.where(jnp.isfinite(x_max), x_max, 0.0))
    result = x_max + jnp.log(jnp.sum(jnp.exp(x - x_max), axis=axis, keepdims=True))

    if not keepdims:
        result = jnp.squeeze(result, axis=axis)
        x_max = jnp.squeeze(x_max, axis=axis)
    if return_max:
        return result, x_max
    return result


def softmax_stable(x: Array, axis: int = -1, temperature: float = 1.0) -> Array:
    """Softmax with the max subtracted before exponentiating.

    Args:
        x: Logits.
        axis: Axis along which probabilities sum to one.
        temperature: Divides the logits; ``> 1`` flattens, ``< 1`` sharpens.
    """
    x = x / temperature
    x_shifted = x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))
    exp_x = jnp.exp(x_shifted)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)


def log_softmax_stable(x: Array, axis: int = -1, temperature: float = 1.0) -> Array:
    """Log-softmax computed as ``shifted - log(sum(exp(shifted)))``.

    This is *not* the same as ``x - logsumexp(x)`` in floating point.  For
    logits around 1000, ``logsumexp`` is ~1000.4 and float32 only resolves it
    to about 6e-5, so ``x - logsumexp(x)`` loses four digits of the answer.
    Shifting first keeps every intermediate O(1).
    """
    x = x / temperature
    shifted = x - lax.stop_gradient(jnp.max(x, axis=axis, keepdims=True))
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True))


def smooth_max(x: Array, axis: Axis = None, alpha: float = 1.0) -> Array:
    """Smooth maximum ``logsumexp(alpha * x) / alpha`` (larger ``alpha`` -> closer to max)."""
    return logsumexp_stable(alpha * x, axis=axis) / alpha


def smooth_min(x: Array, axis: Axis = None, alpha: float = 1.0) -> Array:
    """Smooth minimum ``-smooth_max(-x)``."""
    return -smooth_max(-x, axis=axis, alpha=alpha)


def gumbel_softmax(
    logits: Array, temperature: float, key: Array, axis: int = -1, hard: bool = False
) -> Array:
    """Gumbel-softmax relaxation of a categorical sample.

    Args:
        logits: Unnormalised log-probabilities.
        temperature: Relaxation temperature; ``-> 0`` approaches one-hot.
        key: PRNG key for the Gumbel noise.
        axis: Category axis.
        hard: Return a one-hot sample in the forward pass while keeping the
            soft gradient (straight-through estimator).
    """
    u = jax.random.uniform(key, logits.shape, minval=jnp.finfo(logits.dtype).tiny, maxval=1.0)
    gumbel = -jnp.log(-jnp.log(u))
    y = softmax_stable((logits + gumbel) / temperature, axis=axis)
    if hard:
        y_hard = jax.nn.one_hot(
            jnp.argmax(y, axis=axis), logits.shape[axis], axis=axis, dtype=y.dtype
        )
        y = y_hard - lax.stop_gradient(y) + y
    return y


# ---------------------------------------------------------------------------
# Pytree norms and clipping
# ---------------------------------------------------------------------------


def safe_norm(tree: Any, ord: int | float | str | None = None) -> Array:
    """Global norm over every leaf of a pytree (``ord`` in {None, 2, 'fro', 1, inf})."""
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    if ord is None or ord == 2 or ord == "fro":
        return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))
    if ord == 1:
        return sum(jnp.sum(jnp.abs(leaf)) for leaf in leaves)
    if ord in (jnp.inf, math.inf, "inf"):
        return jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    raise ValueError(f"Unsupported norm order: {ord!r}")


def clip_gradients(
    grads: Any, max_norm: float | None = None, max_value: float | None = None
) -> Any:
    """Clip a gradient pytree by global norm and/or elementwise value.

    Global-norm clipping rescales *all* leaves by ``min(1, max_norm / ||g||)``
    so the update direction is preserved; value clipping changes direction.

    Args:
        grads: Gradient pytree.
        max_norm: If given, rescale so the global L2 norm is at most this.
        max_value: If given, clip each element to ``[-max_value, max_value]``.
    """
    if max_norm is not None:
        global_norm = safe_norm(grads)
        clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
        grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    if max_value is not None:
        grads = jax.tree_util.tree_map(lambda g: jnp.clip(g, -max_value, max_value), grads)
    return grads


# ---------------------------------------------------------------------------
# Finite differences
# ---------------------------------------------------------------------------


def default_fd_step(x: Array, order: int = 2) -> float:
    """Step size that balances truncation and round-off error for ``x``'s dtype.

    For a central difference (``order=2``) the optimal step scales as
    ``eps ** (1/3)``; for a forward difference as ``eps ** (1/2)``.  In float32
    this is ~5e-3 - much larger than the 1e-5 people habitually use, which is
    below the resolution of float32 for ``|x| > ~10``.
    """
    eps = float(jnp.finfo(jnp.result_type(x)).eps)
    scale = float(jnp.maximum(1.0, jnp.max(jnp.abs(x)))) if jnp.size(x) else 1.0
    return (eps ** (1.0 / (order + 1))) * scale


def numerical_gradient(fun: Callable[[Array], Array], x: Array, h: float | None = None) -> Array:
    """Central-difference gradient of a scalar function of one array.

    Args:
        fun: Scalar-valued function.
        x: Point at which to differentiate.
        h: Step size; defaults to :func:`default_fd_step`.  The accuracy you
            can expect is roughly ``h**2`` truncation plus ``eps/h`` round-off,
            i.e. ~1e-4 in float32 and ~1e-10 in float64 at the default step.

    Returns:
        Array with the same shape as ``x``.
    """
    if h is None:
        h = default_fd_step(x, order=2)
    flat = x.ravel()
    n = flat.size

    def partial(i):
        ei = jnp.zeros(n, dtype=flat.dtype).at[i].set(h)
        fp = fun((flat + ei).reshape(x.shape))
        fm = fun((flat - ei).reshape(x.shape))
        return (fp - fm) / (2 * h)

    return jax.vmap(partial)(jnp.arange(n)).reshape(x.shape)


# ---------------------------------------------------------------------------
# Aliases (kept for notebooks / older imports)
# ---------------------------------------------------------------------------

stable_logsumexp = logsumexp_stable
stable_softmax = softmax_stable
