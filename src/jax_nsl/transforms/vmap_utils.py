# File location: src/jax_nsl/transforms/vmap_utils.py

"""
Vectorisation with ``vmap``: batching rules, per-example gradients, and
memory-bounded chunking.

``vmap`` turns a function written for one example into one for a batch by
adding a batch dimension to every primitive - no Python loop, and the result
fuses with ``jit``.  Two patterns deserve names:

* :func:`per_example_gradients` - ``vmap(grad(loss), in_axes=(None, 0, 0))``,
  the building block of DP-SGD, influence functions and gradient-noise analysis.
* :func:`chunked_vmap` - ``lax.map`` over ``vmap``-ed chunks, for batches whose
  fully vectorised intermediates would not fit in memory.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import lax, vmap

Array = jax.Array


def batch_apply(fun: Callable, in_axes: Any = 0, out_axes: Any = 0,
                axis_name: Optional[str] = None, axis_size: Optional[int] = None) -> Callable:
    """``vmap`` with keyword defaults (identical semantics to :func:`jax.vmap`)."""
    return vmap(fun, in_axes=in_axes, out_axes=out_axes, axis_name=axis_name, axis_size=axis_size)


def vectorize_function(in_axes: Any = 0, out_axes: Any = 0) -> Callable:
    """Decorator form of :func:`batch_apply`."""
    return lambda fun: batch_apply(fun, in_axes=in_axes, out_axes=out_axes)


def loop_batch_apply(fun: Callable, xs: Array, *rest) -> Array:
    """Reference implementation: a Python loop over the batch (for benchmarking against ``vmap``)."""
    return jnp.stack([fun(x, *rest) for x in xs])


def parallel_apply(fun: Callable, *args, in_axes: Any = 0) -> Any:
    """Apply ``fun`` elementwise over batched positional arguments: ``vmap(fun, in_axes)(*args)``."""
    return vmap(fun, in_axes=in_axes)(*args)


def batch_outer_product(x: Array, y: Array) -> Array:
    """``(b, m), (b, n) -> (b, m, n)``."""
    return vmap(jnp.outer)(x, y)


def batched_matmul(a: Array, b: Array) -> Array:
    """``(b, m, k) @ (b, k, n)`` via ``vmap``."""
    return vmap(jnp.matmul)(a, b)


def batch_solve(a: Array, b: Array) -> Array:
    """Solve a batch of linear systems ``a[i] x = b[i]``."""
    return vmap(jnp.linalg.solve)(a, b)


def batch_matrix_ops(matrices: Array, operation: str = "inv", **kwargs) -> Any:
    """Apply a ``jnp.linalg`` routine to each matrix in a batch."""
    ops = {
        "inv": jnp.linalg.inv, "det": jnp.linalg.det, "eig": jnp.linalg.eig,
        "eigvals": jnp.linalg.eigvals, "cholesky": jnp.linalg.cholesky, "qr": jnp.linalg.qr,
        "svd": lambda m: jnp.linalg.svd(m, full_matrices=kwargs.get("full_matrices", True)),
    }
    if operation not in ops:
        raise ValueError(f"Unsupported matrix operation: {operation}")
    return vmap(ops[operation])(matrices)


def batch_apply_along_axis(fun: Callable, axis: int, arr: Array, keepdims: bool = False) -> Array:
    """Apply ``fun`` to every slice along ``axis`` (like ``np.apply_along_axis``)."""
    moved = jnp.moveaxis(arr, axis, 0)
    result = vmap(fun)(moved)
    if keepdims:
        result = jnp.moveaxis(jnp.expand_dims(result, 0), 0, axis)
    return result


def nested_vmap(fun: Callable, in_axes_list: Sequence[Any], out_axes_list: Sequence[Any]) -> Callable:
    """Compose several ``vmap`` levels, innermost first."""
    for in_axes, out_axes in zip(reversed(in_axes_list), reversed(out_axes_list)):
        fun = vmap(fun, in_axes=in_axes, out_axes=out_axes)
    return fun


def selective_vmap(fun: Callable, condition_fn: Callable, in_axes: Any = 0, out_axes: Any = 0) -> Callable:
    """Use the ``vmap``-ed version only when ``condition_fn(*args)`` is true (a Python-level check)."""
    vmapped = vmap(fun, in_axes=in_axes, out_axes=out_axes)

    def wrapped(*args, **kwargs):
        return vmapped(*args, **kwargs) if condition_fn(*args, **kwargs) else fun(*args, **kwargs)

    return wrapped


def vmap_with_signature(signature: str) -> Callable:
    """Generalised-ufunc style batching via ``jnp.vectorize(signature=...)``.

    Example: ``@vmap_with_signature('(m,n),(n)->(m)')`` turns a matrix-vector
    product into one that broadcasts over any leading batch dimensions.
    """
    return lambda fun: jnp.vectorize(fun, signature=signature)


# ---------------------------------------------------------------------------
# Gradients over batches
# ---------------------------------------------------------------------------

def batch_gradient(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """``vmap(grad(fun))`` - gradient of a per-example scalar function for every example."""
    return vmap(jax.grad(fun, argnums=argnums))


def batch_jacobian(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """``vmap(jacobian(fun))``."""
    return vmap(jax.jacobian(fun, argnums=argnums))


def batched_gradient(fun: Callable, xs: Array, *rest) -> Array:
    """Per-example gradients of ``fun(x, *rest)`` over the leading axis of ``xs``."""
    in_axes = (0,) + (None,) * len(rest)
    return vmap(jax.grad(fun), in_axes=in_axes)(xs, *rest)


def per_example_gradients(loss_fn: Callable[[Any, Any, Any], Array], params: Any,
                          inputs: Any, targets: Any) -> Any:
    """Gradient of ``loss_fn(params, x_i, y_i)`` for each example ``i``.

    The returned pytree has the structure of ``params`` with a leading batch
    axis on every leaf.  Averaging the leaves recovers the usual minibatch
    gradient; clipping each example's gradient before averaging gives DP-SGD.
    """
    return vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))(params, inputs, targets)


def clip_per_example_gradients(per_example_grads: Any, max_norm: float) -> Any:
    """Clip each example's gradient (across all leaves) to ``max_norm``, then return the mean."""
    def example_norm_sq(*leaves):
        return sum(jnp.sum(jnp.square(leaf.reshape(leaf.shape[0], -1)), axis=1) for leaf in leaves)

    leaves = jax.tree_util.tree_leaves(per_example_grads)
    norms = jnp.sqrt(example_norm_sq(*leaves))
    factors = jnp.minimum(1.0, max_norm / (norms + 1e-12))

    def clip_and_mean(leaf):
        f = factors.reshape((-1,) + (1,) * (leaf.ndim - 1))
        return jnp.mean(leaf * f, axis=0)

    return jax.tree_util.tree_map(clip_and_mean, per_example_grads)


# ---------------------------------------------------------------------------
# Memory-bounded batching
# ---------------------------------------------------------------------------

def chunked_vmap(fun: Callable, xs: Array, chunk_size: int) -> Array:
    """``vmap(fun)`` applied ``chunk_size`` examples at a time with ``lax.map``.

    ``vmap`` alone materialises every intermediate for the whole batch;
    ``lax.map`` over chunks caps that at ``chunk_size`` examples while still
    vectorising inside each chunk.  The last partial chunk is padded and the
    padding dropped afterwards.
    """
    n = xs.shape[0]
    num_chunks = -(-n // chunk_size)
    pad = num_chunks * chunk_size - n
    if pad:
        xs = jnp.concatenate([xs, jnp.zeros((pad,) + xs.shape[1:], xs.dtype)], axis=0)
    chunks = xs.reshape((num_chunks, chunk_size) + xs.shape[1:])
    out = lax.map(vmap(fun), chunks)
    out = out.reshape((num_chunks * chunk_size,) + out.shape[2:])
    return out[:n]


def parallel_map(fun: Callable, xs: Array, chunk_size: Optional[int] = None) -> Array:
    """``vmap(fun)(xs)``, or :func:`chunked_vmap` when ``chunk_size`` is given."""
    if chunk_size is None:
        return vmap(fun)(xs)
    return chunked_vmap(fun, xs, chunk_size)
