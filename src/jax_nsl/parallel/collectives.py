# File location: src/jax_nsl/parallel/collectives.py

"""
Collective communication inside ``pmap``/``shard_map`` bodies.

Every function here must be called *inside* a mapped function with a named
axis (``axis_name``).  The primitives are ``psum``/``pmean``/``pmax``/``pmin``
(all-reduce), ``all_gather``, ``psum_scatter`` (reduce-scatter),
``all_to_all`` and ``ppermute`` (point-to-point along a permutation).

Two educational implementations are included: a broadcast built from
``psum`` and a *real* ring all-reduce built from ``ppermute``, which is
the algorithm behind NCCL's bandwidth-optimal all-reduce.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


# ---------------------------------------------------------------------------
# All-reduce family
# ---------------------------------------------------------------------------


def all_reduce_mean(x: Array, axis_name: str = "batch") -> Array:
    """``pmean``."""
    return lax.pmean(x, axis_name=axis_name)


def all_reduce_sum(x: Array, axis_name: str = "batch") -> Array:
    """``psum``."""
    return lax.psum(x, axis_name=axis_name)


def all_reduce_max(x: Array, axis_name: str = "batch") -> Array:
    """``pmax``."""
    return lax.pmax(x, axis_name=axis_name)


def all_reduce_min(x: Array, axis_name: str = "batch") -> Array:
    """``pmin``."""
    return lax.pmin(x, axis_name=axis_name)


cross_replica_mean = all_reduce_mean


def tree_all_reduce(tree: Any, reduction: str = "mean", axis_name: str = "batch") -> Any:
    """Apply one all-reduce to every leaf of a pytree."""
    ops = {"mean": lax.pmean, "sum": lax.psum, "max": lax.pmax, "min": lax.pmin}
    if reduction not in ops:
        raise ValueError(f"Unknown reduction: {reduction}")
    return ops[reduction](tree, axis_name=axis_name)  # collectives accept pytrees


def all_gather(x: Array, axis_name: str = "batch", axis: int = 0, tiled: bool = False) -> Array:
    """Concatenate (``tiled=True``) or stack every device's ``x`` along ``axis``."""
    return lax.all_gather(x, axis_name=axis_name, axis=axis, tiled=tiled)


def reduce_scatter(x: Array, axis_name: str = "batch", scatter_dimension: int = 0) -> Array:
    """Sum across devices, then give each device one contiguous slice (``psum_scatter``).

    A reduce-scatter followed by an all-gather *is* an all-reduce; ZeRO/FSDP
    use the two halves separately so that each device only ever owns a slice
    of the gradient.
    """
    return lax.psum_scatter(x, axis_name, scatter_dimension=scatter_dimension, tiled=True)


def alltoall(
    x: Array, axis_name: str = "batch", split_axis: int = 0, concat_axis: int = 0
) -> Array:
    """``all_to_all``: device ``i`` sends chunk ``j`` of ``x`` to device ``j`` (used in MoE routing)."""
    return lax.all_to_all(x, axis_name, split_axis, concat_axis, tiled=True)


def broadcast(x: Array, root_rank: int = 0, axis_name: str = "batch") -> Array:
    """Every device receives ``x`` from ``root_rank``.

    Implemented as ``psum(where(index == root, x, 0))``: only the root
    contributes a non-zero term, so the sum equals the root's value.
    """
    is_root = lax.axis_index(axis_name) == root_rank
    return lax.psum(jnp.where(is_root, x, jnp.zeros_like(x)), axis_name)


def barrier_sync(axis_name: str = "batch") -> Array:
    """A trivial ``psum`` that forces all replicas to rendezvous."""
    return lax.psum(jnp.ones(()), axis_name)


def distributed_dot(x: Array, y: Array, axis_name: str = "batch") -> Array:
    """Dot product of vectors distributed across devices (local dot + ``psum``)."""
    return lax.psum(jnp.vdot(x, y), axis_name)


def sync_batch_stats(batch_stats: Any, axis_name: str = "batch") -> Any:
    """Average batch-norm statistics (a pytree) across replicas."""
    return lax.pmean(batch_stats, axis_name)


def gradient_synchronization(
    grads: Any, axis_name: str = "batch", clip_norm: float | None = None
) -> Any:
    """Average gradients across replicas, optionally clipping by the *global* norm afterwards.

    Clipping the averaged gradient (what is actually applied) is the
    standard choice; clipping per replica before averaging would bias the
    result.
    """
    grads = lax.pmean(grads, axis_name)
    if clip_norm is not None:
        norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)))
        factor = jnp.minimum(1.0, clip_norm / (norm + 1e-8))
        grads = jax.tree_util.tree_map(lambda g: g * factor, grads)
    return grads


def hierarchical_all_reduce(
    x: Array, intra_node_axis: str = "local", inter_node_axis: str = "global"
) -> Array:
    """Reduce within a node first, then across nodes (two nested mesh axes)."""
    return lax.pmean(lax.pmean(x, intra_node_axis), inter_node_axis)


# ---------------------------------------------------------------------------
# Ring all-reduce from point-to-point sends
# ---------------------------------------------------------------------------


def ring_all_reduce(x: Array, axis_name: str = "batch", num_devices: int | None = None) -> Array:
    """Bandwidth-optimal ring all-reduce (sum) built only from ``ppermute``.

    With ``N`` devices, ``x`` is split into ``N`` chunks.  Phase 1
    (reduce-scatter): for ``N - 1`` steps each device sends one chunk to its
    right neighbour and adds the chunk it receives, so afterwards chunk ``i``
    is fully reduced on device ``(i + 1) % N``.  Phase 2 (all-gather): the
    reduced chunks circulate for another ``N - 1`` steps.  Each device sends
    ``2 (N-1)/N`` of the data in total - independent of ``N``, which is why
    rings scale.  The result equals ``psum(x)``.

    Args:
        x: Per-device array whose leading axis is divisible by ``N``.
        axis_name: The mapped axis.
        num_devices: ``N``; needed statically for the chunking (defaults to
            ``lax.axis_size``).
    """
    n = num_devices if num_devices is not None else lax.axis_size(axis_name)
    rank = lax.axis_index(axis_name)
    chunks = x.reshape((n, -1) + x.shape[1:])  # (n, chunk, ...)
    right = [(i, (i + 1) % n) for i in range(n)]

    def send_right(c):
        return lax.ppermute(c, axis_name, perm=right)

    def take(chunks, idx):
        return lax.dynamic_index_in_dim(chunks, idx, axis=0, keepdims=False)

    # Phase 1: reduce-scatter.  Device r starts by sending chunk r.
    def rs_step(k, chunks):
        send_idx = (rank - k) % n
        recv_idx = (rank - k - 1) % n
        received = send_right(take(chunks, send_idx))
        return chunks.at[recv_idx].add(received)

    chunks = lax.fori_loop(0, n - 1, rs_step, chunks)
    # Now chunk (rank + 1) % n on device `rank` is fully reduced.

    # Phase 2: all-gather.
    def ag_step(k, chunks):
        send_idx = (rank + 1 - k) % n
        recv_idx = (rank - k) % n
        received = send_right(take(chunks, send_idx))
        return chunks.at[recv_idx].set(received)

    chunks = lax.fori_loop(0, n - 1, ag_step, chunks)
    return chunks.reshape(x.shape)


# ---------------------------------------------------------------------------
# Cost models
# ---------------------------------------------------------------------------


def compute_communication_volume(
    array_shapes: Sequence[Sequence[int]],
    num_devices: int,
    collective: str = "all_reduce",
    dtype: Any = jnp.float32,
) -> dict[str, float]:
    """Bytes each device *sends* for a collective over arrays of the given shapes.

    Ring all-reduce: ``2 (N-1)/N * size``; reduce-scatter or all-gather alone:
    ``(N-1)/N * size``; all-to-all: ``(N-1)/N * size``; naive broadcast-based
    all-reduce (everyone sends everything): ``(N-1) * size``.
    """
    itemsize = jnp.dtype(dtype).itemsize
    total = sum(int(jnp.prod(jnp.array(s))) for s in array_shapes) * itemsize
    frac = (num_devices - 1) / num_devices
    factors = {
        "all_reduce": 2 * frac,
        "reduce_scatter": frac,
        "all_gather": frac,
        "all_to_all": frac,
        "naive_all_reduce": num_devices - 1,
    }
    if collective not in factors:
        raise ValueError(f"Unknown collective: {collective}")
    mb = 1024 * 1024
    return {
        "total_data_mb": total / mb,
        "bytes_sent_per_device_mb": total * factors[collective] / mb,
        "num_arrays": len(array_shapes),
        "num_devices": num_devices,
    }
