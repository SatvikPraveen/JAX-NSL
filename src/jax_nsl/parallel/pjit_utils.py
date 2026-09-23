# File location: src/jax_nsl/parallel/pjit_utils.py

"""
Sharding with ``jax.jit`` and ``shard_map`` (the modern replacement for ``pjit``).

Since JAX 0.4 ``jit`` *is* ``pjit``: give it ``in_shardings``/``out_shardings``
built from a :class:`jax.sharding.Mesh` and ``PartitionSpec`` and XLA's
SPMD partitioner inserts the collectives.  The mental model:

* A ``Mesh`` names the device axes, e.g. ``('data', 'model')`` over a 2x4 grid.
* A ``PartitionSpec`` says, per *array* axis, which mesh axis (or ``None``)
  it is split across.  ``P('data', None)`` shards rows over the data axis and
  replicates columns.
* ``NamedSharding(mesh, spec)`` attaches that to an array (``device_put``) or
  to a jitted function's inputs/outputs.
* ``shard_map`` is the escape hatch: you write the *per-device* program and
  call collectives (``psum`` etc.) explicitly.

Everything here works on a single host with virtual CPU devices
(``XLA_FLAGS=--xla_force_host_platform_device_count=8``), which is how the
tests run.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import jax
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

P = PartitionSpec
Array = jax.Array


# ---------------------------------------------------------------------------
# Meshes and shardings
# ---------------------------------------------------------------------------

def create_mesh(mesh_shape: Sequence[int], axis_names: Sequence[str],
                devices: Optional[Sequence[jax.Device]] = None,
                axis_types: Optional[Sequence[AxisType]] = None) -> Mesh:
    """Device mesh with the given shape and axis names (``jax.make_mesh``).

    ``make_mesh`` orders devices to favour fast interconnect for the *last*
    axis on TPU pods; on CPU/GPU it is simply a reshape of the device list.

    Axis types default to ``AxisType.Auto``: shardings are hints and XLA may
    re-shard freely (the classic ``pjit`` behaviour).  Recent JAX defaults
    ``make_mesh`` to *explicit* sharding, where array types carry their
    sharding and ``shard_map`` refuses inputs whose sharding differs from
    ``in_specs``; pass ``axis_types`` explicitly to opt into that.
    """
    if axis_types is None:
        axis_types = (AxisType.Auto,) * len(axis_names)
    if devices is None:
        return jax.make_mesh(tuple(mesh_shape), tuple(axis_names), axis_types=tuple(axis_types))
    return Mesh(np.array(devices, dtype=object).reshape(tuple(mesh_shape)), tuple(axis_names),
                axis_types=tuple(axis_types))


def named_sharding(mesh: Mesh, *spec: Any) -> NamedSharding:
    """``NamedSharding(mesh, P(*spec))``."""
    return NamedSharding(mesh, P(*spec))


def replicated(mesh: Mesh) -> NamedSharding:
    """Sharding that places a full copy of the array on every device."""
    return NamedSharding(mesh, P())


def _current_mesh() -> Mesh:
    from jax._src.mesh import thread_resources  # context-manager mesh

    mesh = thread_resources.env.physical_mesh
    if mesh.empty:
        raise ValueError("No mesh given and no 'with mesh:' context is active")
    return mesh


def create_sharded_array(array: Array, partition_spec: PartitionSpec, mesh: Optional[Mesh] = None) -> Array:
    """``device_put`` an array with ``NamedSharding(mesh, partition_spec)``.

    If ``mesh`` is ``None`` the mesh from the enclosing ``with mesh:`` block is
    used.
    """
    mesh = mesh or _current_mesh()
    return jax.device_put(array, NamedSharding(mesh, partition_spec))


shard_array = create_sharded_array


def check_sharding_compatibility(array: Array, partition_spec: PartitionSpec, mesh: Mesh) -> bool:
    """True iff every sharded array axis is divisible by the size of its mesh axis."""
    if len(partition_spec) > array.ndim:
        return False
    for dim, axes in zip(array.shape, partition_spec):
        if axes is None:
            continue
        axes = (axes,) if isinstance(axes, str) else tuple(axes)
        size = 1
        for a in axes:
            size *= mesh.shape[a]
        if dim % size:
            return False
    return True


def sharding_summary(tree: Any) -> Dict[str, str]:
    """``{path: sharding}`` for every leaf - handy to check a partitioned model."""
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        spec = getattr(getattr(leaf, "sharding", None), "spec", None)
        out[jax.tree_util.keystr(path)] = str(spec) if spec is not None else "unsharded"
    return out


# ---------------------------------------------------------------------------
# Partitioning parameter trees
# ---------------------------------------------------------------------------

def partition_specs(params: Any, partition_rules: Dict[str, PartitionSpec],
                    default: PartitionSpec = P()) -> Any:
    """Pytree of ``PartitionSpec`` (same structure as ``params``) from name rules.

    Rules are matched against the leaf's key path (as produced by
    ``jax.tree_util.keystr``, e.g. ``"['layers']['attention']['query']"``)
    with :func:`re.search`, so a rule ``r"attention.*query"`` applies to every
    layer's query projection.  Unmatched leaves get ``default`` (replicated).
    """
    compiled = [(re.compile(pattern), spec) for pattern, spec in partition_rules.items()]

    def pick(path, leaf):
        name = jax.tree_util.keystr(path)
        for pattern, spec in compiled:
            if pattern.search(name):
                return spec
        return default

    return jax.tree_util.tree_map_with_path(pick, params)


def partition_params(params: Any, partition_rules: Dict[str, PartitionSpec], mesh: Optional[Mesh] = None,
                     default: PartitionSpec = P()) -> Any:
    """Shard a parameter pytree according to name-based rules (see :func:`partition_specs`).

    Returns the tree with every leaf ``device_put`` under its ``NamedSharding``.
    """
    mesh = mesh or _current_mesh()
    specs = partition_specs(params, partition_rules, default)
    is_spec = lambda x: isinstance(x, PartitionSpec)  # noqa: E731
    return jax.tree_util.tree_map(lambda leaf, spec: jax.device_put(leaf, NamedSharding(mesh, spec)),
                                  params, specs, is_leaf=lambda x: is_spec(x))


def fsdp_rules(axis: str = "data", min_size: int = 1) -> Callable[[Any], Any]:
    """Fully-sharded data parallel: shard every parameter's *largest* axis over ``axis``.

    Returns a function ``params -> specs``.  Arrays smaller than ``min_size``
    elements (biases, norms) are replicated - sharding them costs more in
    all-gathers than it saves.
    """
    def specs(params):
        def pick(leaf):
            if leaf.ndim == 0 or leaf.size < min_size:
                return P()
            largest = max(range(leaf.ndim), key=lambda i: leaf.shape[i])
            entries = [None] * leaf.ndim
            entries[largest] = axis
            return P(*entries)

        return jax.tree_util.tree_map(pick, params)

    return specs


def create_transformer_partition_specs(model_axis: str = "model") -> Dict[str, PartitionSpec]:
    """Megatron-style tensor-parallel rules for the ``jax_nsl`` transformer.

    Column-parallel for the projections that *produce* the hidden dimension
    (``query``/``key``/``value``/``W1``), row-parallel for those that
    *consume* it (``out``/``W2``), so each block needs exactly one
    all-reduce per sub-layer.  Layer norms are replicated.  The leading
    layer-stack axis (index 0 of every stacked leaf) is never sharded.
    """
    return {
        r"attention.*(query|key|value)": P(None, None, model_axis),
        r"attention.*out": P(None, model_axis, None),
        r"ffn.*W1": P(None, None, model_axis),
        r"ffn.*W2": P(None, model_axis, None),
        r"ffn.*b1": P(None, model_axis),
        r"embedding": P(None, model_axis),
    }


# ---------------------------------------------------------------------------
# Sharded computations
# ---------------------------------------------------------------------------

def _to_shardings(mesh: Mesh, specs: Any) -> Any:
    return jax.tree_util.tree_map(lambda spec: NamedSharding(mesh, spec), specs,
                                  is_leaf=lambda x: isinstance(x, PartitionSpec))


def setup_model_parallelism(fn: Callable, mesh: Mesh, in_specs: Any, out_specs: Any,
                            static_argnums: Optional[Tuple[int, ...]] = None) -> Callable:
    """``jit(fn, in_shardings=..., out_shardings=...)`` from partition specs.

    ``in_specs``/``out_specs`` are pytrees of ``PartitionSpec`` matching the
    function's arguments/outputs.  This is exactly what ``pjit`` used to be.
    """
    return jax.jit(fn, in_shardings=_to_shardings(mesh, in_specs),
                   out_shardings=_to_shardings(mesh, out_specs), static_argnums=static_argnums or ())


def model_parallel_forward(forward_fn: Callable, params: Any, inputs: Array, mesh: Mesh,
                           param_specs: Any, input_spec: PartitionSpec, output_spec: PartitionSpec) -> Array:
    """Run ``forward_fn(params, inputs)`` sharded as specified (one-off convenience)."""
    fn = setup_model_parallelism(forward_fn, mesh, (param_specs, input_spec), output_spec)
    return fn(params, inputs)


def make_sharded_train_step(loss_fn: Callable[[Any, Any], Array], optimizer_update: Callable,
                            mesh: Mesh, param_specs: Any, batch_specs: Any) -> Callable:
    """Data/model-parallel training step for ``loss_fn(params, batch)``.

    The step is ordinary single-program code; ``jit`` with shardings lets XLA
    partition it.  Gradient all-reduces across the data axis are inserted
    automatically because the loss reduces over the sharded batch axis.
    Optimiser-state leaves that mirror a parameter (moments) take that
    parameter's spec; scalars (the step counter) are replicated.
    """
    def step(opt_state, batch):
        loss, grads = jax.value_and_grad(loss_fn)(opt_state.params, batch)
        return optimizer_update(opt_state, grads), loss

    is_spec = lambda x: isinstance(x, PartitionSpec)  # noqa: E731
    flat_param_specs = {jax.tree_util.keystr(p): s
                        for p, s in jax.tree_util.tree_leaves_with_path(param_specs, is_leaf=is_spec)}

    def opt_state_specs(opt_state):
        def pick(path, leaf):
            if leaf.ndim == 0:
                return P()
            # path[0] is the NamedTuple field (params / mu / ...); the rest names the parameter.
            return flat_param_specs.get(jax.tree_util.keystr(path[1:]), P())

        return jax.tree_util.tree_map_with_path(pick, opt_state)

    def sharded_step(opt_state, batch):
        specs = opt_state_specs(opt_state)
        fn = setup_model_parallelism(step, mesh, (specs, batch_specs), (specs, P()))
        return fn(opt_state, batch)

    return sharded_step


def shard_map_fn(fn: Callable, mesh: Mesh, in_specs: Any, out_specs: Any, check_vma: bool = True) -> Callable:
    """``jax.shard_map`` with a mesh bound - write per-device code with explicit collectives."""
    return jax.shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_specs, check_vma=check_vma)


def sharded_matmul_shard_map(mesh: Mesh, axis: str = "model") -> Callable[[Array, Array], Array]:
    """``x @ w`` with ``w`` column-sharded over ``axis``, written with ``shard_map``.

    Each device multiplies the full ``x`` by its slice of ``w`` and the
    outputs are concatenated by the ``out_specs`` - no communication at all.
    Compare with the row-sharded version which needs a ``psum``.
    """
    def per_device(x, w_shard):
        return x @ w_shard

    return jax.shard_map(per_device, mesh=mesh, in_specs=(P(), P(None, axis)), out_specs=P(None, axis))


def sharded_matmul_row_parallel(mesh: Mesh, axis: str = "model") -> Callable[[Array, Array], Array]:
    """``x @ w`` with ``x`` column- and ``w`` row-sharded: partial products summed by ``psum``."""
    def per_device(x_shard, w_shard):
        return jax.lax.psum(x_shard @ w_shard, axis)

    return jax.shard_map(per_device, mesh=mesh, in_specs=(P(None, axis), P(axis, None)), out_specs=P())


# ---------------------------------------------------------------------------
# Memory estimates
# ---------------------------------------------------------------------------

def estimate_memory_per_device(params: Any, mesh: Mesh, specs: Any) -> Dict[str, float]:
    """Bytes of parameters in total vs. per device under the given specs (MB)."""
    total = 0
    per_device = 0.0
    is_spec = lambda x: isinstance(x, PartitionSpec)  # noqa: E731
    for leaf, spec in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(specs, is_leaf=is_spec)):
        total += leaf.nbytes
        factor = 1
        for axes in spec:
            if axes is None:
                continue
            for a in ((axes,) if isinstance(axes, str) else axes):
                factor *= mesh.shape[a]
        per_device += leaf.nbytes / factor
    mb = 1024 * 1024
    return {"total_params_mb": total / mb, "params_per_device_mb": per_device / mb,
            "memory_reduction_factor": total / per_device if per_device else 1.0}


def shard_large_layer(weights: Array, bias: Array, mesh: Mesh, shard_axis: int = 1,
                      model_axis: str = "model") -> Tuple[Array, Array]:
    """Shard a dense layer column-wise (``shard_axis=1``) or row-wise (``0``)."""
    if shard_axis == 1:
        w_spec, b_spec = P(None, model_axis), P(model_axis)
    elif shard_axis == 0:
        w_spec, b_spec = P(model_axis, None), P()
    else:
        raise ValueError(f"Invalid shard_axis: {shard_axis}")
    return (jax.device_put(weights, NamedSharding(mesh, w_spec)),
            jax.device_put(bias, NamedSharding(mesh, b_spec)))
