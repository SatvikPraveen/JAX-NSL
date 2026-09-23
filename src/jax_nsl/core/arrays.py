# File location: src/jax_nsl/core/arrays.py

"""
Array and pytree utilities: dtype introspection, safe casting, tree summaries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

Array = jax.Array


def get_dtype_info(dtype: Any) -> dict[str, Any]:
    """Describe a dtype: name, itemsize, kind, plus ``finfo``/``iinfo`` limits.

    Accepts anything ``jnp.dtype`` accepts (``jnp.float32``, ``"bfloat16"``,
    an array's ``.dtype`` ...).
    """
    dtype = jnp.dtype(dtype)
    info: dict[str, Any] = {"name": dtype.name, "itemsize": dtype.itemsize, "kind": dtype.kind}

    if jnp.issubdtype(dtype, jnp.floating):
        finfo = jnp.finfo(dtype)
        info.update(
            eps=float(finfo.eps),
            max=float(finfo.max),
            min=float(finfo.min),
            tiny=float(finfo.tiny),
            precision=int(finfo.precision),
            resolution=float(finfo.resolution),
            bits=int(finfo.bits),
        )
    elif jnp.issubdtype(dtype, jnp.integer):
        iinfo = jnp.iinfo(dtype)
        info.update(max=int(iinfo.max), min=int(iinfo.min), bits=int(iinfo.bits))
    return info


def safe_cast(x: Array, dtype: Any, clip: bool = True) -> Array:
    """Cast ``x`` to ``dtype``, clipping to the target's representable range first.

    Casting ``1e12`` to ``int32`` is undefined behaviour in XLA; clipping makes
    it saturate instead.  Returns ``x`` itself when no cast is needed.
    """
    dtype = jnp.dtype(dtype)
    if x.dtype == dtype:
        return x
    if clip and jnp.issubdtype(dtype, jnp.integer):
        iinfo = jnp.iinfo(dtype)
        x = jnp.clip(x, iinfo.min, iinfo.max)
    elif clip and jnp.issubdtype(dtype, jnp.floating):
        finfo = jnp.finfo(dtype)
        x = jnp.clip(x, finfo.min, finfo.max)
    return x.astype(dtype)


def check_finite(x: Array) -> bool:
    """True iff every element of ``x`` is finite (forces a device sync)."""
    return bool(jnp.all(jnp.isfinite(x)))


# ---------------------------------------------------------------------------
# Pytree analysis
# ---------------------------------------------------------------------------


def _array_leaves(tree: Any):
    return [leaf for leaf in tree_util.tree_leaves(tree) if hasattr(leaf, "shape")]


def tree_size(tree: Any) -> int:
    """Total number of scalar elements across all array leaves."""
    return int(sum(leaf.size for leaf in _array_leaves(tree)))


def tree_bytes(tree: Any) -> int:
    """Total bytes across all array leaves."""
    return int(sum(leaf.nbytes for leaf in _array_leaves(tree)))


def tree_summary(tree: Any, name: str = "Tree") -> dict[str, Any]:
    """Summarise a pytree: leaf count, elements, bytes, shapes, dtypes, devices."""
    arrays = _array_leaves(tree)
    if not arrays:
        return {"name": name, "empty": True}

    devices = set()
    for arr in arrays:
        if hasattr(arr, "devices"):
            devices.update(str(d) for d in arr.devices())
        else:
            devices.add("host")

    return {
        "name": name,
        "num_arrays": len(arrays),
        "total_elements": tree_size(tree),
        "total_bytes": tree_bytes(tree),
        "shapes": [arr.shape for arr in arrays],
        "dtypes": sorted({str(arr.dtype) for arr in arrays}),
        "devices": sorted(devices),
        "tree_structure": tree_util.tree_structure(tree),
    }


def tree_map_with_path(f: Callable[[Any, Any], Any], tree: Any) -> Any:
    """``tree_map`` whose function receives ``(path, leaf)``.

    Thin alias for :func:`jax.tree_util.tree_map_with_path`; use
    :func:`jax.tree_util.keystr` to turn the path into a string.
    """
    return tree_util.tree_map_with_path(f, tree)


# ---------------------------------------------------------------------------
# Common array patterns
# ---------------------------------------------------------------------------


def create_mesh_grid(
    shape: tuple[int, ...], bounds: tuple[tuple[float, float], ...] | None = None
) -> list:
    """``meshgrid(indexing='ij')`` over ``linspace`` axes with the given bounds."""
    if bounds is None:
        bounds = tuple((0.0, float(s - 1)) for s in shape)
    coords = [jnp.linspace(lo, hi, s) for s, (lo, hi) in zip(shape, bounds)]
    return jnp.meshgrid(*coords, indexing="ij")


def sliding_window(x: Array, window_size: int, stride: int = 1) -> Array:
    """Windows over the last axis: ``(..., n) -> (..., num_windows, window_size)``."""
    if window_size > x.shape[-1]:
        raise ValueError(f"Window size {window_size} larger than array size {x.shape[-1]}")
    num_windows = (x.shape[-1] - window_size) // stride + 1
    idx = jnp.arange(window_size)[None, :] + jnp.arange(num_windows)[:, None] * stride
    return x[..., idx]


def pad_to_shape(
    x: Array, target_shape: tuple[int, ...], mode: str = "constant", constant_values: float = 0.0
) -> Array:
    """Right-pad ``x`` so that its shape equals ``target_shape``."""
    if x.ndim != len(target_shape):
        raise ValueError(f"Rank mismatch: {x.shape} vs {target_shape}")
    pad_widths = []
    for current, target in zip(x.shape, target_shape):
        if current > target:
            raise ValueError(f"Dimension {current} larger than target {target}")
        pad_widths.append((0, target - current))
    if mode == "constant":
        return jnp.pad(x, pad_widths, mode=mode, constant_values=constant_values)
    return jnp.pad(x, pad_widths, mode=mode)
