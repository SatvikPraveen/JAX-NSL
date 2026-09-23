# File location: src/jax_nsl/utils/tree_utils.py

"""
Pytree utilities built on :mod:`jax.tree_util`.

A *pytree* is any nested structure of dicts/lists/tuples/NamedTuples (and
registered classes) with arrays at the leaves.  JAX transformations map
over leaves; these helpers cover the operations that come up around them:
path-aware maps, whole-tree arithmetic, stacking/unstacking (for ``scan``
and ``vmap``), and conversion to a flat ``{"a/b/c": array}`` dict for
serialisation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import tree_util

Array = jax.Array


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def path_to_str(path: tuple[Any, ...], sep: str = "/") -> str:
    """Render a JAX key path as ``"layer1/W"`` (dict keys, sequence indices, attribute names)."""
    parts = []
    for entry in path:
        if isinstance(entry, tree_util.DictKey):
            parts.append(str(entry.key))
        elif isinstance(entry, tree_util.SequenceKey):
            parts.append(str(entry.idx))
        elif isinstance(entry, tree_util.GetAttrKey):
            parts.append(str(entry.name))
        else:
            parts.append(str(entry))
    return sep.join(parts)


def tree_flatten_with_path(tree: Any) -> tuple[list[tuple[tuple[Any, ...], Any]], Any]:
    """``([(key_path, leaf), ...], treedef)`` via :func:`jax.tree_util.tree_flatten_with_path`."""
    return tree_util.tree_flatten_with_path(tree)


def tree_unflatten_with_path(path_leaf_pairs: Sequence[tuple[Any, Any]], treedef: Any) -> Any:
    """Inverse of :func:`tree_flatten_with_path`."""
    return tree_util.tree_unflatten(treedef, [leaf for _, leaf in path_leaf_pairs])


def tree_map_with_key(f: Callable, tree: Any, *rest: Any) -> Any:
    """``f(key_path, leaf, *other_leaves)`` over the tree (alias of ``tree_map_with_path``)."""
    return tree_util.tree_map_with_path(f, tree, *rest)


def tree_paths(tree: Any, sep: str = "/") -> list[str]:
    """String path of every leaf, in flatten order."""
    return [path_to_str(p, sep) for p, _ in tree_util.tree_leaves_with_path(tree)]


def tree_flatten_dict(tree: Any, sep: str = "/") -> dict[str, Any]:
    """``{"layer1/W": array, ...}`` - the layout most checkpoint formats want."""
    return {path_to_str(p, sep): leaf for p, leaf in tree_util.tree_leaves_with_path(tree)}


def tree_unflatten_dict(flat: dict[str, Any], sep: str = "/") -> dict[str, Any]:
    """Inverse of :func:`tree_flatten_dict` (rebuilds nested dicts; list indices become dict keys)."""
    out: dict[str, Any] = {}
    for key, value in flat.items():
        node = out
        parts = key.split(sep)
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def tree_update_at_path(tree: Any, path: Sequence[Any], new_value: Any) -> Any:
    """Functionally replace the subtree at ``path`` (keys/indices, or JAX key-path entries)."""

    def key_of(entry):
        if isinstance(entry, tree_util.DictKey):
            return entry.key
        if isinstance(entry, tree_util.SequenceKey):
            return entry.idx
        if isinstance(entry, tree_util.GetAttrKey):
            return entry.name
        return entry

    def rec(node, remaining):
        if not remaining:
            return new_value
        key, rest = key_of(remaining[0]), remaining[1:]
        if isinstance(node, dict):
            return {**node, key: rec(node[key], rest)}
        if isinstance(node, tuple) and hasattr(node, "_fields"):  # NamedTuple
            return node._replace(**{key: rec(getattr(node, key), rest)})
        if isinstance(node, (list, tuple)):
            items = list(node)
            items[key] = rec(node[key], rest)
            return type(node)(items)
        raise TypeError(f"Cannot index into {type(node).__name__}")

    return rec(tree, list(path))


# ---------------------------------------------------------------------------
# Reductions and arithmetic
# ---------------------------------------------------------------------------


def tree_reduce(tree: Any, reduce_fn: Callable, initializer: Any = None) -> Any:
    """Fold ``reduce_fn`` over the leaves (``jax.tree_util.tree_reduce``)."""
    if initializer is None:
        return tree_util.tree_reduce(reduce_fn, tree)
    return tree_util.tree_reduce(reduce_fn, tree, initializer)


def tree_norm(tree: Any) -> Array:
    """Global L2 norm over all leaves."""
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in tree_util.tree_leaves(tree)))


def tree_dot(a: Any, b: Any) -> Array:
    """Inner product ``sum_i <a_i, b_i>`` over matching leaves."""
    return sum(jnp.vdot(x, y) for x, y in zip(tree_util.tree_leaves(a), tree_util.tree_leaves(b)))


def tree_add(a: Any, b: Any) -> Any:
    """Leaf-wise ``a + b``."""
    return tree_util.tree_map(jnp.add, a, b)


def tree_sub(a: Any, b: Any) -> Any:
    """Leaf-wise ``a - b``."""
    return tree_util.tree_map(jnp.subtract, a, b)


def tree_scale(tree: Any, scalar: float | Array) -> Any:
    """Leaf-wise ``scalar * leaf``."""
    return tree_util.tree_map(lambda x: scalar * x, tree)


def tree_zeros_like(tree: Any) -> Any:
    """Zeros with the structure/shapes/dtypes of ``tree``."""
    return tree_util.tree_map(jnp.zeros_like, tree)


def tree_random_like(key: Array, tree: Any, sampler: Callable = jax.random.normal) -> Any:
    """Independent random leaves (one split key per leaf) shaped like ``tree``."""
    leaves, treedef = tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return treedef.unflatten([sampler(k, leaf.shape, leaf.dtype) for k, leaf in zip(keys, leaves)])


def tree_cast(tree: Any, dtype: Any) -> Any:
    """Cast floating leaves to ``dtype``; leave integer/bool leaves alone."""
    return tree_util.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree
    )


def tree_select(tree: Any, condition_fn: Callable[[Any], bool]) -> Any:
    """Keep leaves for which ``condition_fn(leaf)`` is true; others become ``None``.

    ``None`` is an empty subtree in JAX, so the result can still be mapped
    over - useful for e.g. applying weight decay only to matrices.
    """
    return tree_util.tree_map(lambda leaf: leaf if condition_fn(leaf) else None, tree)


def tree_apply_mask(tree: Any, mask_tree: Any, default_value: Any = 0.0) -> Any:
    """``where(mask, leaf, default)`` leaf-wise (mask leaves may be arrays or bools)."""

    def apply(leaf, mask):
        if hasattr(mask, "shape") and mask.shape != ():
            return jnp.where(mask, leaf, default_value)
        return leaf if bool(mask) else jnp.full_like(leaf, default_value)

    return tree_util.tree_map(apply, tree, mask_tree)


# ---------------------------------------------------------------------------
# Stacking (for scan / vmap over lists of trees)
# ---------------------------------------------------------------------------


def tree_stack(trees: Sequence[Any], axis: int = 0) -> Any:
    """Stack a list of identically structured trees along a new axis."""
    if not trees:
        raise ValueError("Cannot stack an empty list of trees")
    return tree_util.tree_map(lambda *leaves: jnp.stack(leaves, axis=axis), *trees)


def tree_unstack(tree: Any, axis: int = 0) -> list[Any]:
    """Inverse of :func:`tree_stack`."""
    leaves, treedef = tree_util.tree_flatten(tree)
    n = leaves[0].shape[axis]
    return [treedef.unflatten([jnp.take(leaf, i, axis=axis) for leaf in leaves]) for i in range(n)]


def tree_concatenate(trees: Sequence[Any], axis: int = 0) -> Any:
    """Concatenate leaves along an existing axis."""
    if not trees:
        raise ValueError("Cannot concatenate an empty list of trees")
    return tree_util.tree_map(lambda *leaves: jnp.concatenate(leaves, axis=axis), *trees)


def tree_take(tree: Any, indices: Any, axis: int = 0) -> Any:
    """``jnp.take`` on every leaf."""
    return tree_util.tree_map(lambda leaf: jnp.take(leaf, indices, axis=axis), tree)


def tree_slice(tree: Any, slice_obj: slice | tuple[slice, ...], axis: int = 0) -> Any:
    """Slice every leaf along ``axis`` (or with a full index tuple)."""

    def do(leaf):
        if isinstance(slice_obj, slice):
            idx = [slice(None)] * leaf.ndim
            idx[axis] = slice_obj
            return leaf[tuple(idx)]
        return leaf[slice_obj]

    return tree_util.tree_map(do, tree)


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def tree_diff(tree1: Any, tree2: Any, tolerance: float = 1e-8) -> dict[str, Any]:
    """Compare two trees: structure, then per-leaf max/mean absolute difference."""
    pairs1, def1 = tree_flatten_with_path(tree1)
    pairs2, def2 = tree_flatten_with_path(tree2)
    if def1 != def2:
        return {"structure_differs": True, "tree1_structure": def1, "tree2_structure": def2}

    differences = []
    max_diff = 0.0
    for (path, a), (_, b) in zip(pairs1, pairs2):
        name = path_to_str(path)
        if hasattr(a, "shape") and hasattr(b, "shape"):
            if a.shape != b.shape:
                differences.append({"path": name, "error": f"shape {a.shape} vs {b.shape}"})
                continue
            d = jnp.abs(jnp.asarray(a, jnp.float32) - jnp.asarray(b, jnp.float32))
            leaf_max = float(jnp.max(d)) if d.size else 0.0
            max_diff = max(max_diff, leaf_max)
            if leaf_max > tolerance:
                differences.append(
                    {"path": name, "max_diff": leaf_max, "mean_diff": float(jnp.mean(d))}
                )
        elif a != b:
            differences.append({"path": name, "value1": a, "value2": b})
    return {
        "structure_differs": False,
        "num_differences": len(differences),
        "max_difference": max_diff,
        "differences": differences,
        "trees_equal": not differences,
    }


def tree_statistics(tree: Any) -> dict[str, Any]:
    """Leaf counts, element/byte totals, dtypes, depth."""
    pairs, treedef = tree_flatten_with_path(tree)
    if not pairs:
        return {"empty": True}
    arrays = [leaf for _, leaf in pairs if hasattr(leaf, "shape")]
    stats: dict[str, Any] = {
        "empty": False,
        "num_leaves": len(pairs),
        "num_arrays": len(arrays),
        "num_scalars": len(pairs) - len(arrays),
        "max_depth": max(len(p) for p, _ in pairs),
        "tree_structure": treedef,
    }
    if arrays:
        total_bytes = sum(a.nbytes for a in arrays)
        stats["array_statistics"] = {
            "total_elements": int(sum(a.size for a in arrays)),
            "total_bytes": int(total_bytes),
            "total_mb": total_bytes / (1024 * 1024),
            "shapes": [a.shape for a in arrays],
            "unique_dtypes": sorted({str(a.dtype) for a in arrays}),
            "min_ndim": min(a.ndim for a in arrays),
            "max_ndim": max(a.ndim for a in arrays),
        }
    return stats


def tree_shapes(tree: Any) -> Any:
    """Same structure with each leaf replaced by ``(shape, dtype)`` - a quick printable summary."""
    return tree_util.tree_map(lambda x: (tuple(x.shape), str(x.dtype)), tree)
