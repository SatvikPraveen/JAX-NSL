# File location: src/jax_nsl/core/prng.py

"""
PRNG key handling and parameter initialisers.

JAX random numbers are *explicit*: every call takes a key, and you must split
keys yourself to get independent streams.  :class:`PRNGSequence` packages the
split-and-advance pattern; the initialisers compute fan-in/fan-out the same
way :mod:`jax.nn.initializers` does (including convolution kernels).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util

Array = jax.Array
KeyLike = int | Array


def as_key(seed: KeyLike) -> Array:
    """Turn an int or an existing key into a key (typed or legacy is preserved)."""
    if isinstance(seed, (int, jnp.integer)):
        return jr.PRNGKey(int(seed))
    return seed


class PRNGSequence:
    """An iterator of independent PRNG keys derived from one seed.

    Example::

        rng = PRNGSequence(42)
        w = jax.random.normal(next(rng), (3, 3))
        b = jax.random.normal(next(rng), (3,))   # independent of w
    """

    def __init__(self, seed: KeyLike):
        self._key = as_key(seed)

    def __iter__(self) -> Iterator[Array]:
        return self

    def __next__(self) -> Array:
        self._key, subkey = jr.split(self._key)
        return subkey

    def split(self, num: int) -> Array:
        """Return ``num`` fresh keys stacked along axis 0 and advance the state."""
        keys = jr.split(self._key, num + 1)
        self._key = keys[0]
        return keys[1:]

    def fork(self, num: int) -> list[PRNGSequence]:
        """Create ``num`` independent child sequences."""
        return [PRNGSequence(k) for k in self.split(num)]


def split_key_tree(key: Array, tree_structure: Any) -> Any:
    """Split ``key`` into one key per leaf of ``tree_structure`` (same structure)."""
    leaves, treedef = tree_util.tree_flatten(tree_structure)
    if not leaves:
        return tree_structure
    return tree_util.tree_unflatten(treedef, list(jr.split(key, len(leaves))))


def make_rng_state(seed: KeyLike, names: Sequence[str]) -> dict[str, Array]:
    """Named independent streams, e.g. ``{'params': k1, 'dropout': k2}``."""
    keys = jr.split(as_key(seed), len(names))
    return dict(zip(names, keys))


def random_like(key: Array, template: Array, distribution: str = "normal", **kwargs) -> Array:
    """Sample an array with the shape/dtype of ``template``.

    Supported distributions: ``normal`` (``loc``, ``scale``), ``uniform``
    (``minval``, ``maxval``), ``bernoulli`` (``p``), ``categorical`` (``logits``).
    """
    shape, dtype = template.shape, template.dtype
    if distribution == "normal":
        arr = jr.normal(key, shape, dtype=dtype)
        return arr * kwargs.get("scale", 1.0) + kwargs.get("loc", 0.0)
    if distribution == "uniform":
        return jr.uniform(
            key,
            shape,
            dtype=dtype,
            minval=kwargs.get("minval", 0.0),
            maxval=kwargs.get("maxval", 1.0),
        )
    if distribution == "bernoulli":
        return jr.bernoulli(key, kwargs.get("p", 0.5), shape).astype(dtype)
    if distribution == "categorical":
        logits = kwargs.get("logits")
        if logits is None:
            raise ValueError("categorical distribution requires 'logits'")
        return jr.categorical(key, logits, shape=shape).astype(dtype)
    raise ValueError(f"Unknown distribution: {distribution}")


# ---------------------------------------------------------------------------
# Initialisers
# ---------------------------------------------------------------------------


def compute_fans(
    shape: tuple[int, ...], in_axis: int = -2, out_axis: int = -1
) -> tuple[float, float]:
    """``(fan_in, fan_out)`` following the convention of ``jax.nn.initializers``.

    Every axis that is neither ``in_axis`` nor ``out_axis`` is part of the
    receptive field, so for a conv kernel ``(O, I, kh, kw)`` with
    ``in_axis=1, out_axis=0`` we get ``fan_in = I * kh * kw``.
    """
    if len(shape) < 1:
        return 1.0, 1.0
    if len(shape) == 1:
        return float(shape[0]), float(shape[0])
    in_axis %= len(shape)
    out_axis %= len(shape)
    receptive_field = math.prod(s for i, s in enumerate(shape) if i not in (in_axis, out_axis))
    return shape[in_axis] * receptive_field, shape[out_axis] * receptive_field


def _variance_scaling(
    key: Array,
    shape: tuple[int, ...],
    scale: float,
    mode: str,
    distribution: str,
    dtype: Any,
    in_axis: int,
    out_axis: int,
) -> Array:
    fan_in, fan_out = compute_fans(shape, in_axis, out_axis)
    denominator = {"fan_in": fan_in, "fan_out": fan_out, "fan_avg": (fan_in + fan_out) / 2}[mode]
    variance = scale / max(denominator, 1.0)
    if distribution == "normal":
        return jr.normal(key, shape, dtype) * jnp.asarray(math.sqrt(variance), dtype)
    bound = math.sqrt(3.0 * variance)
    return jr.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def glorot_uniform_init(
    key: Array,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
    in_axis: int = -2,
    out_axis: int = -1,
) -> Array:
    """Glorot/Xavier uniform: ``U(-b, b)`` with ``b = sqrt(6 / (fan_in + fan_out))``."""
    return _variance_scaling(key, shape, 1.0, "fan_avg", "uniform", dtype, in_axis, out_axis)


def glorot_normal_init(
    key: Array,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
    in_axis: int = -2,
    out_axis: int = -1,
) -> Array:
    """Glorot/Xavier normal: ``N(0, 2 / (fan_in + fan_out))``."""
    return _variance_scaling(key, shape, 1.0, "fan_avg", "normal", dtype, in_axis, out_axis)


def he_uniform_init(
    key: Array,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
    in_axis: int = -2,
    out_axis: int = -1,
) -> Array:
    """He/Kaiming uniform for ReLU nets: variance ``2 / fan_in``."""
    return _variance_scaling(key, shape, 2.0, "fan_in", "uniform", dtype, in_axis, out_axis)


def he_normal_init(
    key: Array,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
    in_axis: int = -2,
    out_axis: int = -1,
) -> Array:
    """He/Kaiming normal for ReLU nets: ``N(0, 2 / fan_in)``."""
    return _variance_scaling(key, shape, 2.0, "fan_in", "normal", dtype, in_axis, out_axis)


def lecun_uniform_init(
    key: Array,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
    in_axis: int = -2,
    out_axis: int = -1,
) -> Array:
    """LeCun uniform (SELU nets): variance ``1 / fan_in``."""
    return _variance_scaling(key, shape, 1.0, "fan_in", "uniform", dtype, in_axis, out_axis)


def lecun_normal_init(
    key: Array,
    shape: tuple[int, ...],
    dtype: Any = jnp.float32,
    in_axis: int = -2,
    out_axis: int = -1,
) -> Array:
    """LeCun normal (SELU nets): ``N(0, 1 / fan_in)``."""
    return _variance_scaling(key, shape, 1.0, "fan_in", "normal", dtype, in_axis, out_axis)


def orthogonal_init(
    key: Array, shape: tuple[int, ...], dtype: Any = jnp.float32, scale: float = 1.0
) -> Array:
    """Orthogonal matrix (via QR of a Gaussian) reshaped to ``shape``.

    Rows (or columns, whichever is shorter) are orthonormal, which preserves
    the norm of activations layer to layer - useful for deep nets and RNNs.
    """
    if len(shape) < 2:
        raise ValueError("orthogonal_init needs at least a 2-D shape")
    rows = math.prod(shape[:-1])
    cols = shape[-1]
    n_min, n_max = min(rows, cols), max(rows, cols)
    a = jr.normal(key, (n_max, n_min), dtype)
    q, r = jnp.linalg.qr(a)
    q = q * jnp.sign(jnp.diagonal(r))  # make the decomposition unique
    if rows < cols:
        q = q.T
    return (scale * q).reshape(shape)
