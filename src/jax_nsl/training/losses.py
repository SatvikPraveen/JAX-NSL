# File location: src/jax_nsl/training/losses.py

"""
Loss functions that take *logits* where possible and reduce consistently.

All losses accept ``reduction`` in ``{'mean', 'sum', 'none'}`` and, where it
makes sense, per-example ``weights``.  Working with logits (rather than
probabilities) lets us use the stable log-softmax and avoids ``log(0)``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from jax_nsl.core.numerics import log_softmax_stable, safe_log

Array = jax.Array


def _reduce(loss: Array, reduction: str, weights: Array | None = None) -> Array:
    if weights is not None:
        loss = loss * weights
    if reduction == "mean":
        if weights is not None:
            return jnp.sum(loss) / jnp.maximum(jnp.sum(weights), 1e-12)
        return jnp.mean(loss)
    if reduction == "sum":
        return jnp.sum(loss)
    if reduction == "none":
        return loss
    raise ValueError(f"Unknown reduction {reduction!r}")


def _one_hot_like(labels: Array, logits: Array) -> Array:
    if labels.ndim == logits.ndim - 1:
        return jax.nn.one_hot(labels, logits.shape[-1], dtype=logits.dtype)
    return labels.astype(logits.dtype)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def cross_entropy_loss(
    logits: Array,
    labels: Array,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    weights: Array | None = None,
) -> Array:
    """Softmax cross-entropy from logits with optional label smoothing.

    Args:
        logits: ``(..., num_classes)``.
        labels: Integer classes ``(...)`` or one-hot/soft targets ``(..., num_classes)``.
        reduction: ``'mean'``, ``'sum'`` or ``'none'``.
        label_smoothing: Mix targets with the uniform distribution:
            ``(1 - eps) * onehot + eps / K``.
        weights: Optional per-example weights (``'mean'`` normalises by their sum).
    """
    targets = _one_hot_like(labels, logits)
    if label_smoothing > 0.0:
        k = logits.shape[-1]
        targets = (1.0 - label_smoothing) * targets + label_smoothing / k
    loss = -jnp.sum(targets * log_softmax_stable(logits), axis=-1)
    return _reduce(loss, reduction, weights)


def binary_cross_entropy(
    logits: Array, labels: Array, reduction: str = "mean", pos_weight: float | None = None
) -> Array:
    """Sigmoid cross-entropy from logits, ``max(x, 0) - x y + log(1 + exp(-|x|))``.

    That rearrangement never evaluates ``exp`` of a large positive number.
    """
    loss = jnp.maximum(logits, 0.0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    if pos_weight is not None:
        loss = loss * (pos_weight * labels + (1.0 - labels))
    return _reduce(loss, reduction)


def focal_loss(
    logits: Array,
    labels: Array,
    alpha: float | Array = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Array:
    """Multi-class focal loss ``-alpha_t (1 - p_t)^gamma log p_t``.

    ``alpha`` may be a scalar (applied to every class) or a per-class vector;
    it is looked up for the *true* class only.  ``p_t`` comes from the stable
    log-softmax so that confident predictions do not underflow to ``log 0``.
    """
    targets = _one_hot_like(labels, logits)
    log_p = log_softmax_stable(logits)
    log_p_t = jnp.sum(targets * log_p, axis=-1)
    p_t = jnp.exp(log_p_t)
    alpha_t = jnp.sum(
        jnp.broadcast_to(jnp.asarray(alpha, logits.dtype), logits.shape[-1:]) * targets, axis=-1
    )
    loss = -alpha_t * (1.0 - p_t) ** gamma * log_p_t
    return _reduce(loss, reduction)


def kl_divergence(p_logits: Array, q_logits: Array, reduction: str = "mean") -> Array:
    """``KL(p || q)`` between the softmax distributions of two logit arrays."""
    log_p = log_softmax_stable(p_logits)
    log_q = log_softmax_stable(q_logits)
    kl = jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1)
    return _reduce(kl, reduction)


def dice_loss(
    probabilities: Array, targets: Array, smooth: float = 1.0, reduction: str = "mean"
) -> Array:
    """``1 - 2|A n B| / (|A| + |B|)`` per example over flattened spatial dims (segmentation)."""
    p = probabilities.reshape(probabilities.shape[0], -1)
    t = targets.reshape(targets.shape[0], -1).astype(p.dtype)
    intersection = jnp.sum(p * t, axis=1)
    union = jnp.sum(p, axis=1) + jnp.sum(t, axis=1)
    return _reduce(1.0 - (2.0 * intersection + smooth) / (union + smooth), reduction)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def mse_loss(predictions: Array, targets: Array, reduction: str = "mean") -> Array:
    """Mean squared error."""
    return _reduce((predictions - targets) ** 2, reduction)


def huber_loss(
    predictions: Array, targets: Array, delta: float = 1.0, reduction: str = "mean"
) -> Array:
    """Quadratic for ``|r| <= delta``, linear beyond (robust to outliers)."""
    r = jnp.abs(predictions - targets)
    loss = jnp.where(r <= delta, 0.5 * r**2, delta * (r - 0.5 * delta))
    return _reduce(loss, reduction)


def smooth_l1_loss(
    predictions: Array, targets: Array, beta: float = 1.0, reduction: str = "mean"
) -> Array:
    """Huber loss divided by ``beta`` (the object-detection convention)."""
    r = jnp.abs(predictions - targets)
    loss = jnp.where(r < beta, 0.5 * r**2 / beta, r - 0.5 * beta)
    return _reduce(loss, reduction)


def quantile_loss(
    predictions: Array, targets: Array, quantile: float = 0.5, reduction: str = "mean"
) -> Array:
    """Pinball loss; minimised by the ``quantile``-th conditional quantile."""
    r = targets - predictions
    loss = jnp.maximum(quantile * r, (quantile - 1.0) * r)
    return _reduce(loss, reduction)


def mean_squared_residual(residual_fn: Callable[..., Array], *args: Any) -> Array:
    """``mean(residual_fn(*args) ** 2)`` - the physics-informed (PDE residual) loss."""
    return jnp.mean(jnp.square(residual_fn(*args)))


# ---------------------------------------------------------------------------
# Metric learning
# ---------------------------------------------------------------------------


def contrastive_loss(
    embeddings1: Array,
    embeddings2: Array,
    labels: Array,
    margin: float = 1.0,
    reduction: str = "mean",
) -> Array:
    """Hadsell et al. pairwise loss: pull similar pairs together, push others past ``margin``."""
    d = jnp.linalg.norm(embeddings1 - embeddings2, axis=-1)
    loss = 0.5 * (labels * d**2 + (1.0 - labels) * jnp.maximum(0.0, margin - d) ** 2)
    return _reduce(loss, reduction)


def triplet_loss(
    anchor: Array, positive: Array, negative: Array, margin: float = 1.0, reduction: str = "mean"
) -> Array:
    """``max(0, d(a, p) - d(a, n) + margin)``."""
    d_pos = jnp.linalg.norm(anchor - positive, axis=-1)
    d_neg = jnp.linalg.norm(anchor - negative, axis=-1)
    return _reduce(jnp.maximum(0.0, d_pos - d_neg + margin), reduction)


def cosine_similarity_loss(
    embeddings1: Array, embeddings2: Array, labels: Array, reduction: str = "mean"
) -> Array:
    """Squared error between the cosine similarity and a target in ``[-1, 1]``."""
    n1 = embeddings1 / (jnp.linalg.norm(embeddings1, axis=-1, keepdims=True) + 1e-8)
    n2 = embeddings2 / (jnp.linalg.norm(embeddings2, axis=-1, keepdims=True) + 1e-8)
    return _reduce((jnp.sum(n1 * n2, axis=-1) - labels) ** 2, reduction)


def info_nce_loss(queries: Array, keys: Array, temperature: float = 0.1) -> Array:
    """InfoNCE / NT-Xent: each query's positive is the key at the same index.

    Uses the stable log-softmax over the similarity matrix; ``safe_log`` is
    not needed because we never form probabilities explicitly.
    """
    q = queries / (jnp.linalg.norm(queries, axis=-1, keepdims=True) + 1e-8)
    k = keys / (jnp.linalg.norm(keys, axis=-1, keepdims=True) + 1e-8)
    logits = q @ k.T / temperature
    labels = jnp.arange(queries.shape[0])
    return cross_entropy_loss(logits, labels)


__all__ = [
    "cross_entropy_loss",
    "binary_cross_entropy",
    "focal_loss",
    "kl_divergence",
    "dice_loss",
    "mse_loss",
    "huber_loss",
    "smooth_l1_loss",
    "quantile_loss",
    "mean_squared_residual",
    "contrastive_loss",
    "triplet_loss",
    "cosine_similarity_loss",
    "info_nce_loss",
    "safe_log",
]
