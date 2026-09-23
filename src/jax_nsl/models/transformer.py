# File location: src/jax_nsl/models/transformer.py

"""
Transformer components from scratch: attention, feed-forward, normalisation,
positional encodings, and a layer stack that runs as a single ``scan``.

Numerical details that matter:

* Masked scores use the dtype's most negative finite value, not ``-inf``:
  a row that is *entirely* masked (a fully padded query) would otherwise
  become ``softmax([-inf, ...]) = nan``.
* Attention logits are computed in float32 even when activations are
  bfloat16 (``preferred_element_type``), because the softmax is sensitive to
  the ~3 significant digits of bf16.
* The layer stack is stored as one pytree with a leading layer axis and
  applied with ``lax.scan`` (optionally under ``jax.checkpoint``), so the
  compiled program is independent of depth.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from jax_nsl.core.prng import glorot_uniform_init

Array = jax.Array


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


def init_attention_params(
    key: Array, d_model: int, num_heads: int, dtype: Any = jnp.float32
) -> dict[str, Array]:
    """Projection matrices ``query``, ``key``, ``value``, ``out`` (all ``d_model x d_model``)."""
    if d_model % num_heads:
        raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
    keys = jr.split(key, 4)
    return {
        name: glorot_uniform_init(k, (d_model, d_model), dtype)
        for name, k in zip(("query", "key", "value", "out"), keys)
    }


def init_feed_forward_params(
    key: Array, d_model: int, d_ff: int, dtype: Any = jnp.float32
) -> dict[str, Array]:
    """``W1 (d_model x d_ff), b1, W2 (d_ff x d_model), b2``."""
    k1, k2 = jr.split(key)
    return {
        "W1": glorot_uniform_init(k1, (d_model, d_ff), dtype),
        "b1": jnp.zeros(d_ff, dtype),
        "W2": glorot_uniform_init(k2, (d_ff, d_model), dtype),
        "b2": jnp.zeros(d_model, dtype),
    }


def init_layer_norm_params(d: int, dtype: Any = jnp.float32) -> dict[str, Array]:
    """``scale`` ones and ``bias`` zeros."""
    return {"scale": jnp.ones(d, dtype), "bias": jnp.zeros(d, dtype)}


def init_transformer_block_params(
    key: Array, d_model: int, num_heads: int, d_ff: int | None = None, dtype: Any = jnp.float32
) -> dict[str, Any]:
    """Nested params for one block: ``attention``, ``ffn``, ``ln1``, ``ln2``."""
    d_ff = d_ff or 4 * d_model
    k_attn, k_ff = jr.split(key)
    return {
        "attention": init_attention_params(k_attn, d_model, num_heads, dtype),
        "ffn": init_feed_forward_params(k_ff, d_model, d_ff, dtype),
        "ln1": init_layer_norm_params(d_model, dtype),
        "ln2": init_layer_norm_params(d_model, dtype),
    }


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def layer_norm(x: Array, scale: Array, bias: Array, epsilon: float = 1e-6) -> Array:
    """Normalise over the last axis; statistics are computed in float32."""
    x32 = x.astype(jnp.float32)
    mean = jnp.mean(x32, axis=-1, keepdims=True)
    var = jnp.var(x32, axis=-1, keepdims=True)
    normalized = ((x32 - mean) * lax.rsqrt(var + epsilon)).astype(x.dtype)
    return scale * normalized + bias


def rms_norm(x: Array, scale: Array, epsilon: float = 1e-6) -> Array:
    """RMSNorm (no mean subtraction, no bias) as used in LLaMA-style models."""
    x32 = x.astype(jnp.float32)
    rms = lax.rsqrt(jnp.mean(jnp.square(x32), axis=-1, keepdims=True) + epsilon)
    return scale * (x32 * rms).astype(x.dtype)


# ---------------------------------------------------------------------------
# Positional information
# ---------------------------------------------------------------------------


def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> Array:
    """Sinusoidal encoding ``PE[pos, 2i] = sin(pos / base^(2i/d))``, ``PE[pos, 2i+1] = cos(...)``."""
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(base) / d_model))
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[: d_model // 2]))
    return pe


def rotary_embedding(x: Array, positions: Array | None = None, base: float = 10000.0) -> Array:
    """Rotary position embedding (RoPE) applied to the last axis of ``x``.

    Pairs of features ``(x[2i], x[2i+1])`` are rotated by an angle
    ``pos * base^(-2i/d)``; applying this to *both* queries and keys makes
    the dot product depend only on the relative position ``q_pos - k_pos``.

    Args:
        x: ``(..., seq_len, d)`` with even ``d``.
        positions: Optional ``(seq_len,)`` integer positions (default ``arange``).
    """
    seq_len, d = x.shape[-2], x.shape[-1]
    if positions is None:
        positions = jnp.arange(seq_len)
    inv_freq = base ** (-jnp.arange(0, d, 2, dtype=jnp.float32) / d)
    angles = positions[:, None].astype(jnp.float32) * inv_freq[None, :]  # (seq, d/2)
    cos, sin = jnp.cos(angles).astype(x.dtype), jnp.sin(angles).astype(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return out.reshape(x.shape)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def create_causal_mask(seq_len: int) -> Array:
    """Boolean ``(seq_len, seq_len)`` lower-triangular mask (True = may attend)."""
    return jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))


def create_padding_mask(tokens: Array, pad_token: int = 0) -> Array:
    """``(batch, 1, 1, seq)`` key-padding mask that broadcasts against ``(batch, heads, q, k)``."""
    return (tokens != pad_token)[:, None, None, :]


def combine_masks(*masks: Array | None) -> Array | None:
    """Logical AND of broadcastable boolean masks (``None`` entries ignored)."""
    present = [m for m in masks if m is not None]
    if not present:
        return None
    out = present[0]
    for m in present[1:]:
        out = jnp.logical_and(out, m)
    return out


def scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Array | None = None,
    dropout_rate: float = 0.0,
    key_rng: Array | None = None,
    training: bool = True,
) -> tuple[Array, Array]:
    """``softmax(Q K^T / sqrt(d_k)) V`` for inputs of shape ``(..., seq, d)``.

    Returns ``(output, attention_weights)``.  ``mask`` is boolean and must
    broadcast to ``(..., q_len, k_len)``; True means "may attend".
    """
    d_k = query.shape[-1]
    scores = jnp.einsum(
        "...qd,...kd->...qk", query, key, preferred_element_type=jnp.float32
    ) / math.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    weights = jax.nn.softmax(scores, axis=-1)
    if training and dropout_rate > 0.0 and key_rng is not None:
        keep = jr.bernoulli(key_rng, 1.0 - dropout_rate, weights.shape)
        weights = jnp.where(keep, weights / (1.0 - dropout_rate), 0.0)
    weights = weights.astype(value.dtype)
    return jnp.einsum("...qk,...kd->...qd", weights, value), weights


def _split_heads(x: Array, num_heads: int) -> Array:
    b, s, d = x.shape
    return x.reshape(b, s, num_heads, d // num_heads).transpose(0, 2, 1, 3)


def _merge_heads(x: Array) -> Array:
    b, h, s, dk = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, s, h * dk)


def multi_head_attention(
    x: Array,
    params: dict[str, Array],
    num_heads: int,
    mask: Array | None = None,
    context: Array | None = None,
    dropout_rate: float = 0.0,
    key_rng: Array | None = None,
    training: bool = True,
    rotary: bool = False,
) -> tuple[Array, Array]:
    """Multi-head (self- or cross-) attention.

    Args:
        x: Queries source ``(batch, q_len, d_model)``.
        params: From :func:`init_attention_params`.
        num_heads: Number of heads.
        mask: Boolean, broadcastable to ``(batch, heads, q_len, k_len)``.
        context: Keys/values source ``(batch, k_len, d_model)``; defaults to ``x``.
        dropout_rate, key_rng, training: Attention-weight dropout.
        rotary: Apply RoPE to queries and keys.

    Returns:
        ``(output (batch, q_len, d_model), weights (batch, heads, q_len, k_len))``.
    """
    kv = x if context is None else context
    q = _split_heads(x @ params["query"], num_heads)
    k = _split_heads(kv @ params["key"], num_heads)
    v = _split_heads(kv @ params["value"], num_heads)
    if rotary:
        q, k = rotary_embedding(q), rotary_embedding(k)
    out, weights = scaled_dot_product_attention(q, k, v, mask, dropout_rate, key_rng, training)
    return _merge_heads(out) @ params["out"], weights


def feed_forward_network(x: Array, params: dict[str, Array], activation: str = "gelu") -> Array:
    """``W2 act(W1 x + b1) + b2`` applied position-wise."""
    act = {"relu": jax.nn.relu, "gelu": jax.nn.gelu, "silu": jax.nn.silu}[activation]
    return act(x @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]


def transformer_block(
    x: Array,
    params: dict[str, Any],
    num_heads: int,
    mask: Array | None = None,
    dropout_rate: float = 0.0,
    key_rng: Array | None = None,
    training: bool = True,
    pre_norm: bool = True,
    rotary: bool = False,
) -> Array:
    """One encoder block.

    ``pre_norm=True`` (LayerNorm before each sub-layer, residual around it)
    trains stably without warmup and is the modern default; ``False`` gives
    the original post-norm formulation.
    """
    if pre_norm:
        h = layer_norm(x, **params["ln1"])
        attn, _ = multi_head_attention(
            h, params["attention"], num_heads, mask, None, dropout_rate, key_rng, training, rotary
        )
        x = x + attn
        h = layer_norm(x, **params["ln2"])
        return x + feed_forward_network(h, params["ffn"])
    attn, _ = multi_head_attention(
        x, params["attention"], num_heads, mask, None, dropout_rate, key_rng, training, rotary
    )
    x = layer_norm(x + attn, **params["ln1"])
    return layer_norm(x + feed_forward_network(x, params["ffn"]), **params["ln2"])


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


def create_transformer(
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int | None = None,
    max_seq_len: int = 1024,
    vocab_size: int | None = None,
    seed: int = 42,
    dtype: Any = jnp.float32,
    remat: bool = False,
    pre_norm: bool = True,
) -> tuple[dict[str, Any], Callable]:
    """Encoder stack whose layer parameters are *stacked* along a leading axis.

    Returns ``(params, forward_fn)``; ``forward_fn(params, x, mask=None,
    training=True, key_rng=None)`` accepts token ids ``(batch, seq)`` when
    ``vocab_size`` is set, or embeddings ``(batch, seq, d_model)``.

    Because all layers share one function, the forward pass is a
    ``lax.scan`` over the stacked parameters; with ``remat=True`` each layer
    is wrapped in ``jax.checkpoint`` so activation memory stays flat in depth.
    """
    keys = jr.split(jr.PRNGKey(seed), num_layers + 1)
    layer_params = [
        init_transformer_block_params(k, d_model, num_heads, d_ff, dtype) for k in keys[:num_layers]
    ]
    params: dict[str, Any] = {
        "layers": jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *layer_params),
        "final_ln": init_layer_norm_params(d_model, dtype),
        "pos_encoding": positional_encoding(max_seq_len, d_model).astype(dtype),
    }
    if vocab_size is not None:
        params["embedding"] = glorot_uniform_init(keys[-1], (vocab_size, d_model), dtype)

    def forward_fn(params, x, mask=None, training=True, key_rng=None, dropout_rate=0.0):
        return transformer_forward(
            params,
            x,
            num_heads,
            mask,
            training,
            key_rng,
            dropout_rate,
            remat=remat,
            pre_norm=pre_norm,
        )

    return params, forward_fn


def transformer_forward(
    params: dict[str, Any],
    x: Array,
    num_heads: int,
    mask: Array | None = None,
    training: bool = True,
    key_rng: Array | None = None,
    dropout_rate: float = 0.0,
    remat: bool = False,
    pre_norm: bool = True,
) -> Array:
    """Embed (if needed), add positions, scan over the stacked layers, final LayerNorm."""
    if "embedding" in params and x.ndim == 2:
        x = params["embedding"][x]
    seq_len = x.shape[1]
    x = x + params["pos_encoding"][:seq_len]

    num_layers = jax.tree_util.tree_leaves(params["layers"])[0].shape[0]
    layer_keys = (
        jr.split(key_rng, num_layers)
        if key_rng is not None
        else jnp.zeros((num_layers, 2), jnp.uint32)
    )

    def layer(carry, inputs):
        layer_params, k = inputs
        k = k if key_rng is not None else None
        return (
            transformer_block(
                carry, layer_params, num_heads, mask, dropout_rate, k, training, pre_norm
            ),
            None,
        )

    body = jax.checkpoint(layer) if remat else layer
    x, _ = lax.scan(body, x, (params["layers"], layer_keys))
    return layer_norm(x, **params["final_ln"]) if pre_norm else x
