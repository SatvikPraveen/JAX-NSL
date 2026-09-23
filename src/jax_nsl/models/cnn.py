# File location: src/jax_nsl/models/cnn.py

"""
Convolutional networks with ``lax.conv_general_dilated``.

Layout: activations are ``NHWC`` and kernels ``OIHW``.  ``conv_general_dilated``
takes an explicit ``dimension_numbers`` triple, so no transposes are needed
to mix the two - transposing on every layer is a common source of needless
copies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax

from jax_nsl.core.prng import glorot_uniform_init, he_normal_init
from jax_nsl.models.mlp import activation_fn

Array = jax.Array
_DN = ("NHWC", "OIHW", "NHWC")


def init_conv_params(
    key: Array,
    kernel_shape: tuple[int, int],
    in_channels: int,
    out_channels: int,
    init_type: str = "he",
    dtype: Any = jnp.float32,
) -> dict[str, Array]:
    """Kernel ``(out, in, kh, kw)`` and bias ``(out,)``.

    Fan-in for the initialiser is ``in * kh * kw`` (the receptive field), as
    in :func:`jax_nsl.core.prng.compute_fans`.
    """
    w_key, _ = jr.split(key)
    shape = (out_channels, in_channels) + tuple(kernel_shape)
    if init_type == "he":
        weights = he_normal_init(w_key, shape, dtype, in_axis=1, out_axis=0)
    elif init_type == "glorot":
        weights = glorot_uniform_init(w_key, shape, dtype, in_axis=1, out_axis=0)
    else:
        weights = 0.1 * jr.normal(w_key, shape, dtype)
    return {"weights": weights, "biases": jnp.zeros(out_channels, dtype)}


def conv2d(
    x: Array,
    weights: Array,
    stride: tuple[int, int] = (1, 1),
    padding: str = "SAME",
    dilation: tuple[int, int] = (1, 1),
    feature_group_count: int = 1,
) -> Array:
    """Raw 2-D convolution, ``x: NHWC``, ``weights: OIHW``."""
    return lax.conv_general_dilated(
        x,
        weights,
        window_strides=stride,
        padding=padding,
        rhs_dilation=dilation,
        dimension_numbers=_DN,
        feature_group_count=feature_group_count,
    )


def conv2d_layer(
    x: Array,
    params: dict[str, Array],
    stride: tuple[int, int] = (1, 1),
    padding: str = "SAME",
    dilation: tuple[int, int] = (1, 1),
    activation: str = "relu",
) -> Array:
    """Convolution + bias + activation."""
    out = conv2d(x, params["weights"], stride, padding, dilation) + params["biases"]
    return activation_fn(out, activation)


def conv_output_shape(
    input_hw: tuple[int, int], kernel: tuple[int, int], stride: tuple[int, int], padding: str
) -> tuple[int, int]:
    """Spatial output size for ``'SAME'`` or ``'VALID'`` padding."""
    if padding == "SAME":
        return tuple(-(-h // s) for h, s in zip(input_hw, stride))
    return tuple((h - k) // s + 1 for h, k, s in zip(input_hw, kernel, stride))


def pooling_layer(
    x: Array,
    pool_type: str = "max",
    window_shape: tuple[int, int] = (2, 2),
    stride: tuple[int, int] = (2, 2),
    padding: str = "VALID",
) -> Array:
    """Max or average pooling over the spatial axes of an ``NHWC`` array."""
    dims = (1,) + tuple(window_shape) + (1,)
    strides = (1,) + tuple(stride) + (1,)
    if pool_type == "max":
        return lax.reduce_window(x, -jnp.inf, lax.max, dims, strides, padding)
    if pool_type == "avg":
        summed = lax.reduce_window(x, 0.0, lax.add, dims, strides, padding)
        if padding == "VALID":
            return summed / (window_shape[0] * window_shape[1])
        counts = lax.reduce_window(jnp.ones_like(x), 0.0, lax.add, dims, strides, padding)
        return summed / counts
    raise ValueError(f"Unknown pool type: {pool_type}")


def global_average_pooling(x: Array) -> Array:
    """``NHWC -> NC`` by averaging over H and W."""
    return jnp.mean(x, axis=(1, 2))


# ---------------------------------------------------------------------------
# Batch normalisation for feature maps
# ---------------------------------------------------------------------------


def init_batch_norm_2d(
    channels: int, dtype: Any = jnp.float32
) -> tuple[dict[str, Array], dict[str, Array]]:
    """``(params, state)`` for per-channel batch norm."""
    params = {"scale": jnp.ones(channels, dtype), "offset": jnp.zeros(channels, dtype)}
    state = {"running_mean": jnp.zeros(channels, dtype), "running_var": jnp.ones(channels, dtype)}
    return params, state


def batch_norm_2d(
    x: Array,
    params: dict[str, Array],
    state: dict[str, Array],
    training: bool = True,
    momentum: float = 0.9,
    epsilon: float = 1e-5,
) -> tuple[Array, dict[str, Array]]:
    """Per-channel batch norm over ``(N, H, W)``; returns ``(out, new_state)``."""
    if training:
        mean = jnp.mean(x, axis=(0, 1, 2))
        var = jnp.var(x, axis=(0, 1, 2))
        new_state = {
            "running_mean": momentum * state["running_mean"] + (1 - momentum) * mean,
            "running_var": momentum * state["running_var"] + (1 - momentum) * var,
        }
    else:
        mean, var = state["running_mean"], state["running_var"]
        new_state = state
    x_hat = (x - mean) / jnp.sqrt(var + epsilon)
    return params["scale"] * x_hat + params["offset"], new_state


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def depthwise_conv2d(
    x: Array, kernel: Array, stride: tuple[int, int] = (1, 1), padding: str = "SAME"
) -> Array:
    """Depthwise convolution; ``kernel`` is ``(kh, kw, in_channels, multiplier)``."""
    kh, kw, in_ch, mult = kernel.shape
    weights = kernel.transpose(2, 3, 0, 1).reshape(in_ch * mult, 1, kh, kw)
    return conv2d(x, weights, stride, padding, feature_group_count=in_ch)


def separable_conv2d(
    x: Array,
    depthwise_kernel: Array,
    pointwise_kernel: Array,
    stride: tuple[int, int] = (1, 1),
    padding: str = "SAME",
) -> Array:
    """Depthwise conv followed by a 1x1 conv; ``pointwise_kernel`` is ``(out, in*mult, 1, 1)``."""
    h = depthwise_conv2d(x, depthwise_kernel, stride, padding)
    return conv2d(h, pointwise_kernel)


def residual_block(
    x: Array,
    params1: dict[str, Array],
    params2: dict[str, Array],
    shortcut_params: dict[str, Array] | None = None,
) -> Array:
    """``relu(conv2(relu(conv1(x))) + shortcut(x))``."""
    h = conv2d_layer(x, params1, activation="relu")
    h = conv2d_layer(h, params2, activation="linear")
    skip = x if shortcut_params is None else conv2d_layer(x, shortcut_params, activation="linear")
    return jax.nn.relu(h + skip)


# ---------------------------------------------------------------------------
# A small classifier
# ---------------------------------------------------------------------------


def cnn_forward(params: dict[str, Any], x: Array, training: bool = True) -> Array:
    """conv(3x3, SAME) -> relu -> maxpool(2) per conv layer, then dense layers (logits out)."""
    h = x
    for layer in params["conv_layers"]:
        h = conv2d_layer(h, layer, activation="relu")
        h = pooling_layer(h, "max")
    h = h.reshape(h.shape[0], -1)
    dense = params["dense_layers"]
    for i, layer in enumerate(dense):
        h = h @ layer["weights"] + layer["biases"]
        if i < len(dense) - 1:
            h = jax.nn.relu(h)
    return h


def create_cnn(
    input_shape: tuple[int, int, int],
    conv_channels: Sequence[int],
    dense_layers: Sequence[int],
    num_classes: int,
    kernel: tuple[int, int] = (3, 3),
    seed: int = 42,
) -> tuple[dict[str, Any], Callable]:
    """Build the parameters for :func:`cnn_forward`.

    The flattened feature size after the conv stack is *computed* by tracing
    the conv layers with ``jax.eval_shape`` on a dummy input - no hard-coded
    magic number.

    Args:
        input_shape: ``(H, W, C)`` of one example.
        conv_channels: Output channels of each conv layer.
        dense_layers: Hidden sizes of the dense head.
        num_classes: Output size.
        kernel: Conv kernel size.
        seed: PRNG seed.
    """
    key = jr.PRNGKey(seed)
    keys = jr.split(key, len(conv_channels) + len(dense_layers) + 1)

    conv_params: list[dict[str, Array]] = []
    in_channels = input_shape[-1]
    for k, out_channels in zip(keys, conv_channels):
        conv_params.append(init_conv_params(k, kernel, in_channels, out_channels))
        in_channels = out_channels

    def conv_stack(x):
        for layer in conv_params:
            x = pooling_layer(conv2d_layer(x, layer), "max")
        return x.reshape(x.shape[0], -1)

    flat = jax.eval_shape(conv_stack, jax.ShapeDtypeStruct((1,) + tuple(input_shape), jnp.float32))
    in_features = flat.shape[-1]

    dense_params: list[dict[str, Array]] = []
    for k, out_features in zip(keys[len(conv_channels) :], [*dense_layers, num_classes]):
        dense_params.append(
            {
                "weights": glorot_uniform_init(k, (in_features, out_features)),
                "biases": jnp.zeros(out_features),
            }
        )
        in_features = out_features

    params = {"conv_layers": conv_params, "dense_layers": dense_params}
    return params, cnn_forward
