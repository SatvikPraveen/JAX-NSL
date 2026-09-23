# File location: src/jax_nsl/models/mlp.py

"""
Multi-layer perceptrons in pure JAX.

Design rule: *parameters are pytrees of arrays, configuration is Python*.
Mixing the two (e.g. storing an activation name inside the params dict)
breaks ``jax.grad``, ``jit`` donation and optimisers, all of which map over
every leaf.  Configuration is therefore passed as keyword arguments or
captured in closures by :func:`create_mlp`.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr

from jax_nsl.core.prng import glorot_uniform_init, he_normal_init, lecun_normal_init

Array = jax.Array
Params = dict[str, list[Array]]

_INITIALIZERS: dict[str, Callable] = {
    "glorot": glorot_uniform_init,
    "he": he_normal_init,
    "lecun": lecun_normal_init,
    "normal": lambda key, shape, dtype=jnp.float32: 0.1 * jr.normal(key, shape, dtype),
}

_ACTIVATIONS: dict[str, Callable[[Array], Array]] = {
    "relu": jax.nn.relu,
    "tanh": jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "gelu": jax.nn.gelu,
    "swish": jax.nn.silu,
    "silu": jax.nn.silu,
    "elu": jax.nn.elu,
    "selu": jax.nn.selu,
    "softplus": jax.nn.softplus,
    "leaky_relu": jax.nn.leaky_relu,
    "linear": lambda x: x,
    "identity": lambda x: x,
    "softmax": lambda x: jax.nn.softmax(x, axis=-1),
}


def activation_fn(x: Array, name: str) -> Array:
    """Apply the activation called ``name`` (see ``_ACTIVATIONS`` for the list)."""
    try:
        return _ACTIVATIONS[name](x)
    except KeyError as e:
        raise ValueError(f"Unknown activation {name!r}; choose from {sorted(_ACTIVATIONS)}") from e


def init_mlp_params(
    key: Array, layer_sizes: Sequence[int], init_type: str = "glorot", dtype: Any = jnp.float32
) -> Params:
    """Initialise ``{'weights': [W_1, ...], 'biases': [b_1, ...]}`` for the given sizes.

    Args:
        key: PRNG key.
        layer_sizes: ``[in, hidden_1, ..., out]``.
        init_type: ``'glorot'`` (tanh/sigmoid nets), ``'he'`` (ReLU nets),
            ``'lecun'`` (SELU nets) or ``'normal'``.
        dtype: Parameter dtype.
    """
    if init_type not in _INITIALIZERS:
        raise ValueError(f"Unknown initialization {init_type!r}")
    init = _INITIALIZERS[init_type]
    keys = jr.split(key, len(layer_sizes) - 1)
    weights = [
        init(k, (n_in, n_out), dtype)
        for k, n_in, n_out in zip(keys, layer_sizes[:-1], layer_sizes[1:])
    ]
    biases = [jnp.zeros(n_out, dtype) for n_out in layer_sizes[1:]]
    return {"weights": weights, "biases": biases}


def dense_layer(x: Array, weights: Array, bias: Array, activation: str = "linear") -> Array:
    """``activation(x @ W + b)``."""
    return activation_fn(x @ weights + bias, activation)


def mlp_forward(
    params: Params, x: Array, activation: str = "relu", output_activation: str = "linear"
) -> Array:
    """Forward pass: hidden layers use ``activation``, the last layer ``output_activation``."""
    weights, biases = params["weights"], params["biases"]
    h = x
    for w, b in zip(weights[:-1], biases[:-1]):
        h = dense_layer(h, w, b, activation)
    return dense_layer(h, weights[-1], biases[-1], output_activation)


mlp_predict = mlp_forward


def mlp_with_dropout(
    params: Params,
    x: Array,
    key: Array,
    dropout_rate: float = 0.1,
    training: bool = True,
    activation: str = "relu",
    output_activation: str = "linear",
) -> Array:
    """Forward pass with inverted dropout after every hidden activation.

    Inverted dropout scales the kept units by ``1 / keep_prob`` at training
    time so that no rescaling is needed at inference.
    """
    weights, biases = params["weights"], params["biases"]
    keys = jr.split(key, max(len(weights) - 1, 1))
    keep_prob = 1.0 - dropout_rate
    h = x
    for k, w, b in zip(keys, weights[:-1], biases[:-1]):
        h = dense_layer(h, w, b, activation)
        if training and dropout_rate > 0.0:
            mask = jr.bernoulli(k, keep_prob, h.shape)
            h = jnp.where(mask, h / keep_prob, 0.0)
    return dense_layer(h, weights[-1], biases[-1], output_activation)


# ---------------------------------------------------------------------------
# Batch normalisation with running statistics
# ---------------------------------------------------------------------------


def init_batch_norm(
    num_features: int, dtype: Any = jnp.float32
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Return ``(params, state)``: learnable ``scale``/``bias`` and running ``mean``/``var``."""
    params = {"scale": jnp.ones(num_features, dtype), "bias": jnp.zeros(num_features, dtype)}
    state = {"mean": jnp.zeros(num_features, dtype), "var": jnp.ones(num_features, dtype)}
    return params, state


def batch_norm(
    x: Array,
    params: dict[str, Array],
    state: dict[str, Array],
    training: bool,
    momentum: float = 0.9,
    epsilon: float = 1e-5,
    axes: tuple[int, ...] = (0,),
) -> tuple[Array, dict[str, Array]]:
    """Batch normalisation over ``axes``; returns ``(output, new_state)``.

    Running statistics are *state*, not parameters: they are updated by an
    exponential moving average during training and used verbatim at
    inference.  Keeping them separate from ``params`` means the optimiser
    never touches them.
    """
    if training:
        mean = jnp.mean(x, axis=axes)
        var = jnp.var(x, axis=axes)
        new_state = {
            "mean": momentum * state["mean"] + (1 - momentum) * mean,
            "var": momentum * state["var"] + (1 - momentum) * var,
        }
    else:
        mean, var = state["mean"], state["var"]
        new_state = state
    x_hat = (x - mean) / jnp.sqrt(var + epsilon)
    return params["scale"] * x_hat + params["bias"], new_state


def init_mlp_with_batch_norm(
    key: Array, layer_sizes: Sequence[int], init_type: str = "he"
) -> tuple[dict[str, Any], dict[str, Any]]:
    """MLP parameters plus one batch-norm (params, state) pair per hidden layer."""
    params = init_mlp_params(key, layer_sizes, init_type)
    bn = [init_batch_norm(n) for n in layer_sizes[1:-1]]
    params["bn"] = [p for p, _ in bn]
    state = {"bn": [s for _, s in bn]}
    return params, state


def mlp_with_batch_norm(
    params: dict[str, Any],
    state: dict[str, Any],
    x: Array,
    training: bool,
    activation: str = "relu",
    output_activation: str = "linear",
) -> tuple[Array, dict[str, Any]]:
    """Dense -> BatchNorm -> activation for each hidden layer; returns ``(out, new_state)``."""
    weights, biases = params["weights"], params["biases"]
    new_bn_states = []
    h = x
    for i, (w, b) in enumerate(zip(weights[:-1], biases[:-1])):
        h = h @ w + b
        h, s = batch_norm(h, params["bn"][i], state["bn"][i], training)
        new_bn_states.append(s)
        h = activation_fn(h, activation)
    out = dense_layer(h, weights[-1], biases[-1], output_activation)
    return out, {"bn": new_bn_states}


# ---------------------------------------------------------------------------
# Factories and inspection
# ---------------------------------------------------------------------------


def create_mlp(
    layer_sizes: Sequence[int],
    activation: str = "relu",
    output_activation: str = "linear",
    init_type: str = "glorot",
    seed: int = 42,
) -> tuple[Params, Callable, Callable]:
    """``(params, forward_fn, predict_fn)`` with the configuration captured in closures.

    ``forward_fn(params, x, training=True)`` keeps a ``training`` flag for
    API symmetry with models that have train/eval behaviour.
    """
    params = init_mlp_params(jr.PRNGKey(seed), layer_sizes, init_type)
    forward = functools.partial(
        mlp_forward, activation=activation, output_activation=output_activation
    )

    def forward_fn(params, x, training=True):
        return forward(params, x)

    def predict_fn(params, x):
        return forward(params, x)

    return params, forward_fn, predict_fn


def create_classifier(
    input_dim: int,
    hidden_dims: Sequence[int],
    num_classes: int,
    activation: str = "relu",
    seed: int = 42,
) -> tuple[Params, Callable]:
    """MLP returning *logits* (use a cross-entropy loss that takes logits)."""
    params, forward_fn, _ = create_mlp(
        [input_dim, *hidden_dims, num_classes],
        activation,
        "linear",
        "he" if activation == "relu" else "glorot",
        seed,
    )
    return params, forward_fn


def create_regressor(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int = 1,
    activation: str = "relu",
    seed: int = 42,
) -> tuple[Params, Callable]:
    """MLP with a linear output layer."""
    params, forward_fn, _ = create_mlp(
        [input_dim, *hidden_dims, output_dim],
        activation,
        "linear",
        "he" if activation == "relu" else "glorot",
        seed,
    )
    return params, forward_fn


def count_parameters(params: Any) -> int:
    """Total number of scalars in a parameter pytree."""
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(params)))


def get_layer_outputs(
    params: Params, x: Array, activation: str = "relu", output_activation: str = "linear"
) -> list[Array]:
    """Return ``[x, h_1, ..., output]`` for inspection of activations."""
    weights, biases = params["weights"], params["biases"]
    outputs = [x]
    h = x
    for w, b in zip(weights[:-1], biases[:-1]):
        h = dense_layer(h, w, b, activation)
        outputs.append(h)
    outputs.append(dense_layer(h, weights[-1], biases[-1], output_activation))
    return outputs
