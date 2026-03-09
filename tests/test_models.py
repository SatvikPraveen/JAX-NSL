# tests/test_models.py
"""Tests for src/models: MLP, CNN, and Transformer building blocks."""

import jax
import jax.numpy as jnp
import pytest
from jax import random, grad

from models.mlp import (
    init_mlp_params, mlp_forward, mlp_predict, activation_fn, create_mlp
)
from models.cnn import init_conv_params, conv2d_layer, pooling_layer
from models.transformer import (
    init_attention_params, scaled_dot_product_attention,
    multi_head_attention, layer_norm, feed_forward_network,
)


# ============================================================
# MLP
# ============================================================

class TestMLP:
    def test_init_shape(self):
        key = random.PRNGKey(0)
        params = init_mlp_params(key, [4, 8, 3])
        assert len(params["weights"]) == 2
        assert params["weights"][0].shape == (4, 8)
        assert params["weights"][1].shape == (8, 3)

    def test_forward_output_shape(self):
        key = random.PRNGKey(0)
        params = init_mlp_params(key, [4, 8, 3])
        x = jnp.ones((16, 4))
        out = mlp_forward(params, x)
        assert out.shape == (16, 3)

    def test_predict_agrees_with_forward(self):
        key = random.PRNGKey(0)
        params = init_mlp_params(key, [4, 8, 3])
        x = jnp.ones((8, 4))
        assert jnp.allclose(mlp_predict(params, x), mlp_forward(params, x, training=False))

    def test_differentiable(self):
        key = random.PRNGKey(0)
        params = init_mlp_params(key, [4, 8, 2])
        x = jnp.ones((8, 4))
        labels = jnp.zeros(8, dtype=jnp.int32)

        def loss(p):
            logits = mlp_forward(p, x)
            return jnp.mean((logits - jax.nn.one_hot(labels, 2)) ** 2)

        grads = grad(loss)(params)
        assert grads["weights"][0].shape == params["weights"][0].shape

    @pytest.mark.parametrize("act", ["relu", "tanh", "sigmoid", "gelu", "swish"])
    def test_activations(self, act):
        x = jnp.linspace(-2, 2, 10)
        out = activation_fn(x, act)
        assert out.shape == x.shape
        assert jnp.all(jnp.isfinite(out))

    def test_create_mlp(self):
        params, fwd, predict = create_mlp([4, 8, 3], seed=0)
        x = jnp.ones((5, 4))
        assert fwd(params, x).shape == (5, 3)


# ============================================================
# CNN
# ============================================================

class TestCNN:
    def test_init_conv_params(self):
        key = random.PRNGKey(0)
        p = init_conv_params(key, (3, 3), in_channels=1, out_channels=8)
        assert p["weights"].shape == (8, 1, 3, 3)
        assert p["biases"].shape == (8,)

    def test_conv2d_output_shape(self):
        key = random.PRNGKey(0)
        params = init_conv_params(key, (3, 3), in_channels=1, out_channels=4)
        # Input NHWC: (batch=2, H=8, W=8, C=1)
        x = jnp.ones((2, 8, 8, 1))
        out = conv2d_layer(x, params, stride=(1, 1), padding="SAME")
        # Output should have same H/W with SAME padding
        assert out.shape[0] == 2
        assert out.shape[-1] == 4

    def test_pooling_reduces_spatial(self):
        x = jnp.ones((2, 8, 8, 4))
        out = pooling_layer(x, pool_type="max", window_shape=(2, 2), stride=(2, 2))
        assert out.shape == (2, 4, 4, 4)


# ============================================================
# Transformer components
# ============================================================

class TestTransformer:
    def test_init_attention_params(self):
        key = random.PRNGKey(0)
        p = init_attention_params(key, d_model=16, num_heads=2)
        assert "query" in p
        assert "key" in p
        assert "value" in p
        assert "out" in p

    def test_scaled_dot_product_attention_shape(self):
        key = random.PRNGKey(0)
        q = random.normal(key, (2, 4, 8))  # (batch, seq, d_k)
        k = random.normal(key, (2, 4, 8))
        v = random.normal(key, (2, 4, 8))
        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape == (2, 4, 8)
        assert jnp.allclose(weights.sum(axis=-1), jnp.ones((2, 4)), atol=1e-5)

    def test_layer_norm_output(self):
        key = random.PRNGKey(0)
        x = random.normal(key, (4, 8))
        params = {
            "scale": jnp.ones(8),
            "bias": jnp.zeros(8),
        }
        out = layer_norm(x, params["scale"], params["bias"])
        assert out.shape == x.shape
        # Normalised rows should have near-zero mean and unit std
        assert jnp.allclose(out.mean(axis=-1), jnp.zeros(4), atol=1e-4)

    def test_feed_forward_shape(self):
        key = random.PRNGKey(0)
        x = random.normal(key, (4, 16))
        k1, k2 = random.split(key)
        ff_params = {
            "W1": random.normal(k1, (16, 32)),
            "b1": jnp.zeros(32),
            "W2": random.normal(k2, (32, 16)),
            "b2": jnp.zeros(16),
        }
        out = feed_forward_network(x, ff_params)
        assert out.shape == (4, 16)
