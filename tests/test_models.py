# tests/test_models.py
"""Tests for jax_nsl.models: MLP, CNN and Transformer building blocks."""

import jax
import jax.numpy as jnp
import pytest
from jax import grad, random

from jax_nsl.models.cnn import (
    batch_norm_2d,
    conv2d_layer,
    conv_output_shape,
    create_cnn,
    depthwise_conv2d,
    init_batch_norm_2d,
    init_conv_params,
    pooling_layer,
    residual_block,
)
from jax_nsl.models.mlp import (
    activation_fn,
    count_parameters,
    create_classifier,
    create_mlp,
    init_mlp_params,
    init_mlp_with_batch_norm,
    mlp_forward,
    mlp_predict,
    mlp_with_batch_norm,
    mlp_with_dropout,
)
from jax_nsl.models.transformer import (
    create_causal_mask,
    create_padding_mask,
    create_transformer,
    feed_forward_network,
    init_attention_params,
    init_transformer_block_params,
    layer_norm,
    multi_head_attention,
    positional_encoding,
    rms_norm,
    rotary_embedding,
    scaled_dot_product_attention,
    transformer_block,
)

# ============================================================
# MLP
# ============================================================


class TestMLP:
    def test_init_shape(self):
        params = init_mlp_params(random.PRNGKey(0), [4, 8, 3])
        assert len(params["weights"]) == 2
        assert params["weights"][0].shape == (4, 8)
        assert params["weights"][1].shape == (8, 3)
        assert all(hasattr(leaf, "shape") for leaf in jax.tree_util.tree_leaves(params))

    def test_forward_output_shape(self):
        params = init_mlp_params(random.PRNGKey(0), [4, 8, 3])
        assert mlp_forward(params, jnp.ones((16, 4))).shape == (16, 3)
        assert jnp.allclose(mlp_predict(params, jnp.ones((8, 4))), mlp_forward(params, jnp.ones((8, 4))))

    def test_differentiable_and_jittable(self):
        params = init_mlp_params(random.PRNGKey(0), [4, 8, 2])
        x = jnp.ones((8, 4))

        def loss(p):
            return jnp.mean(mlp_forward(p, x) ** 2)

        grads = jax.jit(grad(loss))(params)
        assert grads["weights"][0].shape == params["weights"][0].shape
        assert jnp.any(grads["weights"][0] != 0)

    @pytest.mark.parametrize("act", ["relu", "tanh", "sigmoid", "gelu", "swish", "selu", "softplus"])
    def test_activations(self, act):
        x = jnp.linspace(-2, 2, 10)
        out = activation_fn(x, act)
        assert out.shape == x.shape and jnp.all(jnp.isfinite(out))

    def test_unknown_activation(self):
        with pytest.raises(ValueError):
            activation_fn(jnp.ones(2), "nope")

    def test_create_mlp_and_classifier(self):
        params, fwd, predict = create_mlp([4, 8, 3], seed=0)
        x = jnp.ones((5, 4))
        assert fwd(params, x).shape == (5, 3)
        assert jnp.allclose(fwd(params, x), predict(params, x))
        assert count_parameters(params) == 4 * 8 + 8 + 8 * 3 + 3
        cparams, cfwd = create_classifier(4, [6], 3)
        assert cfwd(cparams, x).shape == (5, 3)

    def test_dropout_is_identity_at_eval_and_scaled_in_train(self):
        params = init_mlp_params(random.PRNGKey(0), [4, 64, 2])
        x = jnp.ones((3, 4))
        key = random.PRNGKey(1)
        assert jnp.allclose(mlp_with_dropout(params, x, key, 0.5, training=False), mlp_forward(params, x))
        a = mlp_with_dropout(params, x, key, 0.5, training=True)
        b = mlp_with_dropout(params, x, random.PRNGKey(2), 0.5, training=True)
        assert not jnp.allclose(a, b)

    def test_batch_norm_running_stats_update(self):
        params, state = init_mlp_with_batch_norm(random.PRNGKey(0), [4, 8, 2])
        x = random.normal(random.PRNGKey(1), (32, 4)) * 5 + 3
        out, new_state = mlp_with_batch_norm(params, state, x, training=True)
        assert out.shape == (32, 2)
        assert not jnp.allclose(new_state["bn"][0]["mean"], state["bn"][0]["mean"])
        out_eval, same_state = mlp_with_batch_norm(params, new_state, x, training=False)
        assert jnp.allclose(same_state["bn"][0]["mean"], new_state["bn"][0]["mean"])


# ============================================================
# CNN
# ============================================================


class TestCNN:
    def test_init_conv_params(self):
        p = init_conv_params(random.PRNGKey(0), (3, 3), in_channels=1, out_channels=8)
        assert p["weights"].shape == (8, 1, 3, 3) and p["biases"].shape == (8,)

    def test_conv2d_output_shape(self):
        params = init_conv_params(random.PRNGKey(0), (3, 3), in_channels=1, out_channels=4)
        x = jnp.ones((2, 8, 8, 1))
        assert conv2d_layer(x, params, padding="SAME").shape == (2, 8, 8, 4)
        assert conv2d_layer(x, params, padding="VALID").shape == (2, 6, 6, 4)
        assert conv2d_layer(x, params, stride=(2, 2)).shape == (2, 4, 4, 4)
        assert conv_output_shape((8, 8), (3, 3), (2, 2), "SAME") == (4, 4)
        assert conv_output_shape((8, 8), (3, 3), (1, 1), "VALID") == (6, 6)

    def test_conv_matches_manual_cross_correlation(self):
        x = random.normal(random.PRNGKey(0), (1, 5, 5, 1))
        w = random.normal(random.PRNGKey(1), (1, 1, 3, 3))
        out = conv2d_layer(x, {"weights": w, "biases": jnp.zeros(1)}, padding="VALID", activation="linear")
        manual = sum(x[0, i:i + 3, j:j + 3, 0].ravel() @ w[0, 0].ravel() for i in [0] for j in [0])
        assert jnp.allclose(out[0, 0, 0, 0], manual, atol=1e-5)

    def test_pooling(self):
        x = jnp.arange(16.0).reshape(1, 4, 4, 1)
        mx = pooling_layer(x, "max", (2, 2), (2, 2))
        av = pooling_layer(x, "avg", (2, 2), (2, 2))
        assert mx.shape == (1, 2, 2, 1) and mx[0, 0, 0, 0] == 5.0
        assert av[0, 0, 0, 0] == 2.5

    def test_depthwise_conv_keeps_channels_separate(self):
        x = random.normal(random.PRNGKey(0), (1, 6, 6, 3))
        kernel = jnp.zeros((3, 3, 3, 1)).at[1, 1, :, 0].set(1.0)  # identity per channel
        assert jnp.allclose(depthwise_conv2d(x, kernel), x, atol=1e-6)

    def test_batch_norm_2d(self):
        params, state = init_batch_norm_2d(3)
        x = random.normal(random.PRNGKey(0), (4, 5, 5, 3)) * 2 + 1
        out, new_state = batch_norm_2d(x, params, state, training=True)
        assert jnp.allclose(jnp.mean(out, axis=(0, 1, 2)), 0.0, atol=1e-5)
        assert jnp.allclose(jnp.var(out, axis=(0, 1, 2)), 1.0, atol=1e-3)
        assert jnp.all(new_state["running_mean"] != 0)

    def test_residual_block_shape(self):
        p1 = init_conv_params(random.PRNGKey(0), (3, 3), 4, 4)
        p2 = init_conv_params(random.PRNGKey(1), (3, 3), 4, 4)
        x = random.normal(random.PRNGKey(2), (2, 8, 8, 4))
        assert residual_block(x, p1, p2).shape == x.shape

    def test_create_cnn_infers_flatten_size(self):
        params, fwd = create_cnn((16, 16, 3), conv_channels=[4, 8], dense_layers=[16], num_classes=5)
        # two 2x2 pools: 16 -> 8 -> 4, times 8 channels
        assert params["dense_layers"][0]["weights"].shape == (4 * 4 * 8, 16)
        logits = jax.jit(fwd)(params, jnp.ones((2, 16, 16, 3)))
        assert logits.shape == (2, 5)


# ============================================================
# Transformer components
# ============================================================


class TestTransformer:
    def test_init_attention_params(self):
        p = init_attention_params(random.PRNGKey(0), d_model=16, num_heads=2)
        assert set(p) == {"query", "key", "value", "out"}
        with pytest.raises(ValueError):
            init_attention_params(random.PRNGKey(0), d_model=15, num_heads=2)

    def test_scaled_dot_product_attention_shape_and_rows_sum_to_one(self):
        q = random.normal(random.PRNGKey(0), (2, 4, 8))
        out, weights = scaled_dot_product_attention(q, q, q)
        assert out.shape == (2, 4, 8)
        assert jnp.allclose(weights.sum(axis=-1), 1.0, atol=1e-5)

    def test_causal_mask_blocks_future_and_fully_masked_rows_are_finite(self):
        q = random.normal(random.PRNGKey(0), (1, 4, 8))
        _, w = scaled_dot_product_attention(q, q, q, mask=create_causal_mask(4))
        assert jnp.allclose(jnp.triu(w[0], 1), 0.0)
        _, w_all_masked = scaled_dot_product_attention(q, q, q, mask=jnp.zeros((4, 4), bool))
        assert jnp.all(jnp.isfinite(w_all_masked))

    def test_padding_mask_shape(self):
        tokens = jnp.array([[5, 3, 0, 0]])
        m = create_padding_mask(tokens)
        assert m.shape == (1, 1, 1, 4) and bool(m[0, 0, 0, 1]) and not bool(m[0, 0, 0, 2])

    def test_multi_head_attention_shapes_and_cross_attention(self):
        p = init_attention_params(random.PRNGKey(0), 16, 4)
        x = random.normal(random.PRNGKey(1), (2, 5, 16))
        ctx = random.normal(random.PRNGKey(2), (2, 7, 16))
        out, w = multi_head_attention(x, p, 4)
        assert out.shape == (2, 5, 16) and w.shape == (2, 4, 5, 5)
        out_c, w_c = multi_head_attention(x, p, 4, context=ctx)
        assert out_c.shape == (2, 5, 16) and w_c.shape == (2, 4, 5, 7)

    def test_layer_norm_and_rms_norm(self):
        x = random.normal(random.PRNGKey(0), (4, 8)) * 3 + 2
        out = layer_norm(x, jnp.ones(8), jnp.zeros(8))
        assert jnp.allclose(out.mean(axis=-1), 0.0, atol=1e-5)
        assert jnp.allclose(out.std(axis=-1), 1.0, atol=1e-2)
        r = rms_norm(x, jnp.ones(8))
        assert jnp.allclose(jnp.sqrt(jnp.mean(r**2, axis=-1)), 1.0, atol=1e-3)

    def test_feed_forward_shape(self):
        x = random.normal(random.PRNGKey(0), (4, 16))
        k1, k2 = random.split(random.PRNGKey(0))
        ff = {"W1": random.normal(k1, (16, 32)), "b1": jnp.zeros(32),
              "W2": random.normal(k2, (32, 16)), "b2": jnp.zeros(16)}
        assert feed_forward_network(x, ff).shape == (4, 16)

    def test_positional_encoding_properties(self):
        pe = positional_encoding(10, 8)
        assert pe.shape == (10, 8)
        assert jnp.allclose(pe[0, 0::2], 0.0) and jnp.allclose(pe[0, 1::2], 1.0)

    def test_rotary_embedding_is_relative(self):
        d = 8
        q = random.normal(random.PRNGKey(0), (1, 1, 6, d))
        k = random.normal(random.PRNGKey(1), (1, 1, 6, d))
        # score between positions (i, j) must depend only on i - j: compare
        # (2, 0) with (5, 3) after shifting both sequences by 3.
        qr, kr = rotary_embedding(q), rotary_embedding(k)
        s_a = jnp.sum(qr[0, 0, 2] * kr[0, 0, 0])
        pos = jnp.arange(6) + 3
        qs, ks = rotary_embedding(q, pos), rotary_embedding(k, pos)
        s_b = jnp.sum(qs[0, 0, 2] * ks[0, 0, 0])
        assert jnp.allclose(s_a, s_b, atol=1e-4)
        assert jnp.allclose(jnp.linalg.norm(qr, axis=-1), jnp.linalg.norm(q, axis=-1), atol=1e-5)

    def test_transformer_block_shape_pre_and_post_norm(self):
        p = init_transformer_block_params(random.PRNGKey(0), 16, 4)
        x = random.normal(random.PRNGKey(1), (2, 5, 16))
        assert transformer_block(x, p, 4, pre_norm=True).shape == x.shape
        assert transformer_block(x, p, 4, pre_norm=False).shape == x.shape

    @pytest.mark.parametrize("remat", [False, True])
    def test_create_transformer_scan_matches_loop(self, remat):
        params, fwd = create_transformer(16, 4, num_layers=3, vocab_size=20, max_seq_len=8, remat=remat)
        assert params["layers"]["attention"]["query"].shape == (3, 16, 16)
        tokens = random.randint(random.PRNGKey(0), (2, 6), 0, 20)
        out = jax.jit(fwd)(params, tokens, mask=create_causal_mask(6))
        assert out.shape == (2, 6, 16)

        # Reference: python loop over the per-layer params.
        x = params["embedding"][tokens] + params["pos_encoding"][:6]
        for i in range(3):
            layer_i = jax.tree_util.tree_map(lambda a: a[i], params["layers"])
            x = transformer_block(x, layer_i, 4, mask=create_causal_mask(6))
        x = layer_norm(x, **params["final_ln"])
        assert jnp.allclose(out, x, atol=1e-4)

    def test_transformer_is_differentiable_with_dropout_keys(self):
        params, fwd = create_transformer(16, 2, num_layers=2, vocab_size=10, max_seq_len=8)
        tokens = random.randint(random.PRNGKey(0), (2, 4), 0, 10)

        def loss(p):
            return jnp.mean(fwd(p, tokens, key_rng=random.PRNGKey(3), dropout_rate=0.1) ** 2)

        g = grad(loss)(params)
        assert jnp.all(jnp.isfinite(g["layers"]["ffn"]["W1"]))
