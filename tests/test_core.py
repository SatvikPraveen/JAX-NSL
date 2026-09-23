# tests/test_core.py
"""Tests for jax_nsl.core: arrays, prng, and numerics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import grad, random

from jax_nsl.core.arrays import (
    check_finite,
    get_dtype_info,
    pad_to_shape,
    safe_cast,
    sliding_window,
    tree_bytes,
    tree_size,
    tree_summary,
)
from jax_nsl.core.numerics import (
    clip_gradients,
    default_fd_step,
    log_softmax_stable,
    logsumexp_stable,
    numerical_gradient,
    safe_divide,
    safe_exp,
    safe_log,
    safe_norm,
    safe_sqrt,
    softmax_stable,
    stable_logsumexp,
    stable_sigmoid,
    stable_softmax,
    stable_tanh,
)
from jax_nsl.core.prng import (
    PRNGSequence,
    compute_fans,
    glorot_normal_init,
    glorot_uniform_init,
    he_normal_init,
    he_uniform_init,
    make_rng_state,
    orthogonal_init,
    random_like,
    split_key_tree,
)

# ============================================================
# core.arrays
# ============================================================


class TestArrayUtils:
    def test_get_dtype_info_float(self):
        info = get_dtype_info(jnp.float32)
        assert info["name"] == "float32"
        assert "eps" in info
        assert info["itemsize"] == 4

    def test_get_dtype_info_accepts_strings_and_arrays(self):
        assert get_dtype_info("bfloat16")["bits"] == 16
        assert get_dtype_info(jnp.ones(2).dtype)["kind"] == "f"

    def test_get_dtype_info_int(self):
        info = get_dtype_info(jnp.int32)
        assert info["max"] == 2**31 - 1
        assert info["min"] == -(2**31)

    def test_safe_cast_int32_overflow(self):
        large = jnp.array([1e12, -1e12], dtype=jnp.float32)
        result = safe_cast(large, jnp.int32, clip=True)
        assert int(result[0]) == jnp.iinfo(jnp.int32).max
        assert int(result[1]) == jnp.iinfo(jnp.int32).min

    def test_safe_cast_noop(self):
        x = jnp.ones((3,), dtype=jnp.float32)
        assert safe_cast(x, jnp.float32) is x

    def test_tree_size(self):
        tree = {"a": jnp.ones((3, 4)), "b": jnp.zeros(5)}
        assert tree_size(tree) == 17

    def test_tree_bytes(self):
        tree = {"w": jnp.ones((4,), dtype=jnp.float32)}
        assert tree_bytes(tree) == 16  # 4 floats x 4 bytes

    def test_tree_summary(self):
        tree = {"w": jnp.ones((2, 3))}
        summary = tree_summary(tree)
        assert summary["total_elements"] == 6
        assert summary["num_arrays"] == 1
        assert len(summary["devices"]) >= 1

    def test_tree_summary_empty(self):
        assert tree_summary({})["empty"] is True

    def test_check_finite(self):
        assert check_finite(jnp.array([1.0, 2.0, 3.0]))
        assert not check_finite(jnp.array([1.0, float("nan"), 3.0]))
        assert not check_finite(jnp.array([1.0, float("inf"), 3.0]))

    def test_sliding_window(self):
        x = jnp.arange(6.0)
        w = sliding_window(x, window_size=3, stride=1)
        assert w.shape == (4, 3)
        assert jnp.array_equal(w[1], jnp.array([1.0, 2.0, 3.0]))

    def test_pad_to_shape(self):
        x = jnp.ones((2, 3))
        y = pad_to_shape(x, (4, 5))
        assert y.shape == (4, 5)
        assert float(y.sum()) == 6.0


# ============================================================
# core.prng
# ============================================================


class TestPRNG:
    def test_sequence_keys_distinct(self):
        seq = PRNGSequence(random.PRNGKey(0))
        keys = [next(seq) for _ in range(5)]
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                if i != j:
                    assert not jnp.array_equal(ki, kj)

    def test_sequence_from_int_and_typed_key(self):
        seq_int = PRNGSequence(0)
        seq_typed = PRNGSequence(random.key(0))
        assert next(seq_int).shape == (2,)
        assert jax.dtypes.issubdtype(next(seq_typed).dtype, jax.dtypes.prng_key)

    def test_split_and_fork(self):
        seq = PRNGSequence(random.PRNGKey(0))
        k1, k2 = seq.split(2)
        assert k1.shape == (2,)
        assert not jnp.array_equal(k1, k2)
        children = seq.fork(3)
        assert len(children) == 3
        assert not jnp.array_equal(next(children[0]), next(children[1]))

    def test_split_key_tree_structure(self):
        tree = {"a": 0, "b": [1, 2]}
        keys = split_key_tree(random.PRNGKey(0), tree)
        assert set(keys) == {"a", "b"}
        assert len(keys["b"]) == 2

    def test_make_rng_state(self):
        state = make_rng_state(0, ["params", "dropout"])
        assert set(state) == {"params", "dropout"}
        assert not jnp.array_equal(state["params"], state["dropout"])

    def test_random_like_normal(self):
        out = random_like(random.PRNGKey(1), jnp.zeros((4, 4)), distribution="normal")
        assert out.shape == (4, 4)
        assert jnp.all(jnp.isfinite(out))

    def test_random_like_uniform(self):
        out = random_like(random.PRNGKey(2), jnp.zeros((10,)), distribution="uniform")
        assert jnp.all(out >= 0.0) and jnp.all(out <= 1.0)


class TestInitializers:
    @pytest.mark.parametrize(
        "init", [glorot_uniform_init, glorot_normal_init, he_uniform_init, he_normal_init]
    )
    def test_shape_and_dtype(self, init):
        w = init(random.PRNGKey(0), (4, 8), dtype=jnp.bfloat16)
        assert w.shape == (4, 8)
        assert w.dtype == jnp.bfloat16

    def test_he_normal_scale(self):
        w = he_normal_init(random.PRNGKey(0), (1000, 4))
        # std should be sqrt(2 / fan_in) = sqrt(2/1000) ~ 0.0447
        assert abs(float(jnp.std(w)) - 0.0447) < 0.01

    def test_compute_fans_dense_and_conv(self):
        assert compute_fans((4, 8)) == (4, 8)
        # conv kernel (O, I, kh, kw): fan_in = I * kh * kw
        assert compute_fans((16, 3, 3, 3), in_axis=1, out_axis=0) == (27, 144)

    def test_orthogonal_init_is_orthonormal(self):
        q = orthogonal_init(random.PRNGKey(0), (6, 4))
        assert jnp.allclose(q.T @ q, jnp.eye(4), atol=1e-5)
        q_wide = orthogonal_init(random.PRNGKey(1), (4, 6))
        assert jnp.allclose(q_wide @ q_wide.T, jnp.eye(4), atol=1e-5)


# ============================================================
# core.numerics
# ============================================================


class TestNumerics:
    def test_safe_log(self):
        x = jnp.array([1.0, 2.0])
        assert jnp.allclose(safe_log(x), jnp.log(x))
        assert jnp.isfinite(safe_log(jnp.array([0.0])))

    def test_safe_exp_large(self):
        assert jnp.isfinite(safe_exp(jnp.array([1000.0])))

    def test_safe_sqrt_negative(self):
        result = safe_sqrt(jnp.array([-1.0, 0.0, 4.0]))
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result[-1], 2.0)

    def test_safe_divide(self):
        assert jnp.isfinite(safe_divide(jnp.array([1.0]), jnp.array([0.0])))
        assert jnp.allclose(safe_divide(jnp.array([6.0]), jnp.array([2.0])), 3.0, atol=1e-6)

    def test_logsumexp_aliases_and_large_values(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(logsumexp_stable(x), stable_logsumexp(x))
        assert jnp.allclose(logsumexp_stable(x), jnp.log(jnp.sum(jnp.exp(x))))
        assert jnp.isfinite(logsumexp_stable(jnp.array([1000.0, 1001.0])))

    def test_logsumexp_all_neg_inf(self):
        x = jnp.array([-jnp.inf, -jnp.inf])
        assert logsumexp_stable(x) == -jnp.inf

    def test_logsumexp_matches_jax_with_gradient(self):
        x = random.normal(random.PRNGKey(0), (5, 7))
        g_mine = grad(lambda t: logsumexp_stable(t, axis=-1).sum())(x)
        g_jax = grad(lambda t: jax.nn.logsumexp(t, axis=-1).sum())(x)
        assert jnp.allclose(g_mine, g_jax, atol=1e-6)

    def test_softmax_sums_to_one(self):
        x = jnp.array([0.5, -1.0, 2.0])
        assert jnp.allclose(jnp.sum(softmax_stable(x)), 1.0)
        assert jnp.allclose(softmax_stable(x), stable_softmax(x))

    def test_log_softmax_precision_at_large_logits(self):
        # x - logsumexp(x) loses ~4 digits here; the shifted form must not.
        logits = jnp.array([[1000.0, 999.0, 1001.0], [500.0, 501.0, 499.0]])
        mine = log_softmax_stable(logits)
        ref = jax.nn.log_softmax(logits)
        assert jnp.allclose(mine, ref, atol=1e-6)
        assert jnp.allclose(jnp.exp(mine).sum(-1), 1.0)

    def test_stable_sigmoid_values_and_gradients_at_extremes(self):
        x = jnp.array([-1000.0, -20.0, 0.0, 20.0, 1000.0])
        out = stable_sigmoid(x)
        assert jnp.allclose(out, jax.nn.sigmoid(x), atol=1e-7)
        g = grad(lambda t: stable_sigmoid(t).sum())(x)
        assert jnp.all(jnp.isfinite(g))  # naive where-form gives NaN here
        assert jnp.allclose(stable_tanh(x), jnp.tanh(x), atol=1e-6)

    def test_default_fd_step_scales_with_dtype(self):
        assert default_fd_step(jnp.ones(3, jnp.float32)) > 1e-3
        h16 = default_fd_step(jnp.ones(3, jnp.float16))
        h32 = default_fd_step(jnp.ones(3, jnp.float32))
        assert h16 > h32

    def test_numerical_gradient_accuracy(self):
        def f(x):
            return jnp.sum(x**3)

        x = jnp.array([1.0, 2.0, 3.0])
        analytical = grad(f)(x)
        numerical = numerical_gradient(f, x)
        # float32 central differences at the dtype-optimal step: ~1e-3 relative.
        assert jnp.allclose(analytical, numerical, rtol=2e-3)

    def test_safe_norm_orders(self):
        tree = {"a": jnp.array([3.0, 0.0]), "b": jnp.array([[0.0, 4.0]])}
        assert jnp.allclose(safe_norm(tree), 5.0)
        assert jnp.allclose(safe_norm(tree, ord=1), 7.0)
        assert jnp.allclose(safe_norm(tree, ord=jnp.inf), 4.0)
        assert float(safe_norm({})) == 0.0

    def test_clip_gradients_norm_preserves_direction(self):
        grads = {"w": jnp.array([[10.0, -5.0], [3.0, -8.0]]), "b": jnp.array([2.0, -1.0])}
        clipped = clip_gradients(grads, max_norm=1.0)
        assert float(safe_norm(clipped)) <= 1.0 + 1e-6
        ratio = clipped["w"] / grads["w"]
        assert jnp.allclose(ratio, ratio[0, 0])

    def test_clip_gradients_value(self):
        grads = {"w": jnp.array([10.0, -10.0, 0.5])}
        clipped = clip_gradients(grads, max_value=1.0)
        assert jnp.array_equal(clipped["w"], jnp.array([1.0, -1.0, 0.5]))

    def test_numpy_interop(self):
        x = np.array([1.0, 2.0], dtype=np.float32)
        assert jnp.allclose(safe_log(x), np.log(x))
