# tests/test_core.py
"""Tests for src/core: arrays, prng, and numerics."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from core.arrays import (
    get_dtype_info, safe_cast, tree_size, tree_bytes, tree_summary, check_finite
)
from core.prng import (
    PRNGSequence, glorot_uniform_init, glorot_normal_init,
    he_uniform_init, he_normal_init, random_like,
)
from core.numerics import (
    safe_log, safe_exp, logsumexp_stable, stable_logsumexp,
    softmax_stable, stable_softmax, log_softmax_stable,
    clip_gradients, safe_divide, stable_sigmoid, safe_sqrt,
    numerical_gradient,
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

    def test_get_dtype_info_int(self):
        info = get_dtype_info(jnp.int32)
        assert "max" in info
        assert "min" in info

    def test_safe_cast_int32_overflow(self):
        large = jnp.array([1e12, -1e12], dtype=jnp.float32)
        result = safe_cast(large, jnp.int32, clip=True)
        assert jnp.all(result <= jnp.iinfo(jnp.int32).max)
        assert jnp.all(result >= jnp.iinfo(jnp.int32).min)

    def test_safe_cast_noop(self):
        x = jnp.ones((3,), dtype=jnp.float32)
        assert safe_cast(x, jnp.float32) is x

    def test_tree_size(self):
        tree = {"a": jnp.ones((3, 4)), "b": jnp.zeros(5)}
        assert tree_size(tree) == 17

    def test_tree_bytes(self):
        tree = {"w": jnp.ones((4,), dtype=jnp.float32)}
        assert tree_bytes(tree) == 16  # 4 floats × 4 bytes

    def test_tree_summary(self):
        tree = {"w": jnp.ones((2, 3))}
        summary = tree_summary(tree)
        assert summary["total_elements"] == 6
        assert summary["num_arrays"] == 1

    def test_check_finite_valid(self):
        assert check_finite(jnp.array([1.0, 2.0, 3.0]))

    def test_check_finite_nan(self):
        assert not check_finite(jnp.array([1.0, float("nan"), 3.0]))

    def test_check_finite_inf(self):
        assert not check_finite(jnp.array([1.0, float("inf"), 3.0]))


# ============================================================
# core.prng
# ============================================================

class TestPRNGSequence:
    def test_iteration(self):
        seq = PRNGSequence(random.PRNGKey(0))
        keys = [next(seq) for _ in range(5)]
        # All keys should be distinct
        for i, ki in enumerate(keys):
            for j, kj in enumerate(keys):
                if i != j:
                    assert not jnp.array_equal(ki, kj)

    def test_split(self):
        seq = PRNGSequence(random.PRNGKey(0))
        k1, k2 = seq.split(2)
        assert k1.shape == (2,)

    def test_random_like_normal(self):
        key = random.PRNGKey(1)
        template = jnp.zeros((4, 4))
        out = random_like(key, template, distribution="normal")
        assert out.shape == template.shape
        assert jnp.all(jnp.isfinite(out))

    def test_random_like_uniform(self):
        key = random.PRNGKey(2)
        template = jnp.zeros((10,))
        out = random_like(key, template, distribution="uniform")
        assert jnp.all(out >= 0.0) and jnp.all(out <= 1.0)


class TestInitializers:
    def test_glorot_uniform_shape(self):
        key = random.PRNGKey(0)
        w = glorot_uniform_init(key, (4, 8))
        assert w.shape == (4, 8)

    def test_glorot_normal_shape(self):
        key = random.PRNGKey(0)
        w = glorot_normal_init(key, (4, 8))
        assert w.shape == (4, 8)

    def test_he_uniform_shape(self):
        key = random.PRNGKey(0)
        w = he_uniform_init(key, (4, 8))
        assert w.shape == (4, 8)

    def test_he_normal_scale(self):
        key = random.PRNGKey(0)
        w = he_normal_init(key, (1000, 4))
        # He-normal std ≈ sqrt(2 / fan_in); fan_in=1000 → std ≈ 0.045
        assert float(jnp.std(w)) < 0.1


# ============================================================
# core.numerics
# ============================================================

class TestNumerics:
    def test_safe_log_positive(self):
        x = jnp.array([1.0, 2.0])
        result = safe_log(x)
        assert jnp.allclose(result, jnp.log(x))

    def test_safe_log_zero(self):
        result = safe_log(jnp.array([0.0]))
        assert jnp.isfinite(result)

    def test_safe_exp_large(self):
        result = safe_exp(jnp.array([1000.0]), max_val=88.0)
        assert jnp.isfinite(result)

    def test_logsumexp_aliases_agree(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(logsumexp_stable(x), stable_logsumexp(x))

    def test_logsumexp_large(self):
        x = jnp.array([1000.0, 1001.0])
        result = logsumexp_stable(x)
        assert jnp.isfinite(result)

    def test_softmax_aliases_agree(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(softmax_stable(x), stable_softmax(x))

    def test_softmax_sums_to_one(self):
        x = jnp.array([0.5, -1.0, 2.0])
        out = softmax_stable(x)
        assert jnp.allclose(jnp.sum(out), 1.0)

    def test_log_softmax_stable(self):
        x = jnp.array([0.5, -1.0, 2.0])
        out = log_softmax_stable(x)
        assert jnp.all(jnp.isfinite(out))
        assert jnp.allclose(jnp.exp(out).sum(), 1.0)

    def test_safe_divide_by_zero(self):
        result = safe_divide(jnp.array([1.0]), jnp.array([0.0]))
        assert jnp.isfinite(result)

    def test_safe_divide_normal(self):
        assert jnp.allclose(safe_divide(jnp.array([6.0]), jnp.array([2.0])),
                            jnp.array([3.0]))

    def test_stable_sigmoid_bounds(self):
        x = jnp.array([-1000.0, 0.0, 1000.0])
        out = stable_sigmoid(x)
        assert jnp.allclose(out[0], 0.0, atol=1e-6)
        assert jnp.allclose(out[-1], 1.0, atol=1e-6)

    def test_safe_sqrt_negative(self):
        result = safe_sqrt(jnp.array([-1.0, 0.0, 4.0]))
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result[-1], 2.0)

    def test_numerical_gradient_accuracy(self):
        from jax import grad as jax_grad
        def f(x):
            return jnp.sum(x ** 3)

        x = jnp.array([1.0, 2.0, 3.0])
        analytical = jax_grad(f)(x)
        numerical = numerical_gradient(f, x, h=1e-5)
        assert jnp.allclose(analytical, numerical, atol=1e-4)

    def test_clip_gradients_norm(self):
        import jax
        grads = {"w": jnp.array([[10.0, -5.0], [3.0, -8.0]]),
                 "b": jnp.array([2.0, -1.0])}
        clipped = clip_gradients(grads, max_norm=1.0)

        def tree_norm(t):
            leaves = jax.tree_util.tree_leaves(t)
            return float(jnp.sqrt(sum(jnp.sum(l ** 2) for l in leaves)))

        assert tree_norm(clipped) <= 1.0 + 1e-6
