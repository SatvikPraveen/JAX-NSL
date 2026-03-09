# tests/test_linalg.py
"""Tests for src/linalg: matrix ops and iterative solvers."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from linalg.ops import (
    safe_matmul, batched_matmul, stable_svd, stable_eigh,
    qr_decomposition, cholesky_safe, frobenius_norm,
    spectral_norm, condition_number,
)
from linalg.solvers import (
    conjugate_gradient, gradient_descent, eigenvalue_power_method,
)


# ============================================================
# linalg.ops
# ============================================================

class TestMatrixOps:
    def test_safe_matmul_shape(self):
        key = random.PRNGKey(0)
        a = random.normal(key, (3, 4))
        b = random.normal(key, (4, 5))
        result = safe_matmul(a, b)
        assert result.shape == (3, 5)

    def test_safe_matmul_rejects_bad_shapes(self):
        a = jnp.ones((3, 4))
        b = jnp.ones((5, 3))
        with pytest.raises(ValueError):
            safe_matmul(a, b)

    def test_batched_matmul_shape(self):
        key = random.PRNGKey(0)
        a = random.normal(key, (8, 3, 4))
        b = random.normal(key, (8, 4, 5))
        result = batched_matmul(a, b)
        assert result.shape == (8, 3, 5)

    def test_batched_matmul_correctness(self):
        a = jnp.eye(3)[None].repeat(4, axis=0)  # (4, 3, 3) identity
        b = jnp.arange(12, dtype=float).reshape(4, 3, 1)
        result = batched_matmul(a, b)
        assert jnp.allclose(result, b)

    def test_stable_svd_reconstruction(self):
        key = random.PRNGKey(1)
        m = random.normal(key, (5, 4))
        u, s, vt = stable_svd(m, full_matrices=False)
        reconstructed = u @ jnp.diag(s) @ vt
        assert jnp.allclose(reconstructed, m, atol=1e-5)

    def test_stable_svd_singular_values_positive(self):
        key = random.PRNGKey(2)
        m = random.normal(key, (6, 4))
        _, s, _ = stable_svd(m)
        assert jnp.all(s >= 0)

    def test_stable_eigh_eigenvalues_sorted(self):
        key = random.PRNGKey(3)
        a = random.normal(key, (4, 4))
        sym = a @ a.T + jnp.eye(4)
        eigenvalues, _ = stable_eigh(sym)
        assert jnp.all(jnp.diff(eigenvalues) >= -1e-6)

    def test_qr_orthogonality(self):
        key = random.PRNGKey(4)
        a = random.normal(key, (5, 4))
        q, r = qr_decomposition(a, mode="reduced")
        assert jnp.allclose(q.T @ q, jnp.eye(4), atol=1e-5)

    def test_cholesky_safe_spd(self):
        key = random.PRNGKey(5)
        a = random.normal(key, (4, 4))
        spd = a @ a.T + 2 * jnp.eye(4)
        l = cholesky_safe(spd)
        assert jnp.allclose(l @ l.T, spd, atol=1e-5)

    def test_frobenius_norm(self):
        a = jnp.array([[3.0, 0.0], [4.0, 0.0]])
        assert jnp.allclose(frobenius_norm(a), 5.0)

    def test_spectral_norm_identity(self):
        # Spectral norm of identity = 1.0
        norm = spectral_norm(jnp.eye(4))
        assert jnp.allclose(norm, 1.0, atol=1e-3)

    def test_condition_number_identity(self):
        cond = condition_number(jnp.eye(4))
        assert jnp.allclose(cond, 1.0, atol=1e-3)


# ============================================================
# linalg.solvers
# ============================================================

class TestSolvers:
    def test_conjugate_gradient_solves_system(self):
        # Ax = b where A = diag(1,2,3,4)
        A = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
        b = jnp.array([1.0, 2.0, 3.0, 4.0])
        x, info = conjugate_gradient(A, b, tol=1e-6, max_iter=50)
        assert jnp.allclose(A @ x, b, atol=1e-4)

    def test_gradient_descent_minimises_quadratic(self):
        def f(x):
            return jnp.sum((x - 3.0) ** 2)

        x0 = jnp.zeros(3)
        result, _ = gradient_descent(f, x0, learning_rate=0.1, max_iter=200)
        assert jnp.allclose(result, jnp.full(3, 3.0), atol=0.05)

    def test_eigenvalue_power_method(self):
        # Largest eigenvalue of diag(5, 3, 1) should be 5
        A = jnp.diag(jnp.array([5.0, 3.0, 1.0]))
        eigenvalue, _ = eigenvalue_power_method(A, max_iter=100)
        assert jnp.allclose(eigenvalue, 5.0, atol=0.1)
