# tests/test_linalg.py
"""Tests for jax_nsl.linalg: matrix ops and iterative solvers."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from jax_nsl.linalg.ops import (
    batched_matmul,
    cholesky_safe,
    condition_number,
    frobenius_norm,
    gram_schmidt,
    matrix_logarithm,
    matrix_power,
    matrix_sqrt,
    pseudoinverse_stable,
    qr_decomposition,
    safe_matmul,
    spectral_norm,
    stable_eigh,
    stable_svd,
    trace_product,
)
from jax_nsl.linalg.solvers import (
    conjugate_gradient,
    eigenvalue_power_method,
    gradient_descent,
    jacobi_method,
    lanczos_algorithm,
    lbfgs_solver,
    least_squares_solver,
    nesterov_momentum,
)


def _spd(key, n, jitter=1.0):
    a = random.normal(key, (n, n))
    return a @ a.T + jitter * jnp.eye(n)


# ============================================================
# linalg.ops
# ============================================================


class TestMatrixOps:
    def test_safe_matmul_shape(self):
        key = random.PRNGKey(0)
        assert safe_matmul(random.normal(key, (3, 4)), random.normal(key, (4, 5))).shape == (3, 5)

    def test_safe_matmul_rejects_bad_shapes(self):
        with pytest.raises(ValueError):
            safe_matmul(jnp.ones((3, 4)), jnp.ones((5, 3)))

    def test_batched_matmul(self):
        a = jnp.eye(3)[None].repeat(4, axis=0)
        b = jnp.arange(12, dtype=float).reshape(4, 3, 1)
        assert jnp.allclose(batched_matmul(a, b), b)

    def test_stable_svd_reconstruction(self):
        m = random.normal(random.PRNGKey(1), (5, 4))
        u, s, vt = stable_svd(m, full_matrices=False)
        assert jnp.allclose(u @ jnp.diag(s) @ vt, m, atol=1e-5)
        assert jnp.all(s >= 0)

    def test_stable_eigh_ascending_and_reconstructs(self):
        sym = _spd(random.PRNGKey(3), 4)
        w, v = stable_eigh(sym)
        assert jnp.all(jnp.diff(w) >= -1e-6)
        assert jnp.allclose((v * w) @ v.T, sym, atol=1e-4)

    def test_stable_eigh_symmetrizes(self):
        a = random.normal(random.PRNGKey(4), (4, 4))
        w_lower, _ = stable_eigh(a, UPLO="L")
        w_upper, _ = stable_eigh(a, UPLO="U")
        assert jnp.allclose(w_lower, w_upper, atol=1e-5)

    def test_qr_orthogonality(self):
        q, r = qr_decomposition(random.normal(random.PRNGKey(4), (5, 4)), mode="reduced")
        assert jnp.allclose(q.T @ q, jnp.eye(4), atol=1e-5)
        assert jnp.allclose(jnp.tril(r, -1), 0.0, atol=1e-6)

    def test_cholesky_safe_spd(self):
        spd = _spd(random.PRNGKey(5), 4, jitter=2.0)
        lower = cholesky_safe(spd)
        assert jnp.allclose(lower @ lower.T, spd, atol=1e-4)

    def test_cholesky_safe_regularizes_psd(self):
        psd = jnp.array([[1.0, 1.0], [1.0, 1.0]])  # rank 1
        assert jnp.any(jnp.isnan(jnp.linalg.cholesky(psd)))
        assert jnp.all(jnp.isfinite(cholesky_safe(psd, regularization=1e-4)))

    def test_matrix_power(self):
        a = jnp.array([[1.0, 1.0], [0.0, 1.0]])
        assert jnp.allclose(matrix_power(a, 5), jnp.array([[1.0, 5.0], [0.0, 1.0]]))
        assert jnp.allclose(matrix_power(a, 0), jnp.eye(2))
        assert jnp.allclose(matrix_power(a, -1) @ a, jnp.eye(2), atol=1e-6)

    def test_matrix_sqrt_and_log(self):
        spd = _spd(random.PRNGKey(6), 3)
        root = matrix_sqrt(spd)
        assert jnp.allclose(root @ root, spd, atol=1e-3)
        logm = matrix_logarithm(spd)
        w, v = stable_eigh(spd)
        assert jnp.allclose((v * jnp.exp(jnp.linalg.eigvalsh(logm))) @ v.T, spd, atol=1e-3)

    def test_pseudoinverse_rank_deficient(self):
        a = jnp.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])  # rank 1
        pinv = pseudoinverse_stable(a)
        assert jnp.allclose(pinv, jnp.linalg.pinv(a), atol=1e-5)

    def test_gram_schmidt_orthonormal(self):
        v = random.normal(random.PRNGKey(7), (6, 4))
        q = gram_schmidt(v)
        assert jnp.allclose(q.T @ q, jnp.eye(4), atol=1e-5)
        assert jnp.allclose(jax.jit(gram_schmidt)(v), q, atol=1e-6)

    def test_trace_product(self):
        a = random.normal(random.PRNGKey(8), (4, 3))
        b = random.normal(random.PRNGKey(9), (3, 4))
        assert jnp.allclose(trace_product(a, b), jnp.trace(a @ b), atol=1e-5)

    def test_frobenius_norm(self):
        assert jnp.allclose(frobenius_norm(jnp.array([[3.0, 0.0], [4.0, 0.0]])), 5.0)

    def test_spectral_norm_matches_svd(self):
        a = random.normal(random.PRNGKey(10), (6, 4))
        assert jnp.allclose(spectral_norm(a, max_iterations=100), jnp.linalg.norm(a, 2), rtol=1e-3)
        assert jnp.allclose(spectral_norm(jnp.eye(4)), 1.0, atol=1e-3)

    def test_condition_number_identity(self):
        assert jnp.allclose(condition_number(jnp.eye(4)), 1.0, atol=1e-3)


# ============================================================
# linalg.solvers
# ============================================================


class TestLinearSolvers:
    def test_conjugate_gradient_solves_system(self):
        a = jnp.diag(jnp.array([1.0, 2.0, 3.0, 4.0]))
        b = jnp.array([1.0, 2.0, 3.0, 4.0])
        x, info = conjugate_gradient(a, b, tolerance=1e-6, max_iterations=50)
        assert jnp.allclose(a @ x, b, atol=1e-4)
        assert bool(info.converged)

    def test_conjugate_gradient_matrix_free_and_jit(self):
        a = _spd(random.PRNGKey(0), 6)
        b = random.normal(random.PRNGKey(1), (6,))
        solve = jax.jit(lambda b: conjugate_gradient(lambda v: a @ v, b, tolerance=1e-6,
                                                     max_iterations=60)[0])
        assert jnp.allclose(a @ solve(b), b, atol=1e-3)

    def test_conjugate_gradient_implicit_gradients(self):
        a = _spd(random.PRNGKey(0), 4)
        b = jnp.ones(4)

        def solution_sum(a, b):
            return conjugate_gradient(a, b, tolerance=1e-7, max_iterations=40)[0].sum()

        g_a, g_b = jax.grad(solution_sum, argnums=(0, 1))(a, b)
        lam = jnp.linalg.solve(a.T, jnp.ones(4))  # adjoint solve
        x = jnp.linalg.solve(a, b)
        assert jnp.allclose(g_b, lam, atol=1e-3)
        assert jnp.allclose(g_a, -jnp.outer(lam, x), atol=1e-3)

    def test_conjugate_gradient_matrix_free_gradient_through_closure(self):
        a = _spd(random.PRNGKey(0), 4)
        b = jnp.ones(4)

        def solution_sum(scale):
            matvec = lambda v: (scale * a) @ v  # noqa: E731
            return conjugate_gradient(matvec, b, tolerance=1e-7, max_iterations=40)[0].sum()

        # x = (s A)^{-1} b  =>  d sum(x)/ds = -sum(A^{-1} b) / s^2
        g = jax.grad(solution_sum)(2.0)
        expected = -jnp.sum(jnp.linalg.solve(a, b)) / 4.0
        assert jnp.allclose(g, expected, rtol=1e-3)

    def test_jacobi_method_diag_dominant(self):
        a = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        b = jnp.array([1.0, 2.0])
        x, info = jacobi_method(a, b, tolerance=1e-6, max_iterations=500)
        assert jnp.allclose(a @ x, b, atol=1e-4)
        assert bool(info.converged)

    def test_least_squares_matches_lstsq(self):
        a = random.normal(random.PRNGKey(2), (8, 3))
        b = random.normal(random.PRNGKey(3), (8,))
        x = least_squares_solver(a, b)
        assert jnp.allclose(x, jnp.linalg.lstsq(a, b)[0], atol=1e-4)

    def test_least_squares_regularized_shrinks(self):
        a = random.normal(random.PRNGKey(2), (8, 3))
        b = random.normal(random.PRNGKey(3), (8,))
        x_plain = least_squares_solver(a, b)
        x_reg = least_squares_solver(a, b, regularization=10.0)
        assert jnp.linalg.norm(x_reg) < jnp.linalg.norm(x_plain)


class TestOptimizers:
    def _quadratic(self):
        a = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        b = jnp.array([1.0, 2.0])
        return (lambda x: 0.5 * x @ a @ x - b @ x), jnp.linalg.solve(a, b)

    def test_gradient_descent(self):
        f, x_star = self._quadratic()
        x, info = gradient_descent(f, jnp.zeros(2), learning_rate=0.1, max_iterations=500)
        assert jnp.allclose(x, x_star, atol=1e-3)
        assert bool(info.converged)

    def test_nesterov_momentum(self):
        f, x_star = self._quadratic()
        x, _ = nesterov_momentum(f, jnp.zeros(2), learning_rate=0.05, max_iterations=500)
        assert jnp.allclose(x, x_star, atol=1e-3)

    def test_lbfgs_rosenbrock_and_jit(self):
        def rosenbrock(x):
            return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)

        x0 = jnp.array([-1.2, 1.0, -1.0, 0.5])
        x, info = jax.jit(lambda x0: lbfgs_solver(rosenbrock, x0, tolerance=1e-4,
                                                  max_iterations=500))(x0)
        assert jnp.allclose(x, jnp.ones(4), atol=1e-2)
        assert int(info.iteration) < 500

    def test_lbfgs_beats_gradient_descent_in_iterations(self):
        f, x_star = self._quadratic()
        _, gd_info = gradient_descent(f, jnp.zeros(2), learning_rate=0.1, tolerance=1e-5,
                                      max_iterations=1000)
        _, lb_info = lbfgs_solver(f, jnp.zeros(2), tolerance=1e-5, max_iterations=100)
        assert int(lb_info.iteration) < int(gd_info.iteration)


class TestEigen:
    def test_power_method(self):
        a = jnp.diag(jnp.array([5.0, 3.0, 1.0]))
        lam, v = eigenvalue_power_method(a, max_iterations=200, tolerance=1e-8)
        assert jnp.allclose(lam, 5.0, atol=1e-3)
        assert jnp.allclose(jnp.abs(v), jnp.array([1.0, 0.0, 0.0]), atol=1e-2)

    def test_lanczos_ritz_values_match_extremes(self):
        a = _spd(random.PRNGKey(11), 30)
        t, q = lanczos_algorithm(a, num_iterations=30, starting_vector=jnp.ones(30))
        ritz = jnp.linalg.eigvalsh(t)
        exact = jnp.linalg.eigvalsh(a)
        assert jnp.allclose(ritz[-1], exact[-1], rtol=1e-3)
        assert jnp.allclose(ritz[0], exact[0], rtol=1e-2, atol=1e-2)
        assert jnp.allclose(q.T @ q, jnp.eye(30), atol=1e-3)
