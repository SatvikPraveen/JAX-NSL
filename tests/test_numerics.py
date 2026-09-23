# tests/test_numerics.py
"""Numerical-stability scenarios that cut across core, linalg and training.

Every tolerance here is chosen for the *default float32* configuration; the
comments explain which float32 limit each test is probing.
"""

import jax
import jax.numpy as jnp
import pytest
from jax import grad, random

from jax_nsl.core.arrays import check_finite, safe_cast, tree_size
from jax_nsl.core.numerics import (
    clip_gradients,
    log_softmax_stable,
    numerical_gradient,
    safe_divide,
    safe_log,
    safe_sqrt,
    stable_logsumexp,
    stable_softmax,
)
from jax_nsl.linalg.solvers import conjugate_gradient, gradient_descent


class TestStableOperations:
    def test_stable_logsumexp(self):
        x = jnp.array([1000.0, 999.0, 1001.0])
        result = stable_logsumexp(x)
        assert result > 1001.0
        assert jnp.isfinite(result)
        x_normal = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(stable_logsumexp(x_normal), jnp.log(jnp.sum(jnp.exp(x_normal))))

    def test_safe_log(self):
        result = safe_log(jnp.array([0.0, -1.0, 1e-10, 1.0]))
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result[-1], 0.0)

    def test_safe_sqrt(self):
        result = safe_sqrt(jnp.array([-1.0, 0.0, 1e-20, 4.0]))
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result[-1], 2.0)
        assert jnp.allclose(result[1], 0.0)

    def test_safe_divide(self):
        result = safe_divide(jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 0.0, 1e-15]))
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result[0], 0.5)

    def test_stable_softmax(self):
        x = jnp.array([1000.0, 999.0, 1001.0])
        result = stable_softmax(x)
        assert jnp.allclose(jnp.sum(result), 1.0)
        assert jnp.all(result > 0)
        assert jnp.all(jnp.isfinite(result))
        x_normal = jnp.array([1.0, 2.0, 3.0])
        naive = jnp.exp(x_normal) / jnp.sum(jnp.exp(x_normal))
        assert jnp.allclose(stable_softmax(x_normal), naive)


class TestGradientStability:
    def test_clip_gradients_global_norm(self):
        grads = {
            "layer1": {"w": jnp.array([[10.0, -5.0], [3.0, -8.0]]), "b": jnp.array([2.0, -1.0])},
            "layer2": {"w": jnp.array([[1.0, 2.0]]), "b": jnp.array([0.5])},
        }
        clipped = clip_gradients(grads, max_norm=1.0)
        leaves = jax.tree_util.tree_leaves(clipped)
        assert float(jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))) <= 1.0 + 1e-6

    def test_numerical_gradient(self):
        def f(x):
            return jnp.sum(x**3)

        x = jnp.array([1.0, 2.0, 3.0])
        # Default step is dtype-aware; float32 buys ~3 significant digits.
        assert jnp.allclose(grad(f)(x), numerical_gradient(f, x), rtol=2e-3)

    def test_numerical_gradient_too_small_step_is_worse(self):
        # Demonstrates *why* the step must scale with the dtype: h=1e-5 is
        # below float32 resolution at |x|=3 and the estimate degrades badly.
        def f(x):
            return jnp.sum(x**3)

        x = jnp.array([1.0, 2.0, 3.0])
        err_default = jnp.max(jnp.abs(numerical_gradient(f, x) - grad(f)(x)))
        err_tiny = jnp.max(jnp.abs(numerical_gradient(f, x, h=1e-5) - grad(f)(x)))
        assert err_default < err_tiny

    def test_gradient_numerical_stability(self):
        def f(x):
            return jnp.sum(jnp.log(jnp.exp(x) + 1e-8))

        assert jnp.all(jnp.isfinite(grad(f)(jnp.array([10.0, -10.0, 0.0]))))


class TestArrayStability:
    def test_safe_cast(self):
        casted = safe_cast(jnp.array([1e10, -1e10, 1e5]), jnp.int32)
        assert jnp.all(jnp.abs(casted) <= 2**31 - 1)
        casted_small = safe_cast(jnp.array([1e-10, 1e-50, 0.0]), jnp.float16)
        assert jnp.all(jnp.isfinite(casted_small))

    def test_check_finite(self):
        assert check_finite(jnp.array([1.0, 2.0, 3.0]))
        assert not check_finite(jnp.array([1.0, jnp.nan, 3.0]))
        assert not check_finite(jnp.array([1.0, jnp.inf, 3.0]))

    def test_tree_size(self):
        tree = {
            "layer1": {"w": jnp.ones((10, 5)), "b": jnp.ones(5)},
            "layer2": {"w": jnp.ones((5, 2)), "b": jnp.ones(2)},
        }
        assert tree_size(tree) == 10 * 5 + 5 + 5 * 2 + 2


class TestLinearSolverStability:
    def test_conjugate_gradient_well_conditioned(self):
        key = random.PRNGKey(0)
        a_base = random.normal(key, (5, 5))
        a = a_base.T @ a_base + jnp.eye(5)
        b = random.normal(random.PRNGKey(1), (5,))
        x, info = conjugate_gradient(a, b, tolerance=1e-6, max_iterations=50)
        assert jnp.linalg.norm(a @ x - b) < 1e-4
        assert bool(info.converged)

    def test_conjugate_gradient_ill_conditioned_needs_preconditioner(self):
        # kappa = 1e12: unpreconditioned float32 CG cannot resolve the 1e-12
        # component of the search direction, so it stalls.  A Jacobi
        # preconditioner turns this diagonal system into the identity and CG
        # converges in one step.
        a = jnp.array([[1e6, 0.0], [0.0, 1e-6]])
        b = jnp.array([1.0, 1.0])
        x_plain, _ = conjugate_gradient(a, b, tolerance=1e-3, max_iterations=100)
        x_pcg, info = conjugate_gradient(
            a, b, tolerance=1e-3, max_iterations=100, preconditioner=1.0 / jnp.diag(a)
        )
        rel = lambda x: jnp.linalg.norm(a @ x - b) / jnp.linalg.norm(b)  # noqa: E731
        assert rel(x_pcg) < 1e-3
        assert rel(x_pcg) <= rel(x_plain)
        assert int(info.iteration) <= 2

    def test_gradient_descent_convergence(self):
        a = jnp.array([[2.0, 0.5], [0.5, 3.0]])
        b = jnp.array([1.0, 2.0])

        def quadratic(x):
            return 0.5 * x @ a @ x - b @ x

        x_opt, _ = gradient_descent(quadratic, jnp.zeros(2), learning_rate=0.1, max_iterations=200)
        assert jnp.allclose(x_opt, jnp.linalg.solve(a, b), atol=1e-3)


class TestNumericalEdgeCases:
    def test_extreme_float32_values(self):
        # float32 max is ~3.4e38; these are within range but naive exp/log fail.
        large = jnp.array([1e30, 1e37, 3e38], dtype=jnp.float32)
        assert jnp.all(jnp.isfinite(safe_log(large)))
        assert jnp.allclose(jnp.sum(stable_softmax(large)), 1.0)
        small = jnp.array([1e-30, 1e-38, 1e-45], dtype=jnp.float32)
        assert jnp.all(jnp.isfinite(safe_sqrt(small)))

    def test_softmax_matches_across_precisions(self):
        x_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        x_f16 = x_f32.astype(jnp.float16)
        assert jnp.allclose(
            stable_softmax(x_f32 * 100), stable_softmax(x_f16 * 100).astype(jnp.float32), atol=1e-3
        )

    def test_gradient_explosion_prevention(self):
        def unstable(x):
            return jnp.sum(jnp.exp(x * 10))

        raw = grad(unstable)(jnp.array([1.0, 2.0, 3.0]))
        clipped = clip_gradients({"grad": raw}, max_norm=1.0)["grad"]
        assert jnp.linalg.norm(clipped) <= 1.0 + 1e-6

    def test_loss_computation_stability(self):
        logits = jnp.array([[1000.0, 999.0, 1001.0], [500.0, 501.0, 499.0]])
        labels = jnp.array([2, 1])
        loss_stable = -log_softmax_stable(logits)[jnp.arange(2), labels]
        loss_jax = -jax.nn.log_softmax(logits)[jnp.arange(2), labels]
        assert jnp.allclose(loss_stable, loss_jax, atol=1e-6)
        assert jnp.all(jnp.isfinite(loss_stable))


class TestRoundoffErrorAccumulation:
    def test_kahan_summation_beats_naive_in_float32(self):
        # 1e6 + 10000 * 1e-3: each 1e-3 is below half an ulp of 1e6 in float32
        # (ulp = 0.0625), so naive sequential accumulation drops every term.
        big = jnp.float32(1e6)
        small = jnp.full(10_000, 1e-3, dtype=jnp.float32)

        def naive(carry, x):
            return carry + x, None

        def kahan(carry, x):
            total, comp = carry
            y = x - comp
            t = total + y
            comp = (t - total) - y
            return (t, comp), None

        naive_sum, _ = jax.lax.scan(naive, big, small)
        (kahan_sum, _), _ = jax.lax.scan(kahan, (big, jnp.float32(0.0)), small)
        exact = 1e6 + 10.0
        assert abs(float(kahan_sum) - exact) < abs(float(naive_sum) - exact)
        assert abs(float(kahan_sum) - exact) < 0.1

    def test_iterative_refinement_reduces_residual(self):
        key = random.PRNGKey(42)
        n = 10
        a = random.normal(key, (n, n))
        a = a.T @ a + 0.01 * jnp.eye(n)
        x_true = random.normal(random.PRNGKey(1), (n,))
        b = a @ x_true

        x0 = jnp.linalg.solve(a, b)
        residual = b - a @ x0
        x_refined = x0 + jnp.linalg.solve(a, residual)

        r0 = jnp.linalg.norm(b - a @ x0)
        r1 = jnp.linalg.norm(b - a @ x_refined)
        # One refinement step should not make the residual worse and should
        # leave it at float32 round-off level relative to ||b||.
        assert r1 <= r0 * 1.5 + 1e-6
        assert r1 / jnp.linalg.norm(b) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__])
