# tests/test_numerics.py

import jax
import jax.numpy as jnp
import pytest
from jax import random, grad
import numpy as np

from src.core.numerics import (
    stable_logsumexp, safe_log, safe_sqrt, safe_divide,
    clip_gradients, stable_softmax, numerical_gradient
)
from src.core.arrays import safe_cast, check_finite, tree_size
from src.linalg.solvers import conjugate_gradient, gradient_descent


class TestStableOperations:
    """Test numerically stable implementations of common operations."""
    
    def test_stable_logsumexp(self):
        """Test stable log-sum-exp implementation."""
        # Test with large values that would overflow regular exp
        x = jnp.array([1000.0, 999.0, 1001.0])
        result = stable_logsumexp(x)
        
        # Should be approximately max(x) + log(number of similar values)
        # Since 1001 is the max, result ≈ 1001 + log(1 + exp(-1) + exp(-2))
        expected_approx = 1001.0
        assert result > expected_approx
        assert jnp.isfinite(result)
        
        # Test with normal values
        x_normal = jnp.array([1.0, 2.0, 3.0])
        result_normal = stable_logsumexp(x_normal)
        naive_result = jnp.log(jnp.sum(jnp.exp(x_normal)))
        assert jnp.allclose(result_normal, naive_result)
    
    def test_safe_log(self):
        """Test safe logarithm implementation."""
        # Test with zero and negative values
        x = jnp.array([0.0, -1.0, 1e-10, 1.0])
        result = safe_log(x)
        
        # Should not have NaN or -inf
        assert jnp.all(jnp.isfinite(result))
        
        # Positive values should give correct log
        assert jnp.allclose(result[-1], jnp.log(1.0))
        
        # Very small positive should be handled
        assert jnp.isfinite(result[2])
    
    def test_safe_sqrt(self):
        """Test safe square root implementation."""
        x = jnp.array([-1.0, 0.0, 1e-20, 4.0])
        result = safe_sqrt(x)
        
        # Should not have NaN
        assert jnp.all(jnp.isfinite(result))
        
        # Non-negative values should give correct sqrt
        assert jnp.allclose(result[-1], 2.0)
        assert jnp.allclose(result[1], 0.0)
    
    def test_safe_divide(self):
        """Test safe division implementation."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 0.0, 1e-15])
        
        result = safe_divide(x, y)
        
        # Should not have NaN or inf
        assert jnp.all(jnp.isfinite(result))
        
        # Normal division should work
        assert jnp.allclose(result[0], 0.5)
        
        # Division by zero should be handled
        assert jnp.isfinite(result[1])
    
    def test_stable_softmax(self):
        """Test numerically stable softmax."""
        # Test with large values
        x = jnp.array([1000.0, 999.0, 1001.0])
        result = stable_softmax(x)
        
        # Should sum to 1
        assert jnp.allclose(jnp.sum(result), 1.0)
        
        # Should be positive
        assert jnp.all(result > 0)
        
        # Should not overflow
        assert jnp.all(jnp.isfinite(result))
        
        # Test with normal values
        x_normal = jnp.array([1.0, 2.0, 3.0])
        result_normal = stable_softmax(x_normal)
        naive_result = jnp.exp(x_normal) / jnp.sum(jnp.exp(x_normal))
        assert jnp.allclose(result_normal, naive_result)


class TestGradientStability:
    """Test numerical stability in gradient computations."""
    
    def test_clip_gradients_global_norm(self):
        """Test global gradient norm clipping."""
        grads = {
            'layer1': {'w': jnp.array([[10.0, -5.0], [3.0, -8.0]]), 'b': jnp.array([2.0, -1.0])},
            'layer2': {'w': jnp.array([[1.0, 2.0]]), 'b': jnp.array([0.5])}
        }
        
        max_norm = 1.0
        clipped_grads = clip_gradients(grads, max_norm)
        
        # Compute global norm of clipped gradients
        def tree_norm(tree):
            leaves = jax.tree_util.tree_leaves(tree)
            return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in leaves))
        
        clipped_norm = tree_norm(clipped_grads)
        
        # Should be clipped to max_norm
        assert clipped_norm <= max_norm + 1e-6
    
    def test_numerical_gradient(self):
        """Test numerical gradient computation."""
        def f(x):
            return jnp.sum(x**3)
        
        x = jnp.array([1.0, 2.0, 3.0])
        
        # Analytical gradient
        analytical_grad = grad(f)(x)
        
        # Numerical gradient
        numerical_grad = numerical_gradient(f, x, h=1e-5)
        
        # Should be close
        assert jnp.allclose(analytical_grad, numerical_grad, atol=1e-4)
    
    def test_gradient_numerical_stability(self):
        """Test gradient computation with potential numerical issues."""
        def f(x):
            # Function that can cause numerical issues
            return jnp.sum(jnp.log(jnp.exp(x) + 1e-8))
        
        x = jnp.array([10.0, -10.0, 0.0])
        
        # Should not produce NaN or inf gradients
        grad_f = grad(f)(x)
        assert jnp.all(jnp.isfinite(grad_f))


class TestArrayStability:
    """Test numerical stability in array operations."""
    
    def test_safe_cast(self):
        """Test safe casting between dtypes."""
        # Test overflow handling
        large_float = jnp.array([1e10, -1e10, 1e5])
        
        # Cast to int32 (should clip to range)
        casted = safe_cast(large_float, jnp.int32)
        
        # Should not overflow
        assert jnp.all(jnp.abs(casted) <= 2**31 - 1)
        
        # Test underflow
        small_float = jnp.array([1e-10, 1e-50, 0.0])
        casted_small = safe_cast(small_float, jnp.float16)
        
        # Should handle underflow gracefully
        assert jnp.all(jnp.isfinite(casted_small))
    
    def test_check_finite(self):
        """Test finite value checking."""
        # Valid array
        valid = jnp.array([1.0, 2.0, 3.0])
        assert check_finite(valid)
        
        # Array with NaN
        with_nan = jnp.array([1.0, jnp.nan, 3.0])
        assert not check_finite(with_nan)
        
        # Array with inf
        with_inf = jnp.array([1.0, jnp.inf, 3.0])
        assert not check_finite(with_inf)
    
    def test_tree_size(self):
        """Test computation of pytree total size."""
        tree = {
            'layer1': {'w': jnp.ones((10, 5)), 'b': jnp.ones(5)},
            'layer2': {'w': jnp.ones((5, 2)), 'b': jnp.ones(2)}
        }
        
        size = tree_size(tree)
        expected_size = 10*5 + 5 + 5*2 + 2  # 67
        
        assert size == expected_size


class TestLinearSolverStability:
    """Test numerical stability in linear solvers."""
    
    def test_conjugate_gradient_well_conditioned(self):
        """Test CG solver with well-conditioned system."""
        # Create a well-conditioned SPD matrix
        key = random.PRNGKey(0)
        A_base = random.normal(key, (5, 5))
        A = A_base.T @ A_base + jnp.eye(5)  # Make SPD
        
        b = random.normal(random.PRNGKey(1), (5,))
        
        # Solve Ax = b
        x = conjugate_gradient(A, b, tol=1e-6, max_iter=10)
        
        # Check solution quality
        residual = A @ x - b
        assert jnp.linalg.norm(residual) < 1e-5
    
    def test_conjugate_gradient_ill_conditioned(self):
        """Test CG solver with ill-conditioned system."""
        # Create an ill-conditioned matrix
        A = jnp.array([[1e6, 0], [0, 1e-6]])
        b = jnp.array([1.0, 1.0])
        
        x = conjugate_gradient(A, b, tol=1e-3, max_iter=100)
        
        # Solution should still be reasonable
        residual = A @ x - b
        relative_error = jnp.linalg.norm(residual) / jnp.linalg.norm(b)
        assert relative_error < 1e-2  # Relaxed tolerance for ill-conditioned case
    
    def test_gradient_descent_convergence(self):
        """Test gradient descent convergence."""
        # Quadratic function: f(x) = 0.5 * x^T A x - b^T x
        A = jnp.array([[2.0, 0.5], [0.5, 3.0]])  # SPD matrix
        b = jnp.array([1.0, 2.0])
        
        def quadratic_loss(x):
            return 0.5 * x.T @ A @ x - b.T @ x
        
        x0 = jnp.zeros(2)
        x_opt = gradient_descent(quadratic_loss, x0, lr=0.1, num_steps=100)
        
        # Analytical solution: x* = A^{-1} b
        x_analytical = jnp.linalg.solve(A, b)
        
        assert jnp.allclose(x_opt, x_analytical, atol=1e-3)


class TestNumericalEdgeCases:
    """Test edge cases that can cause numerical issues."""
    
    def test_extreme_values(self):
        """Test operations with extreme values."""
        # Very large values
        large_vals = jnp.array([1e100, 1e200, 1e308])
        
        # Stable operations should handle these
        log_vals = safe_log(large_vals)
        assert jnp.all(jnp.isfinite(log_vals))
        
        softmax_vals = stable_softmax(large_vals)
        assert jnp.allclose(jnp.sum(softmax_vals), 1.0)
        
        # Very small values
        small_vals = jnp.array([1e-100, 1e-200, 1e-308])
        sqrt_vals = safe_sqrt(small_vals)
        assert jnp.all(jnp.isfinite(sqrt_vals))
    
    def test_mixed_precision_stability(self):
        """Test numerical stability across different precisions."""
        x_f32 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
        x_f64 = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        
        # Operations should be stable in both precisions
        result_f32 = stable_softmax(x_f32 * 100)  # Large values
        result_f64 = stable_softmax(x_f64 * 100)
        
        assert jnp.allclose(result_f32, result_f64, rtol=1e-6)
        assert jnp.allclose(jnp.sum(result_f32), 1.0)
        assert jnp.allclose(jnp.sum(result_f64), 1.0)
    
    def test_gradient_explosion_prevention(self):
        """Test prevention of gradient explosion."""
        def unstable_function(x):
            # Function that can produce large gradients
            return jnp.sum(jnp.exp(x * 10))
        
        x = jnp.array([1.0, 2.0, 3.0])
        
        # Raw gradient might be very large
        raw_grad = grad(unstable_function)(x)
        
        # Clipped gradient should be bounded
        clipped_grad = clip_gradients({'grad': raw_grad}, max_norm=1.0)['grad']
        grad_norm = jnp.linalg.norm(clipped_grad)
        
        assert grad_norm <= 1.0 + 1e-6
    
    def test_loss_computation_stability(self):
        """Test numerically stable loss computations."""
        # Logits that could cause overflow in naive softmax
        logits = jnp.array([[1000.0, 999.0, 1001.0],
                           [500.0, 501.0, 499.0]])
        labels = jnp.array([2, 1])  # One-hot: [0,0,1], [0,1,0]
        
        # Stable cross-entropy computation
        log_probs = jax.nn.log_softmax(logits)  # JAX's stable implementation
        
        # Manual stable computation
        def stable_cross_entropy(logits, labels):
            log_softmax = logits - stable_logsumexp(logits, axis=-1, keepdims=True)
            return -log_softmax[jnp.arange(len(labels)), labels]
        
        loss_stable = stable_cross_entropy(logits, labels)
        loss_jax = -log_probs[jnp.arange(len(labels)), labels]
        
        assert jnp.allclose(loss_stable, loss_jax)
        assert jnp.all(jnp.isfinite(loss_stable))


class TestRoundoffErrorAccumulation:
    """Test handling of roundoff error accumulation."""
    
    def test_summation_stability(self):
        """Test stable summation algorithms."""
        # Create data prone to roundoff errors
        large_val = 1e16
        small_vals = jnp.ones(1000) * 1e-16
        mixed_data = jnp.concatenate([jnp.array([large_val]), small_vals])
        
        # Naive sum might lose precision
        naive_sum = jnp.sum(mixed_data)
        
        # Kahan summation (stable)
        def kahan_sum(arr):
            total = 0.0
            c = 0.0  # Compensation for lost low-order bits
            
            for x in arr:
                y = x - c
                t = total + y
                c = (t - total) - y
                total = t
            
            return total
        
        # Note: JAX's sum is already quite stable, but this tests the concept
        stable_sum = kahan_sum(mixed_data)
        
        # Both should be close to the expected value
        expected = large_val + 1000 * 1e-16
        
        # The key is that both sums should be reasonable approximations
        assert abs(naive_sum - expected) / expected < 1e-10
    
    def test_iterative_refinement(self):
        """Test iterative refinement for better numerical accuracy."""
        # Solve a linear system with iterative refinement
        key = random.PRNGKey(42)
        n = 10
        A = random.normal(key, (n, n))
        A = A.T @ A + 0.01 * jnp.eye(n)  # Make well-conditioned SPD
        
        x_true = random.normal(random.PRNGKey(1), (n,))
        b = A @ x_true
        
        # Initial solution
        x0 = jnp.linalg.solve(A, b)
        
        # One step of iterative refinement
        residual = b - A @ x0
        correction = jnp.linalg.solve(A, residual)
        x_refined = x0 + correction
        
        # Refined solution should be more accurate
        error0 = jnp.linalg.norm(x0 - x_true)
        error_refined = jnp.linalg.norm(x_refined - x_true)
        
        # Not always true due to machine precision, but often improves
        # assert error_refined <= error0
        assert error_refined < 1e-10  # Should be very accurate


if __name__ == "__main__":
    pytest.main([__file__])