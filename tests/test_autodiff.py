# tests/test_autodiff.py

import jax
import jax.numpy as jnp
import pytest
from jax import grad, jacobian, hessian, jvp, vjp
from src.autodiff.grad_jac_hess import compute_gradient, compute_jacobian, compute_hessian
from src.autodiff.custom_vjp import custom_sqrt_vjp, smooth_abs_vjp
from src.autodiff.custom_jvp import custom_sqrt_jvp, smooth_abs_jvp


class TestGradJacHess:
    """Test gradient, jacobian, and hessian computations."""
    
    def test_gradient_simple(self):
        """Test gradient computation for simple functions."""
        def f(x):
            return jnp.sum(x**2)
        
        x = jnp.array([1.0, 2.0, 3.0])
        grad_f = compute_gradient(f, x)
        expected = 2.0 * x
        assert jnp.allclose(grad_f, expected)
    
    def test_gradient_multivariate(self):
        """Test gradient for multivariate functions."""
        def f(x):
            return x[0]**2 + 2*x[1]**2 + x[0]*x[1]
        
        x = jnp.array([2.0, 3.0])
        grad_f = compute_gradient(f, x)
        expected = jnp.array([2*x[0] + x[1], 4*x[1] + x[0]])
        assert jnp.allclose(grad_f, expected)
    
    def test_jacobian_vector_function(self):
        """Test jacobian computation for vector-valued functions."""
        def f(x):
            return jnp.array([x[0]**2 + x[1], x[0] - x[1]**2])
        
        x = jnp.array([2.0, 3.0])
        jac = compute_jacobian(f, x)
        expected = jnp.array([[2*x[0], 1.0], [1.0, -2*x[1]]])
        assert jnp.allclose(jac, expected)
    
    def test_hessian_scalar_function(self):
        """Test hessian computation for scalar functions."""
        def f(x):
            return x[0]**3 + x[1]**2 + x[0]*x[1]
        
        x = jnp.array([1.0, 2.0])
        hess = compute_hessian(f, x)
        expected = jnp.array([[6*x[0], 1.0], [1.0, 2.0]])
        assert jnp.allclose(hess, expected)


class TestCustomVJP:
    """Test custom VJP implementations."""
    
    def test_custom_sqrt_vjp(self):
        """Test custom sqrt with VJP."""
        x = jnp.array([4.0, 9.0, 16.0])
        
        # Forward pass
        y = custom_sqrt_vjp(x)
        expected_y = jnp.sqrt(x)
        assert jnp.allclose(y, expected_y)
        
        # Backward pass
        grad_fn = grad(lambda x: jnp.sum(custom_sqrt_vjp(x)))
        grad_val = grad_fn(x)
        expected_grad = 0.5 / jnp.sqrt(x)
        assert jnp.allclose(grad_val, expected_grad)
    
    def test_smooth_abs_vjp(self):
        """Test smooth absolute value with VJP."""
        x = jnp.array([-2.0, -0.1, 0.0, 0.1, 2.0])
        
        # Forward pass
        y = smooth_abs_vjp(x)
        assert jnp.all(y >= 0)  # Should be non-negative
        
        # Should be smooth at zero
        x_near_zero = jnp.array([-1e-3, 0.0, 1e-3])
        y_smooth = smooth_abs_vjp(x_near_zero, eps=1e-2)
        grad_fn = grad(lambda x: jnp.sum(smooth_abs_vjp(x, eps=1e-2)))
        grad_val = grad_fn(x_near_zero)
        
        # Gradient should be finite everywhere
        assert jnp.all(jnp.isfinite(grad_val))


class TestCustomJVP:
    """Test custom JVP implementations."""
    
    def test_custom_sqrt_jvp(self):
        """Test custom sqrt with JVP."""
        x = jnp.array([4.0, 9.0])
        v = jnp.array([1.0, 1.0])
        
        # Forward mode AD
        primals, tangents = jvp(custom_sqrt_jvp, (x,), (v,))
        
        expected_primals = jnp.sqrt(x)
        expected_tangents = 0.5 / jnp.sqrt(x) * v
        
        assert jnp.allclose(primals, expected_primals)
        assert jnp.allclose(tangents, expected_tangents)
    
    def test_smooth_abs_jvp(self):
        """Test smooth absolute value with JVP."""
        x = jnp.array([-1.0, 0.0, 1.0])
        v = jnp.array([1.0, 1.0, 1.0])
        
        primals, tangents = jvp(lambda x: smooth_abs_jvp(x, eps=1e-2), (x,), (v,))
        
        # Should be smooth and finite
        assert jnp.all(jnp.isfinite(primals))
        assert jnp.all(jnp.isfinite(tangents))


class TestMixedMode:
    """Test combinations of forward and reverse mode AD."""
    
    def test_grad_of_jvp(self):
        """Test gradient of JVP (mixed mode)."""
        def f(x):
            return jnp.sum(x**3)
        
        def jvp_fn(x, v):
            _, tangent = jvp(f, (x,), (v,))
            return tangent
        
        x = jnp.array([1.0, 2.0])
        v = jnp.array([1.0, 0.0])
        
        # Gradient of JVP w.r.t. primal
        grad_jvp = grad(lambda x: jvp_fn(x, v))(x)
        expected = 6 * x * v  # d/dx(3x^2 * v) = 6x * v
        
        assert jnp.allclose(grad_jvp, expected)
    
    def test_jvp_of_grad(self):
        """Test JVP of gradient (mixed mode)."""
        def f(x):
            return jnp.sum(x**4)
        
        grad_f = grad(f)
        
        x = jnp.array([1.0, 2.0])
        v = jnp.array([0.5, 0.5])
        
        _, tangent = jvp(grad_f, (x,), (v,))
        expected = 12 * x**2 * v  # d/dx(4x^3) * v = 12x^2 * v
        
        assert jnp.allclose(tangent, expected)


class TestHigherOrderDerivatives:
    """Test higher-order derivative computations."""
    
    def test_third_derivative(self):
        """Test third derivative computation."""
        def f(x):
            return x**4
        
        x = 2.0
        
        # Third derivative: d^3/dx^3(x^4) = 24x
        third_deriv = grad(grad(grad(f)))(x)
        expected = 24.0 * x
        
        assert jnp.allclose(third_deriv, expected)
    
    def test_mixed_partials(self):
        """Test mixed partial derivatives."""
        def f(x, y):
            return x**2 * y**3
        
        x, y = 2.0, 3.0
        
        # Mixed partial: d^2/dxdy(x^2 * y^3) = 2x * 3y^2 = 6xy^2
        mixed_partial = grad(grad(f, argnums=0), argnums=1)(x, y)
        expected = 6.0 * x * y**2
        
        assert jnp.allclose(mixed_partial, expected)


class TestNumericalStability:
    """Test numerical stability of autodiff operations."""
    
    def test_gradient_numerical_stability(self):
        """Test gradient computation with small numbers."""
        def f(x):
            return jnp.sum(jnp.log(x + 1e-8))
        
        x = jnp.array([1e-6, 1e-4, 1e-2])
        grad_f = grad(f)(x)
        
        # Should not have NaN or Inf
        assert jnp.all(jnp.isfinite(grad_f))
    
    def test_hessian_conditioning(self):
        """Test hessian computation for ill-conditioned functions."""
        def f(x):
            return 1e6 * x[0]**2 + 1e-6 * x[1]**2
        
        x = jnp.array([1.0, 1.0])
        hess = hessian(f)(x)
        
        # Should be finite despite ill-conditioning
        assert jnp.all(jnp.isfinite(hess))
        
        # Check diagonal elements
        assert jnp.allclose(jnp.diag(hess), jnp.array([2e6, 2e-6]))


if __name__ == "__main__":
    pytest.main([__file__])