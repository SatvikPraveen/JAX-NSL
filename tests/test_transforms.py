# tests/test_transforms.py

import jax
import jax.numpy as jnp
import pytest
from jax import jit, vmap, grad, random
from src.transforms.jit_utils import jit_with_static, efficient_jit, benchmark_jit
from src.transforms.vmap_utils import batched_matmul, batched_gradient, parallel_apply
from src.transforms.scan_utils import cumulative_sum, rnn_scan, solve_ode
from src.transforms.control_flow import safe_divide, clip_gradient, stable_softmax


class TestJitUtils:
    """Test JIT compilation utilities."""
    
    def test_jit_with_static(self):
        """Test JIT with static arguments."""
        def compute_power(x, n):
            return x ** n
        
        jitted_fn = jit_with_static(compute_power, static_argnums=(1,))
        
        x = jnp.array([1.0, 2.0, 3.0])
        result = jitted_fn(x, 3)
        expected = x ** 3
        
        assert jnp.allclose(result, expected)
    
    def test_efficient_jit(self):
        """Test efficient JIT compilation."""
        def simple_fn(x):
            return x * 2 + 1
        
        jitted_fn = efficient_jit(simple_fn)
        
        x = jnp.array([1.0, 2.0, 3.0])
        result = jitted_fn(x)
        expected = x * 2 + 1
        
        assert jnp.allclose(result, expected)
    
    def test_benchmark_jit(self):
        """Test JIT benchmarking utility."""
        def matrix_multiply(x, y):
            return jnp.dot(x, y)
        
        key = random.PRNGKey(0)
        x = random.normal(key, (100, 100))
        y = random.normal(key, (100, 100))
        
        warmup_time, run_time = benchmark_jit(matrix_multiply, x, y, 
                                            warmup_runs=3, benchmark_runs=5)
        
        assert warmup_time > 0
        assert run_time > 0
        assert isinstance(warmup_time, float)
        assert isinstance(run_time, float)


class TestVmapUtils:
    """Test vectorization utilities."""
    
    def test_batched_matmul(self):
        """Test batched matrix multiplication."""
        batch_size = 4
        dim = 3
        
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        
        A = random.normal(key1, (batch_size, dim, dim))
        B = random.normal(key2, (batch_size, dim, dim))
        
        result = batched_matmul(A, B)
        
        # Check shape
        assert result.shape == (batch_size, dim, dim)
        
        # Compare with manual batch computation
        expected = jnp.stack([A[i] @ B[i] for i in range(batch_size)])
        assert jnp.allclose(result, expected)
    
    def test_batched_gradient(self):
        """Test batched gradient computation."""
        def quadratic(x):
            return jnp.sum(x**2)
        
        batch_size = 5
        dim = 3
        
        key = random.PRNGKey(42)
        x_batch = random.normal(key, (batch_size, dim))
        
        grads = batched_gradient(quadratic, x_batch)
        
        # Check shape
        assert grads.shape == (batch_size, dim)
        
        # Compare with individual gradients
        grad_fn = grad(quadratic)
        expected = jnp.stack([grad_fn(x_batch[i]) for i in range(batch_size)])
        assert jnp.allclose(grads, expected)
    
    def test_parallel_apply(self):
        """Test parallel function application."""
        def square_and_add(x, offset):
            return x**2 + offset
        
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])
        offsets = jnp.array([0.1, 0.2, 0.3, 0.4])
        
        results = parallel_apply(square_and_add, xs, offsets)
        expected = xs**2 + offsets
        
        assert jnp.allclose(results, expected)


class TestScanUtils:
    """Test scan operation utilities."""
    
    def test_cumulative_sum(self):
        """Test cumulative sum implementation."""
        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = cumulative_sum(xs)
        expected = jnp.cumsum(xs)
        
        assert jnp.allclose(result, expected)
    
    def test_rnn_scan(self):
        """Test RNN-style scanning."""
        # Simple RNN cell: h_t = tanh(W_h * h_{t-1} + W_x * x_t)
        hidden_dim = 4
        input_dim = 3
        seq_len = 5
        
        key = random.PRNGKey(0)
        key1, key2, key3 = random.split(key, 3)
        
        W_h = random.normal(key1, (hidden_dim, hidden_dim))
        W_x = random.normal(key2, (hidden_dim, input_dim))
        inputs = random.normal(key3, (seq_len, input_dim))
        h0 = jnp.zeros(hidden_dim)
        
        def rnn_cell(h, x):
            h_new = jnp.tanh(W_h @ h + W_x @ x)
            return h_new, h_new
        
        final_h, all_h = rnn_scan(rnn_cell, h0, inputs)
        
        # Check shapes
        assert final_h.shape == (hidden_dim,)
        assert all_h.shape == (seq_len, hidden_dim)
        
        # Manual verification for first step
        h1_manual = jnp.tanh(W_h @ h0 + W_x @ inputs[0])
        assert jnp.allclose(all_h[0], h1_manual)
    
    def test_solve_ode(self):
        """Test ODE solving with scan."""
        # Simple ODE: dy/dt = -y, solution: y(t) = y0 * exp(-t)
        def dydt(y, t):
            return -y
        
        y0 = jnp.array([2.0])
        t_span = jnp.linspace(0, 1, 11)
        
        solution = solve_ode(dydt, y0, t_span)
        
        # Analytical solution
        expected = y0 * jnp.exp(-t_span)
        
        # Should be reasonably close (Euler method is approximate)
        assert jnp.allclose(solution[:, 0], expected, rtol=1e-1)


class TestControlFlow:
    """Test control flow utilities."""
    
    def test_safe_divide(self):
        """Test safe division implementation."""
        x = jnp.array([1.0, 2.0, 3.0])
        y = jnp.array([2.0, 0.0, 0.5])
        
        result = safe_divide(x, y)
        
        # Check non-zero divisions
        assert jnp.allclose(result[0], 0.5)
        assert jnp.allclose(result[2], 6.0)
        
        # Check zero division handling
        assert jnp.isfinite(result[1])  # Should not be NaN/Inf
    
    def test_clip_gradient(self):
        """Test gradient clipping."""
        def loss_fn(x):
            return jnp.sum(x**2)
        
        x = jnp.array([10.0, -5.0, 2.0])
        max_norm = 1.0
        
        clipped_loss = clip_gradient(loss_fn, max_norm)
        grad_fn = grad(clipped_loss)
        
        grad_val = grad_fn(x)
        grad_norm = jnp.linalg.norm(grad_val)
        
        # Gradient norm should be clipped
        assert grad_norm <= max_norm + 1e-6
    
    def test_stable_softmax(self):
        """Test numerically stable softmax."""
        # Test with large values that would overflow normal softmax
        x = jnp.array([1000.0, 999.0, 1001.0])
        
        result = stable_softmax(x)
        
        # Should sum to 1
        assert jnp.allclose(jnp.sum(result), 1.0)
        
        # Should be positive
        assert jnp.all(result > 0)
        
        # Should not have NaN or Inf
        assert jnp.all(jnp.isfinite(result))


class TestTransformComposition:
    """Test composition of multiple transforms."""
    
    def test_jit_vmap_composition(self):
        """Test JIT + vmap composition."""
        def matrix_vector_product(A, x):
            return A @ x
        
        # Create batched version
        batched_fn = vmap(matrix_vector_product, in_axes=(0, 0))
        jitted_batched_fn = jit(batched_fn)
        
        batch_size = 3
        dim = 4
        
        key = random.PRNGKey(0)
        key1, key2 = random.split(key)
        
        A_batch = random.normal(key1, (batch_size, dim, dim))
        x_batch = random.normal(key2, (batch_size, dim))
        
        result = jitted_batched_fn(A_batch, x_batch)
        
        # Check shape and correctness
        assert result.shape == (batch_size, dim)
        
        expected = jnp.stack([A_batch[i] @ x_batch[i] for i in range(batch_size)])
        assert jnp.allclose(result, expected)
    
    def test_grad_jit_vmap_composition(self):
        """Test grad + JIT + vmap composition."""
        def quadratic_loss(w, x, y):
            pred = jnp.dot(w, x)
            return 0.5 * (pred - y)**2
        
        # Batched gradient computation
        grad_fn = grad(quadratic_loss, argnums=0)
        batched_grad_fn = vmap(grad_fn, in_axes=(None, 0, 0))
        jitted_batched_grad_fn = jit(batched_grad_fn)
        
        dim = 5
        batch_size = 10
        
        key = random.PRNGKey(42)
        key1, key2, key3 = random.split(key, 3)
        
        w = random.normal(key1, (dim,))
        x_batch = random.normal(key2, (batch_size, dim))
        y_batch = random.normal(key3, (batch_size,))
        
        grads = jitted_batched_grad_fn(w, x_batch, y_batch)
        
        assert grads.shape == (batch_size, dim)
    
    def test_scan_vmap_composition(self):
        """Test scan + vmap composition."""
        def step_fn(carry, x):
            new_carry = carry + x
            output = new_carry
            return new_carry, output
        
        # Apply scan to multiple sequences
        def multi_scan(init_carries, sequences):
            return vmap(lambda carry, seq: jax.lax.scan(step_fn, carry, seq))(
                init_carries, sequences)
        
        batch_size = 3
        seq_len = 5
        
        init_carries = jnp.array([0.0, 1.0, 2.0])
        sequences = jnp.ones((batch_size, seq_len))
        
        final_carries, outputs = multi_scan(init_carries, sequences)
        
        assert final_carries.shape == (batch_size,)
        assert outputs.shape == (batch_size, seq_len)


class TestTransformEdgeCases:
    """Test edge cases for transforms."""
    
    def test_empty_batch_vmap(self):
        """Test vmap with empty batches."""
        def simple_fn(x):
            return x * 2
        
        empty_batch = jnp.zeros((0, 3))
        result = vmap(simple_fn)(empty_batch)
        
        assert result.shape == (0, 3)
    
    def test_scalar_vmap(self):
        """Test vmap with scalar operations."""
        def scalar_fn(x):
            return x**2 + 1
        
        scalars = jnp.array([1.0, 2.0, 3.0])
        result = vmap(scalar_fn)(scalars)
        expected = scalars**2 + 1
        
        assert jnp.allclose(result, expected)
    
    def test_nested_scan(self):
        """Test nested scan operations."""
        def outer_step(outer_carry, outer_x):
            def inner_step(inner_carry, inner_x):
                return inner_carry + inner_x, inner_carry + inner_x
            
            inner_seq = jnp.ones(3) * outer_x
            final_inner, inner_outputs = jax.lax.scan(inner_step, 0.0, inner_seq)
            
            new_outer_carry = outer_carry + final_inner
            return new_outer_carry, inner_outputs
        
        outer_seq = jnp.array([1.0, 2.0])
        final_carry, outputs = jax.lax.scan(outer_step, 0.0, outer_seq)
        
        assert outputs.shape == (2, 3)
        assert jnp.isfinite(final_carry)


if __name__ == "__main__":
    pytest.main([__file__])