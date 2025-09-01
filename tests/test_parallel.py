# tests/test_parallel.py

import jax
import jax.numpy as jnp
import pytest
from jax import random, pmap, jit
from jax.sharding import PartitionSpec, Mesh
from jax.experimental import mesh_utils
import numpy as np

from src.parallel.pmap_utils import data_parallel_step, sync_gradients, replicate_params
from src.parallel.pjit_utils import create_mesh, shard_array, partition_params
from src.parallel.collectives import all_reduce_mean, distributed_dot, sync_batch_stats


class TestPmapUtils:
    """Test pmap (data parallel) utilities."""
    
    def test_replicate_params(self):
        """Test parameter replication across devices."""
        # Skip if not enough devices
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for pmap tests")
        
        params = {
            'W': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            'b': jnp.array([0.5, 1.5])
        }
        
        replicated = replicate_params(params)
        
        # Check that parameters are replicated
        assert replicated['W'].shape == (n_devices, 2, 2)
        assert replicated['b'].shape == (n_devices, 2)
        
        # All replicas should be identical
        for i in range(n_devices):
            assert jnp.allclose(replicated['W'][i], params['W'])
            assert jnp.allclose(replicated['b'][i], params['b'])
    
    def test_sync_gradients(self):
        """Test gradient synchronization across devices."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for pmap tests")
        
        # Create different gradients per device
        grad_shape = (n_devices, 3)
        gradients = jnp.arange(n_devices * 3).reshape(grad_shape)
        
        synced_grads = sync_gradients(gradients)
        
        # All devices should have the same (averaged) gradients
        expected_grad = jnp.mean(gradients, axis=0)
        for i in range(n_devices):
            assert jnp.allclose(synced_grads[i], expected_grad)
    
    def test_data_parallel_step(self):
        """Test data parallel training step."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for pmap tests")
        
        def loss_fn(params, batch):
            return jnp.mean((params['W'] @ batch['x'] - batch['y'])**2)
        
        # Initialize parameters and optimizer state
        key = random.PRNGKey(0)
        params = {
            'W': random.normal(key, (2, 3))
        }
        
        # Create batched data (one batch per device)
        batch_size_per_device = 4
        key1, key2 = random.split(key)
        
        x_batch = random.normal(key1, (n_devices, batch_size_per_device, 3))
        y_batch = random.normal(key2, (n_devices, batch_size_per_device, 2))
        
        batch = {'x': x_batch, 'y': y_batch}
        
        # Replicate parameters
        replicated_params = replicate_params(params)
        
        # Perform data parallel step
        new_params, loss = data_parallel_step(loss_fn, replicated_params, batch, lr=0.01)
        
        # Check shapes
        assert new_params['W'].shape == (n_devices, 2, 3)
        assert loss.shape == (n_devices,)
        
        # Parameters should be synchronized across devices
        for i in range(1, n_devices):
            assert jnp.allclose(new_params['W'][0], new_params['W'][i], rtol=1e-5)


class TestPjitUtils:
    """Test pjit (model parallel) utilities."""
    
    def test_create_mesh(self):
        """Test mesh creation for model parallelism."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for pjit tests")
        
        # Try to create a 1D mesh
        mesh_shape = (n_devices,)
        axis_names = ('batch',)
        
        mesh = create_mesh(mesh_shape, axis_names)
        
        assert isinstance(mesh, Mesh)
        assert mesh.shape == {'batch': n_devices}
    
    def test_shard_array(self):
        """Test array sharding across devices."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for pjit tests")
        
        # Create array and partition spec
        array = jnp.ones((8, 4))  # Shape divisible by device count
        partition_spec = PartitionSpec('batch', None)
        
        mesh_shape = (min(n_devices, 2),)  # Use at most 2 devices for simplicity
        mesh = create_mesh(mesh_shape, ('batch',))
        
        with mesh:
            sharded = shard_array(array, partition_spec)
            
            # Check that array is properly sharded
            assert sharded.shape == (8, 4)
    
    def test_partition_params(self):
        """Test parameter partitioning."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for pjit tests")
        
        params = {
            'embeddings': jnp.ones((1000, 128)),  # Partition along vocab dimension
            'dense': jnp.ones((128, 256)),        # Partition along output dimension
            'bias': jnp.ones((256,))              # Replicated
        }
        
        partition_specs = {
            'embeddings': PartitionSpec('vocab', None),
            'dense': PartitionSpec(None, 'hidden'),
            'bias': PartitionSpec(None)
        }
        
        mesh_shape = (min(n_devices, 2),)
        mesh = create_mesh(mesh_shape, ('vocab',))
        
        with mesh:
            partitioned = partition_params(params, partition_specs)
            
            # Check that parameters are properly shaped
            assert partitioned['embeddings'].shape == (1000, 128)
            assert partitioned['dense'].shape == (128, 256)
            assert partitioned['bias'].shape == (256,)


class TestCollectives:
    """Test collective communication operations."""
    
    def test_all_reduce_mean(self):
        """Test all-reduce mean operation."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for collective tests")
        
        # Create different values per device
        values = jnp.arange(n_devices, dtype=jnp.float32)
        
        # Simulate pmap context
        @pmap
        def test_reduce(x):
            return all_reduce_mean(x)
        
        result = test_reduce(values)
        expected = jnp.mean(values)
        
        # All devices should have the mean value
        for i in range(n_devices):
            assert jnp.allclose(result[i], expected)
    
    def test_distributed_dot(self):
        """Test distributed dot product."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for collective tests")
        
        # Create vectors distributed across devices
        dim = 8
        x = random.normal(random.PRNGKey(0), (n_devices, dim // n_devices))
        y = random.normal(random.PRNGKey(1), (n_devices, dim // n_devices))
        
        @pmap
        def test_distributed_dot(x_shard, y_shard):
            return distributed_dot(x_shard, y_shard)
        
        result = test_distributed_dot(x, y)
        
        # Compute expected result
        x_full = x.reshape(-1)
        y_full = y.reshape(-1)
        expected = jnp.dot(x_full, y_full)
        
        # All devices should have the same result
        for i in range(n_devices):
            assert jnp.allclose(result[i], expected, rtol=1e-5)
    
    def test_sync_batch_stats(self):
        """Test batch statistics synchronization."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for collective tests")
        
        # Create different batch stats per device
        means = random.normal(random.PRNGKey(0), (n_devices, 3))
        vars = random.uniform(random.PRNGKey(1), (n_devices, 3), minval=0.1, maxval=2.0)
        
        @pmap
        def test_sync_stats(mean, var):
            return sync_batch_stats(mean, var)
        
        synced_means, synced_vars = test_sync_stats(means, vars)
        
        # All devices should have synchronized stats
        expected_mean = jnp.mean(means, axis=0)
        expected_var = jnp.mean(vars, axis=0)
        
        for i in range(n_devices):
            assert jnp.allclose(synced_means[i], expected_mean)
            assert jnp.allclose(synced_vars[i], expected_var)


class TestParallelTraining:
    """Test end-to-end parallel training scenarios."""
    
    def test_data_parallel_training_loop(self):
        """Test complete data parallel training loop."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices for parallel training")
        
        # Simple linear model
        def model(params, x):
            return params['W'] @ x + params['b']
        
        def loss_fn(params, batch):
            preds = model(params, batch['x'])
            return jnp.mean((preds - batch['y'])**2)
        
        # Initialize
        key = random.PRNGKey(42)
        key1, key2, key3 = random.split(key, 3)
        
        params = {
            'W': random.normal(key1, (2, 3)),
            'b': random.normal(key2, (2,))
        }
        
        # Create training data
        batch_size_per_device = 8
        x_batch = random.normal(key3, (n_devices, batch_size_per_device, 3))
        y_batch = random.normal(key3, (n_devices, batch_size_per_device, 2))
        batch = {'x': x_batch, 'y': y_batch}
        
        # Replicate parameters
        replicated_params = replicate_params(params)
        
        # Training step
        new_params, losses = data_parallel_step(loss_fn, replicated_params, batch, lr=0.01)
        
        # Verify training occurred
        assert not jnp.allclose(new_params['W'], replicated_params['W'])
        assert not jnp.allclose(new_params['b'], replicated_params['b'])
        
        # Verify synchronization
        for i in range(1, n_devices):
            assert jnp.allclose(new_params['W'][0], new_params['W'][i])
            assert jnp.allclose(new_params['b'][0], new_params['b'][i])
    
    @pytest.mark.skipif(jax.device_count() < 4, reason="Need at least 4 devices")
    def test_model_parallel_computation(self):
        """Test model parallel computation."""
        # This test requires more devices and is more complex
        # Simplified test for model parallelism
        
        def large_matmul(x, w1, w2):
            # First layer
            h = x @ w1
            # Second layer  
            return h @ w2
        
        batch_size = 8
        input_dim = 64
        hidden_dim = 128
        output_dim = 32
        
        key = random.PRNGKey(0)
        key1, key2, key3 = random.split(key, 3)
        
        x = random.normal(key1, (batch_size, input_dim))
        w1 = random.normal(key2, (input_dim, hidden_dim))
        w2 = random.normal(key3, (hidden_dim, output_dim))
        
        # For now, just test that the computation works
        result = large_matmul(x, w1, w2)
        assert result.shape == (batch_size, output_dim)


class TestParallelEdgeCases:
    """Test edge cases in parallel operations."""
    
    def test_single_device_pmap(self):
        """Test pmap behavior with single device."""
        def simple_fn(x):
            return x * 2
        
        x = jnp.array([1.0, 2.0, 3.0])
        # Add batch dimension for pmap
        x_batched = x.reshape(1, -1)
        
        pmapped_fn = pmap(simple_fn)
        result = pmapped_fn(x_batched)
        
        assert result.shape == (1, 3)
        assert jnp.allclose(result[0], x * 2)
    
    def test_uneven_batch_sizes(self):
        """Test handling of uneven batch sizes across devices."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices")
        
        # Create data that doesn't divide evenly
        total_batch_size = n_devices * 3 + 1  # Uneven
        data = jnp.arange(total_batch_size, dtype=jnp.float32)
        
        # Pad to make it divisible
        remainder = total_batch_size % n_devices
        if remainder != 0:
            padding = n_devices - remainder
            data = jnp.concatenate([data, jnp.zeros(padding)])
        
        # Reshape for pmap
        data_per_device = data.reshape(n_devices, -1)
        
        @pmap
        def process_batch(x):
            return jnp.sum(x)
        
        results = process_batch(data_per_device)
        
        assert results.shape == (n_devices,)
        assert jnp.all(jnp.isfinite(results))
    
    def test_gradient_accumulation_parallel(self):
        """Test gradient accumulation in parallel setting."""
        n_devices = jax.device_count()
        if n_devices < 2:
            pytest.skip("Need at least 2 devices")
        
        def loss_fn(params, x, y):
            pred = params['w'] * x
            return (pred - y)**2
        
        # Initialize
        params = {'w': jnp.array(1.0)}
        replicated_params = replicate_params(params)
        
        # Multiple microbatches
        n_microbatches = 3
        x_data = random.normal(random.PRNGKey(0), (n_devices, n_microbatches, 2))
        y_data = random.normal(random.PRNGKey(1), (n_devices, n_microbatches, 2))
        
        @pmap
        def accumulate_gradients(params, x_batches, y_batches):
            def compute_grad(x, y):
                return jax.grad(loss_fn)(params, x, y)
            
            # Compute gradients for each microbatch
            grads = jax.vmap(compute_grad)(x_batches, y_batches)
            
            # Accumulate (average) gradients
            avg_grad = jax.tree_map(lambda g: jnp.mean(g, axis=0), grads)
            
            # Synchronize across devices
            return sync_gradients(avg_grad)
        
        accumulated_grads = accumulate_gradients(replicated_params, x_data, y_data)
        
        # Check that gradients are synchronized
        for i in range(1, n_devices):
            assert jnp.allclose(accumulated_grads['w'][0], accumulated_grads['w'][i])


if __name__ == "__main__":
    pytest.main([__file__])