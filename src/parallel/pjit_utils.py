# File location: jax-nsl/src/parallel/pjit_utils.py

"""
Model parallelism utilities using pjit and mesh.

This module provides utilities for sharding models across devices
using JAX's pjit transformation with PartitionSpec.
"""

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec, Mesh
from jax import pjit
from typing import Any, Callable, Dict, Tuple, Optional, Union
import functools


def create_mesh(mesh_shape: Tuple[int, ...], 
               axis_names: Tuple[str, ...]) -> Mesh:
    """Create device mesh for model parallelism.
    
    Args:
        mesh_shape: Shape of device mesh (e.g., (2, 4) for 2x4 grid)
        axis_names: Names for mesh axes (e.g., ('data', 'model'))
        
    Returns:
        Device mesh
    """
    devices = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(devices, axis_names)


def partition_params(params: Any, 
                    partition_rules: Dict[str, PartitionSpec]) -> Any:
    """Apply partitioning rules to parameters.
    
    Args:
        params: Model parameters
        partition_rules: Mapping from parameter names to partition specs
        
    Returns:
        Partitioned parameters
    """
    def partition_fn(name, param):
        if name in partition_rules:
            return partition_rules[name]
        else:
            # Default: no partitioning
            return PartitionSpec()
    
    return jax.tree_util.tree_map_with_path(
        lambda path, param: (partition_fn('.'.join(str(k.key) for k in path), param), param),
        params
    )


def create_sharded_array(array: jnp.ndarray,
                        mesh: Mesh,
                        partition_spec: PartitionSpec) -> jnp.ndarray:
    """Create sharded array with specified partitioning.
    
    Args:
        array: Input array
        mesh: Device mesh
        partition_spec: Partitioning specification
        
    Returns:
        Sharded array
    """
    with mesh:
        return jax.device_put(array, jax.sharding.NamedSharding(mesh, partition_spec))


def setup_model_parallelism(model_fn: Callable,
                           mesh: Mesh,
                           in_specs: Any,
                           out_specs: Any,
                           static_argnums: Optional[Tuple[int, ...]] = None) -> Callable:
    """Setup model parallel function with pjit.
    
    Args:
        model_fn: Model function to parallelize
        mesh: Device mesh
        in_specs: Input partition specifications
        out_specs: Output partition specifications
        static_argnums: Static arguments
        
    Returns:
        Model parallel function
    """
    with mesh:
        return pjit(
            model_fn,
            in_specs=in_specs,
            out_specs=out_specs,
            static_argnums=static_argnums
        )


def pjit_train_step(forward_fn: Callable,
                   loss_fn: Callable,
                   optimizer_update: Callable,
                   mesh: Mesh,
                   param_specs: Any,
                   data_specs: Any) -> Callable:
    """Create pjit training step with model parallelism.
    
    Args:
        forward_fn: Model forward function
        loss_fn: Loss function
        optimizer_update: Optimizer update function
        mesh: Device mesh
        param_specs: Parameter partition specs
        data_specs: Data partition specs
        
    Returns:
        Pjit training step function
    """
    def train_step(params, optimizer_state, batch):
        def loss_and_metrics(p):
            predictions = forward_fn(p, batch['inputs'], training=True)
            loss = loss_fn(predictions, batch['labels'])
            
            accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch['labels'])
            return loss, {'loss': loss, 'accuracy': accuracy}
        
        # Compute gradients
        (loss_value, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
        
        # Update parameters
        new_optimizer_state = optimizer_update(optimizer_state, grads)
        
        return new_optimizer_state, metrics
    
    with mesh:
        return pjit(
            train_step,
            in_specs=(param_specs, param_specs, data_specs),
            out_specs=(param_specs, PartitionSpec())
        )


def model_parallel_forward(forward_fn: Callable,
                          params: Any,
                          inputs: jnp.ndarray,
                          mesh: Mesh,
                          param_specs: Any,
                          input_spec: PartitionSpec,
                          output_spec: PartitionSpec) -> jnp.ndarray:
    """Model parallel forward pass.
    
    Args:
        forward_fn: Forward function
        params: Model parameters
        inputs: Input data
        mesh: Device mesh
        param_specs: Parameter partition specs
        input_spec: Input partition spec
        output_spec: Output partition spec
        
    Returns:
        Forward pass output
    """
    parallel_forward = setup_model_parallelism(
        forward_fn, mesh, (param_specs, input_spec), output_spec
    )
    
    return parallel_forward(params, inputs)


def create_data_model_parallel_step(forward_fn: Callable,
                                   loss_fn: Callable,
                                   optimizer_update: Callable,
                                   mesh: Mesh,
                                   data_axis: str = 'data',
                                   model_axis: str = 'model') -> Callable:
    """Create training step with both data and model parallelism.
    
    Args:
        forward_fn: Model forward function
        loss_fn: Loss function
        optimizer_update: Optimizer update function
        mesh: Device mesh with both data and model axes
        data_axis: Name of data parallel axis
        model_axis: Name of model parallel axis
        
    Returns:
        Combined data/model parallel training step
    """
    def train_step(params, optimizer_state, batch):
        def loss_and_metrics(p):
            predictions = forward_fn(p, batch['inputs'], training=True)
            loss = loss_fn(predictions, batch['labels'])
            
            accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch['labels'])
            return loss, {'loss': loss, 'accuracy': accuracy}
        
        # Compute gradients
        (loss_value, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
        
        # Reduce gradients across data parallel dimension
        grads = jax.lax.pmean(grads, axis_name=data_axis)
        metrics = jax.lax.pmean(metrics, axis_name=data_axis)
        
        # Update parameters
        new_optimizer_state = optimizer_update(optimizer_state, grads)
        
        return new_optimizer_state, metrics
    
    # Define partition specs
    # Batch dimension sharded across data axis
    batch_spec = PartitionSpec(data_axis, None)
    
    # Parameters can be sharded across model axis (depends on specific model)
    # This is a simplified example - real partition specs depend on model architecture
    param_spec = PartitionSpec(None, model_axis)  # Shard weights across model axis
    
    with mesh:
        return pjit(
            train_step,
            in_specs=(param_spec, param_spec, {'inputs': batch_spec, 'labels': batch_spec}),
            out_specs=(param_spec, PartitionSpec())
        )


def shard_large_layer(weights: jnp.ndarray,
                     bias: jnp.ndarray,
                     mesh: Mesh,
                     shard_axis: int = 1) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Shard large layer weights across devices.
    
    Args:
        weights: Weight matrix
        bias: Bias vector
        mesh: Device mesh
        shard_axis: Axis to shard along (0 for input sharding, 1 for output sharding)
        
    Returns:
        (sharded_weights, sharded_bias) tuple
    """
    if shard_axis == 0:
        # Input sharding: shard input dimension
        weight_spec = PartitionSpec('model', None)
        bias_spec = PartitionSpec()
    elif shard_axis == 1:
        # Output sharding: shard output dimension
        weight_spec = PartitionSpec(None, 'model')
        bias_spec = PartitionSpec('model')
    else:
        raise ValueError(f"Invalid shard_axis: {shard_axis}")
    
    with mesh:
        sharded_weights = jax.device_put(
            weights, jax.sharding.NamedSharding(mesh, weight_spec)
        )
        sharded_bias = jax.device_put(
            bias, jax.sharding.NamedSharding(mesh, bias_spec)
        )
    
    return sharded_weights, sharded_bias


def create_transformer_partition_specs(num_heads: int,
                                      d_model: int,
                                      mesh: Mesh,
                                      model_axis: str = 'model') -> Dict[str, PartitionSpec]:
    """Create partition specs for transformer model.
    
    Args:
        num_heads: Number of attention heads
        d_model: Model dimension
        mesh: Device mesh
        model_axis: Model parallel axis name
        
    Returns:
        Dictionary of partition specifications
    """
    # Attention weights can be sharded across heads or hidden dimension
    attention_spec = PartitionSpec(None, model_axis)
    
    # Feed-forward weights sharded across hidden dimension
    ff_spec = PartitionSpec(None, model_axis)
    
    # Layer norm parameters typically not sharded
    ln_spec = PartitionSpec()
    
    return {
        'attention/W_q': attention_spec,
        'attention/W_k': attention_spec,
        'attention/W_v': attention_spec,
        'attention/W_o': attention_spec,
        'feed_forward/W1': ff_spec,
        'feed_forward/W2': PartitionSpec(model_axis, None),  # Different for second FF layer
        'layer_norm/scale': ln_spec,
        'layer_norm/bias': ln_spec,
    }


def gather_from_model_parallel(x: jnp.ndarray,
                              axis: int,
                              axis_name: str = 'model') -> jnp.ndarray:
    """Gather tensor from model parallel devices.
    
    Args:
        x: Tensor to gather
        axis: Axis to gather along
        axis_name: Model parallel axis name
        
    Returns:
        Gathered tensor
    """
    return jax.lax.all_gather(x, axis_name=axis_name, axis=axis)


def scatter_to_model_parallel(x: jnp.ndarray,
                             axis: int,
                             axis_name: str = 'model') -> jnp.ndarray:
    """Scatter tensor to model parallel devices.
    
    Args:
        x: Tensor to scatter
        axis: Axis to scatter along
        axis_name: Model parallel axis name
        
    Returns:
        Scattered tensor
    """
    num_partitions = jax.lax.psum(1, axis_name=axis_name)
    return jax.lax.dynamic_slice_in_dim(
        x, jax.lax.axis_index(axis_name) * (x.shape[axis] // num_partitions),
        x.shape[axis] // num_partitions, axis=axis
    )


def check_sharding_compatibility(array: jnp.ndarray,
                                partition_spec: PartitionSpec,
                                mesh: Mesh) -> bool:
    """Check if array shape is compatible with partition spec.
    
    Args:
        array: Array to check
        partition_spec: Partition specification
        mesh: Device mesh
        
    Returns:
        True if compatible, False otherwise
    """
    if len(partition_spec) > len(array.shape):
        return False
    
    for i, (dim_size, shard_axis) in enumerate(zip(array.shape, partition_spec)):
        if shard_axis is not None:
            mesh_size = mesh.shape[mesh.axis_names.index(shard_axis)]
            if dim_size % mesh_size != 0:
                return False
    
    return True


def estimate_memory_per_device(params: Any,
                              mesh: Mesh,
                              partition_specs: Dict[str, PartitionSpec]) -> Dict[str, float]:
    """Estimate memory usage per device with model parallelism.
    
    Args:
        params: Model parameters
        mesh: Device mesh
        partition_specs: Partition specifications
        
    Returns:
        Memory estimates per device in MB
    """
    total_params = 0
    sharded_params = 0
    
    def count_params(path, param):
        nonlocal total_params, sharded_params
        param_name = '.'.join(str(k.key) for k in path)
        param_size = param.nbytes
        total_params += param_size
        
        if param_name in partition_specs:
            spec = partition_specs[param_name]
            # Estimate sharding factor (simplified)
            shard_factor = 1
            for axis in spec:
                if axis is not None:
                    shard_factor *= mesh.shape[mesh.axis_names.index(axis)]
            sharded_params += param_size // shard_factor
        else:
            sharded_params += param_size
    
    jax.tree_util.tree_map_with_path(count_params, params)
    
    return {
        'total_params_mb': total_params / (1024 * 1024),
        'params_per_device_mb': sharded_params / (1024 * 1024),
        'memory_reduction_factor': total_params / sharded_params if sharded_params > 0 else 1.0
    }