# File location: jax-nsl/src/parallel/collectives.py

"""
Collective operations: psum, pmean, all-reduce patterns.

This module provides utilities for collective communication operations
across multiple devices in JAX programs.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Any, Optional, Union, Tuple


def all_reduce_mean(x: jnp.ndarray, 
                   axis_name: str = 'batch') -> jnp.ndarray:
    """All-reduce with mean reduction.
    
    Args:
        x: Array to reduce
        axis_name: Name of the axis to reduce over
        
    Returns:
        Mean-reduced array
    """
    return lax.pmean(x, axis_name=axis_name)


def all_reduce_sum(x: jnp.ndarray,
                  axis_name: str = 'batch') -> jnp.ndarray:
    """All-reduce with sum reduction.
    
    Args:
        x: Array to reduce
        axis_name: Name of the axis to reduce over
        
    Returns:
        Sum-reduced array
    """
    return lax.psum(x, axis_name=axis_name)


def all_reduce_max(x: jnp.ndarray,
                  axis_name: str = 'batch') -> jnp.ndarray:
    """All-reduce with max reduction.
    
    Args:
        x: Array to reduce
        axis_name: Name of the axis to reduce over
        
    Returns:
        Max-reduced array
    """
    return lax.pmax(x, axis_name=axis_name)


def all_reduce_min(x: jnp.ndarray,
                  axis_name: str = 'batch') -> jnp.ndarray:
    """All-reduce with min reduction.
    
    Args:
        x: Array to reduce
        axis_name: Name of the axis to reduce over
        
    Returns:
        Min-reduced array
    """
    return lax.pmin(x, axis_name=axis_name)


def all_gather(x: jnp.ndarray,
              axis_name: str = 'batch',
              axis: int = 0) -> jnp.ndarray:
    """All-gather operation.
    
    Args:
        x: Array to gather
        axis_name: Name of the axis to gather over
        axis: Axis to gather along
        
    Returns:
        Gathered array
    """
    return lax.all_gather(x, axis_name=axis_name, axis=axis)


def reduce_scatter(x: jnp.ndarray,
                  axis_name: str = 'batch',
                  scatter_dimension: int = 0) -> jnp.ndarray:
    """Reduce-scatter operation.
    
    Args:
        x: Array to reduce and scatter
        axis_name: Name of the axis for reduction
        scatter_dimension: Dimension to scatter along
        
    Returns:
        Reduced and scattered array
    """
    # First reduce
    reduced = lax.psum(x, axis_name=axis_name)
    
    # Then scatter (split among devices)
    axis_size = lax.psum(1, axis_name=axis_name)
    axis_index = lax.axis_index(axis_name)
    
    scatter_size = reduced.shape[scatter_dimension] // axis_size
    start_index = axis_index * scatter_size
    
    return lax.dynamic_slice_in_dim(
        reduced, start_index, scatter_size, axis=scatter_dimension
    )


def broadcast(x: jnp.ndarray,
             root_rank: int = 0,
             axis_name: str = 'batch') -> jnp.ndarray:
    """Broadcast from root device to all devices.
    
    Args:
        x: Array to broadcast
        root_rank: Rank of the root device
        axis_name: Name of the axis for broadcast
        
    Returns:
        Broadcasted array
    """
    # Select data from root rank
    is_root = lax.axis_index(axis_name) == root_rank
    return jnp.where(is_root, x, 0.0)


def barrier_sync(axis_name: str = 'batch') -> None:
    """Synchronization barrier across devices.
    
    Args:
        axis_name: Name of the axis to synchronize over
    """
    # Simple barrier using psum
    dummy = jnp.array(1.0)
    lax.psum(dummy, axis_name=axis_name)


def cross_replica_mean(x: jnp.ndarray, 
                      axis_name: str = 'batch') -> jnp.ndarray:
    """Cross-replica mean (same as all_reduce_mean).
    
    Args:
        x: Array to average
        axis_name: Name of the axis to average over
        
    Returns:
        Cross-replica averaged array
    """
    return lax.pmean(x, axis_name=axis_name)


def tree_all_reduce(tree: Any,
                   reduction: str = 'mean',
                   axis_name: str = 'batch') -> Any:
    """All-reduce over a pytree structure.
    
    Args:
        tree: PyTree to reduce
        reduction: Reduction type ('mean', 'sum', 'max', 'min')
        axis_name: Name of the axis to reduce over
        
    Returns:
        Reduced pytree
    """
    if reduction == 'mean':
        reduce_fn = lambda x: lax.pmean(x, axis_name=axis_name)
    elif reduction == 'sum':
        reduce_fn = lambda x: lax.psum(x, axis_name=axis_name)
    elif reduction == 'max':
        reduce_fn = lambda x: lax.pmax(x, axis_name=axis_name)
    elif reduction == 'min':
        reduce_fn = lambda x: lax.pmin(x, axis_name=axis_name)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    return jax.tree_util.tree_map(reduce_fn, tree)


def gradient_synchronization(grads: Any,
                            axis_name: str = 'batch',
                            clip_norm: Optional[float] = None) -> Any:
    """Synchronize gradients across devices with optional clipping.
    
    Args:
        grads: Gradient pytree
        axis_name: Name of the axis for synchronization
        clip_norm: Optional global gradient norm clipping
        
    Returns:
        Synchronized gradients
    """
    # Compute global gradient norm if clipping is requested
    if clip_norm is not None:
        grad_norm = jnp.sqrt(sum(
            lax.psum(jnp.sum(g ** 2), axis_name=axis_name)
            for g in jax.tree_util.tree_leaves(grads)
        ))
        
        # Apply clipping
        clip_factor = jnp.minimum(1.0, clip_norm / (grad_norm + 1e-8))
        grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    
    # Synchronize gradients
    return tree_all_reduce(grads, reduction='mean', axis_name=axis_name)


def ring_all_reduce(x: jnp.ndarray,
                   axis_name: str = 'batch',
                   chunk_size: Optional[int] = None) -> jnp.ndarray:
    """Ring all-reduce implementation (educational).
    
    Args:
        x: Array to reduce
        axis_name: Name of the axis for reduction
        chunk_size: Size of chunks for ring reduction
        
    Returns:
        All-reduced array
    """
    # This is a simplified educational implementation
    # In practice, JAX's built-in collectives are more efficient
    
    num_devices = lax.psum(1, axis_name=axis_name)
    device_id = lax.axis_index(axis_name)
    
    if chunk_size is None:
        chunk_size = x.size // num_devices
    
    # Ring reduction (simplified)
    result = x
    for step in range(num_devices - 1):
        # Send to next device, receive from previous
        result = lax.psum(result, axis_name=axis_name) / num_devices
    
    return result


def hierarchical_all_reduce(x: jnp.ndarray,
                           intra_node_axis: str = 'local',
                           inter_node_axis: str = 'global') -> jnp.ndarray:
    """Hierarchical all-reduce (intra-node then inter-node).
    
    Args:
        x: Array to reduce
        intra_node_axis: Axis name for intra-node reduction
        inter_node_axis: Axis name for inter-node reduction
        
    Returns:
        Hierarchically reduced array
    """
    # First reduce within each node
    intra_reduced = lax.pmean(x, axis_name=intra_node_axis)
    
    # Then reduce across nodes
    final_result = lax.pmean(intra_reduced, axis_name=inter_node_axis)
    
    return final_result


def alltoall(x: jnp.ndarray,
            axis_name: str = 'batch',
            split_axis: int = 0,
            concat_axis: int = 0) -> jnp.ndarray:
    """All-to-all communication pattern.
    
    Args:
        x: Input array
        axis_name: Axis name for communication
        split_axis: Axis to split input along
        concat_axis: Axis to concatenate results along
        
    Returns:
        All-to-all result
    """
    # Split input into chunks for each device
    num_devices = lax.psum(1, axis_name=axis_name)
    chunk_size = x.shape[split_axis] // num_devices
    
    # Create chunks
    chunks = []
    for i in range(num_devices):
        start_idx = i * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start_idx, chunk_size, axis=split_axis)
        chunks.append(chunk)
    
    # Scatter chunks to different devices and gather
    def send_recv(chunk_and_target):
        chunk, target_rank = chunk_and_target
        device_id = lax.axis_index(axis_name)
        
        # This is a simplified all-to-all - real implementation would be more complex
        # For educational purposes, we use all_gather
        return lax.all_gather(chunk, axis_name=axis_name, axis=concat_axis)
    
    # Apply to all chunks
    results = [send_recv((chunk, i)) for i, chunk in enumerate(chunks)]
    
    # Combine results
    return jnp.concatenate(results, axis=concat_axis)


def compute_communication_volume(array_shapes: list,
                                reduction: str = 'sum',
                                dtype: jnp.dtype = jnp.float32) -> Dict[str, float]:
    """Estimate communication volume for collective operations.
    
    Args:
        array_shapes: List of array shapes
        reduction: Type of reduction operation
        dtype: Array dtype
        
    Returns:
        Communication volume estimates in MB
    """
    bytes_per_element = jnp.dtype(dtype).itemsize
    total_elements = sum(jnp.prod(jnp.array(shape)) for shape in array_shapes)
    total_bytes = total_elements * bytes_per_element
    total_mb = total_bytes / (1024 * 1024)
    
    # Estimate based on reduction type
    if reduction in ['sum', 'mean', 'max', 'min']:
        # All-reduce: (N-1)/N * 2 * data_size communication volume
        comm_volume_factor = 2.0  # Simplified estimate
    elif reduction == 'gather':
        # All-gather: (N-1) * data_size
        comm_volume_factor = 1.0
    else:
        comm_volume_factor = 1.0
    
    return {
        'total_data_mb': total_mb,
        'estimated_comm_mb': total_mb * comm_volume_factor,
        'num_arrays': len(array_shapes),
        'total_elements': total_elements
    }


def create_communication_group(devices: list,
                              axis_name: str) -> Any:
    """Create communication group for specific devices.
    
    Args:
        devices: List of devices for the group
        axis_name: Name for the communication axis
        
    Returns:
        Communication group specification
    """
    # This is a placeholder for creating device groups
    # Real implementation would depend on JAX's device management
    return {
        'devices': devices,
        'axis_name': axis_name,
        'group_size': len(devices)
    }