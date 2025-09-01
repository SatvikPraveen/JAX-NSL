# File location: jax-nsl/src/parallel/pmap_utils.py

"""
Data parallelism utilities using pmap.

This module provides utilities for distributing computation across
multiple devices using JAX's pmap transformation.
"""

import jax
import jax.numpy as jnp
from jax import pmap, lax
from typing import Any, Callable, Dict, Tuple
import functools


def replicate_params(params: Any, num_devices: int = None) -> Any:
    """Replicate parameters across devices.
    
    Args:
        params: Parameters to replicate
        num_devices: Number of devices (defaults to available devices)
        
    Returns:
        Replicated parameters
    """
    if num_devices is None:
        num_devices = jax.local_device_count()
    
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
        params
    )


def unreplicate_params(replicated_params: Any) -> Any:
    """Extract parameters from replicated form.
    
    Args:
        replicated_params: Replicated parameters
        
    Returns:
        Unreplicated parameters (from device 0)
    """
    return jax.tree_util.tree_map(lambda x: x[0], replicated_params)


def shard_batch(batch: Dict[str, jnp.ndarray], 
               num_devices: int = None) -> Dict[str, jnp.ndarray]:
    """Shard batch across devices.
    
    Args:
        batch: Batch dictionary
        num_devices: Number of devices
        
    Returns:
        Sharded batch
    """
    if num_devices is None:
        num_devices = jax.local_device_count()
    
    batch_size = next(iter(batch.values())).shape[0]
    per_device_batch_size = batch_size // num_devices
    
    if batch_size % num_devices != 0:
        raise ValueError(f"Batch size {batch_size} not divisible by {num_devices} devices")
    
    sharded_batch = {}
    for key, value in batch.items():
        # Reshape to (num_devices, per_device_batch_size, ...)
        sharded_value = value.reshape((num_devices, per_device_batch_size) + value.shape[1:])
        sharded_batch[key] = sharded_value
    
    return sharded_batch


@functools.partial(pmap, axis_name='batch')
def data_parallel_step(params: Any,
                      batch: Dict[str, jnp.ndarray],
                      forward_fn: Callable,
                      loss_fn: Callable,
                      optimizer_update: Callable,
                      optimizer_state: Any) -> Tuple[Any, Any, Dict[str, float]]:
    """Data parallel training step.
    
    Args:
        params: Model parameters (replicated)
        batch: Batch data (sharded)
        forward_fn: Forward function
        loss_fn: Loss function
        optimizer_update: Optimizer update function
        optimizer_state: Optimizer state (replicated)
        
    Returns:
        (new_params, new_optimizer_state, metrics) tuple
    """
    def loss_and_metrics(p):
        predictions = forward_fn(p, batch['inputs'], training=True)
        loss = loss_fn(predictions, batch['labels'])
        
        accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch['labels'])
        return loss, {'loss': loss, 'accuracy': accuracy}
    
    # Compute gradients
    (loss_value, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
    
    # Average gradients across devices
    grads = lax.pmean(grads, axis_name='batch')
    metrics = lax.pmean(metrics, axis_name='batch')
    
    # Update parameters
    new_optimizer_state = optimizer_update(optimizer_state, grads)
    
    return new_optimizer_state.params, new_optimizer_state, metrics


@functools.partial(pmap, axis_name='batch')
def parallel_eval_step(params: Any,
                      batch: Dict[str, jnp.ndarray],
                      forward_fn: Callable,
                      loss_fn: Callable) -> Dict[str, float]:
    """Parallel evaluation step.
    
    Args:
        params: Model parameters (replicated)
        batch: Batch data (sharded)
        forward_fn: Forward function
        loss_fn: Loss function
        
    Returns:
        Averaged metrics across devices
    """
    predictions = forward_fn(params, batch['inputs'], training=False)
    loss = loss_fn(predictions, batch['labels'])
    accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch['labels'])
    
    metrics = {'loss': loss, 'accuracy': accuracy}
    return lax.pmean(metrics, axis_name='batch')


def create_pmap_train_step(forward_fn: Callable,
                          loss_fn: Callable,
                          optimizer_update: Callable) -> Callable:
    """Create pmapped training step function.
    
    Args:
        forward_fn: Model forward function
        loss_fn: Loss function
        optimizer_update: Optimizer update function
        
    Returns:
        Pmapped training step function
    """
    @functools.partial(pmap, axis_name='batch')
    def train_step(params, optimizer_state, batch):
        def loss_and_metrics(p):
            predictions = forward_fn(p, batch['inputs'], training=True)
            loss = loss_fn(predictions, batch['labels'])
            
            accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch['labels'])
            return loss, {'loss': loss, 'accuracy': accuracy}
        
        # Compute gradients
        (loss_value, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
        
        # Synchronize gradients across devices
        grads = lax.pmean(grads, axis_name='batch')
        metrics = lax.pmean(metrics, axis_name='batch')
        
        # Update parameters
        new_optimizer_state = optimizer_update(optimizer_state, grads)
        
        return new_optimizer_state, metrics
    
    return train_step


def parallel_train_epoch(params: Any,
                        optimizer_state: Any,
                        train_loader: Any,
                        train_step_fn: Callable,
                        num_devices: int = None) -> Tuple[Any, Any, Dict[str, float]]:
    """Train for one epoch with data parallelism.
    
    Args:
        params: Model parameters
        optimizer_state: Optimizer state
        train_loader: Training data loader
        train_step_fn: Pmapped training step function
        num_devices: Number of devices
        
    Returns:
        (final_params, final_optimizer_state, avg_metrics) tuple
    """
    if num_devices is None:
        num_devices = jax.local_device_count()
    
    # Replicate initial state
    replicated_params = replicate_params(params, num_devices)
    replicated_opt_state = replicate_params(optimizer_state, num_devices)
    
    epoch_metrics = []
    
    for batch in train_loader:
        # Shard batch across devices
        sharded_batch = shard_batch(batch, num_devices)
        
        # Training step
        replicated_opt_state, batch_metrics = train_step_fn(
            replicated_params, replicated_opt_state, sharded_batch
        )
        
        # Extract metrics from device 0
        epoch_metrics.append(unreplicate_params(batch_metrics))
        
        # Update replicated params
        replicated_params = replicated_opt_state.params
    
    # Average metrics across batches
    avg_metrics = {}
    if epoch_metrics:
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = jnp.mean(jnp.array([m[key] for m in epoch_metrics]))
    
    # Unreplicate final state
    final_params = unreplicate_params(replicated_params)
    final_opt_state = unreplicate_params(replicated_opt_state)
    
    return final_params, final_opt_state, avg_metrics


def create_parallel_inference_fn(forward_fn: Callable) -> Callable:
    """Create parallel inference function.
    
    Args:
        forward_fn: Model forward function
        
    Returns:
        Parallel inference function
    """
    @functools.partial(pmap, axis_name='batch')
    def parallel_inference(params, inputs):
        return forward_fn(params, inputs, training=False)
    
    def inference_fn(params, inputs, num_devices=None):
        if num_devices is None:
            num_devices = jax.local_device_count()
        
        # Replicate params and shard inputs
        replicated_params = replicate_params(params, num_devices)
        
        batch_size = inputs.shape[0]
        if batch_size % num_devices != 0:
            # Pad inputs to make divisible by num_devices
            pad_size = num_devices - (batch_size % num_devices)
            inputs_padded = jnp.concatenate([inputs, inputs[:pad_size]], axis=0)
        else:
            inputs_padded = inputs
            pad_size = 0
        
        # Shard inputs
        per_device_batch = inputs_padded.shape[0] // num_devices
        sharded_inputs = inputs_padded.reshape(
            (num_devices, per_device_batch) + inputs_padded.shape[1:]
        )
        
        # Parallel inference
        sharded_outputs = parallel_inference(replicated_params, sharded_inputs)
        
        # Reshape and remove padding
        outputs = sharded_outputs.reshape((-1,) + sharded_outputs.shape[2:])
        if pad_size > 0:
            outputs = outputs[:-pad_size]
        
        return outputs
    
    return inference_fn


def sync_params_across_devices(params: Any) -> Any:
    """Synchronize parameters across all devices.
    
    Args:
        params: Parameters to synchronize
        
    Returns:
        Synchronized parameters
    """
    # This is typically done automatically in pmap, but can be useful
    # for manual synchronization
    @functools.partial(pmap, axis_name='devices')
    def sync_fn(p):
        return lax.pmean(p, axis_name='devices')
    
    num_devices = jax.local_device_count()
    replicated_params = replicate_params(params, num_devices)
    synced_params = sync_fn(replicated_params)
    
    return unreplicate_params(synced_params)


def device_get(replicated_data: Any) -> Any:
    """Transfer replicated data to host.
    
    Args:
        replicated_data: Data replicated across devices
        
    Returns:
        Data on host (from device 0)
    """
    return jax.device_get(unreplicate_params(replicated_data))


def estimate_memory_usage(params: Any, batch_size: int, num_devices: int = None) -> Dict[str, float]:
    """Estimate memory usage for parallel training.
    
    Args:
        params: Model parameters
        batch_size: Training batch size
        num_devices: Number of devices
        
    Returns:
        Memory usage estimates in MB
    """
    if num_devices is None:
        num_devices = jax.local_device_count()
    
    # Parameter memory
    param_bytes = sum(p.nbytes for p in jax.tree_util.tree_leaves(params))
    param_mb = param_bytes / (1024 * 1024)
    
    # Replicated parameter memory
    replicated_param_mb = param_mb * num_devices
    
    # Estimate gradient memory (same as parameters)
    gradient_mb = replicated_param_mb
    
    # Estimate optimizer state memory (varies by optimizer, use 2x params as estimate)
    optimizer_mb = replicated_param_mb * 2
    
    # Per-device batch size
    per_device_batch_size = batch_size // num_devices
    
    return {
        'parameters_mb': param_mb,
        'replicated_parameters_mb': replicated_param_mb,
        'gradients_mb': gradient_mb,
        'optimizer_state_mb': optimizer_mb,
        'total_estimated_mb': replicated_param_mb + gradient_mb + optimizer_mb,
        'per_device_batch_size': per_device_batch_size,
        'num_devices': num_devices
    }