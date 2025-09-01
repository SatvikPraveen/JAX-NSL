# File location: jax-nsl/src/transforms/vmap_utils.py

"""
Vectorization and batching patterns using vmap.

This module provides utilities for efficiently batching operations
using JAX's vmap transformation.
"""

import jax
import jax.numpy as jnp
from jax import vmap
from typing import Callable, Any, Optional, Union, Tuple, Dict
import functools


def batch_apply(fun: Callable,
               in_axes: Any = 0,
               out_axes: Any = 0,
               axis_name: Optional[str] = None,
               axis_size: Optional[int] = None,
               spmd_axis_name: Optional[Union[str, Tuple[str, ...]]] = None) -> Callable:
    """Enhanced vmap wrapper with better defaults and error handling.
    
    Args:
        fun: Function to vectorize
        in_axes: Input axes specification
        out_axes: Output axes specification  
        axis_name: Name for the mapped axis
        axis_size: Size of the mapped axis (for validation)
        spmd_axis_name: SPMD axis name for parallel execution
        
    Returns:
        Vectorized function
    """
    vmapped_fun = vmap(
        fun,
        in_axes=in_axes,
        out_axes=out_axes,
        axis_name=axis_name,
        axis_size=axis_size,
        spmd_axis_name=spmd_axis_name
    )
    
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        try:
            return vmapped_fun(*args, **kwargs)
        except Exception as e:
            print(f"Vectorization failed for {fun.__name__}: {e}")
            # Fallback to manual batching
            return manual_batch_apply(fun, args, kwargs, in_axes)
    
    return wrapper


def manual_batch_apply(fun: Callable, 
                      args: Tuple, 
                      kwargs: Dict, 
                      in_axes: Any) -> Any:
    """Manual batching fallback when vmap fails.
    
    Args:
        fun: Function to apply
        args: Function arguments
        kwargs: Function keyword arguments
        in_axes: Input axes specification
        
    Returns:
        Batched results
    """
    # Simple case: assume first argument is batched along axis 0
    if isinstance(in_axes, int) and in_axes == 0 and len(args) > 0:
        batch_size = args[0].shape[0]
        results = []
        
        for i in range(batch_size):
            batch_args = tuple(arg[i] if j == 0 else arg 
                             for j, arg in enumerate(args))
            result = fun(*batch_args, **kwargs)
            results.append(result)
        
        return jnp.stack(results)
    
    # More complex cases would need more sophisticated handling
    raise NotImplementedError("Complex in_axes not supported in manual batching")


def vectorize_function(in_axes: Any = 0, out_axes: Any = 0) -> Callable:
    """Decorator for vectorizing functions.
    
    Args:
        in_axes: Input axes specification
        out_axes: Output axes specification
        
    Returns:
        Vectorization decorator
    """
    def decorator(fun: Callable) -> Callable:
        return batch_apply(fun, in_axes=in_axes, out_axes=out_axes)
    return decorator


def batch_outer_product(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute batched outer products efficiently.
    
    Args:
        x: Array with shape (batch_size, m)
        y: Array with shape (batch_size, n) 
        
    Returns:
        Batched outer products with shape (batch_size, m, n)
    """
    return vmap(jnp.outer)(x, y)


def parallel_map(fun: Callable, 
                xs: jnp.ndarray,
                chunk_size: Optional[int] = None) -> jnp.ndarray:
    """Parallel map over array elements with chunking.
    
    Args:
        fun: Function to apply to each element
        xs: Input array
        chunk_size: Optional chunk size for processing
        
    Returns:
        Array with function applied to each element
    """
    if chunk_size is None:
        return vmap(fun)(xs)
    
    # Process in chunks to manage memory
    num_chunks = (len(xs) + chunk_size - 1) // chunk_size
    chunks = jnp.array_split(xs, num_chunks)
    
    results = []
    for chunk in chunks:
        chunk_results = vmap(fun)(chunk)
        results.append(chunk_results)
    
    return jnp.concatenate(results, axis=0)


def batch_matrix_ops(matrices: jnp.ndarray, 
                    operation: str = 'inv',
                    **kwargs) -> jnp.ndarray:
    """Apply matrix operations to batches of matrices.
    
    Args:
        matrices: Batch of matrices with shape (batch_size, n, n)
        operation: Matrix operation ('inv', 'det', 'eig', 'svd', etc.)
        **kwargs: Additional arguments for the operation
        
    Returns:
        Results of applying operation to each matrix
    """
    if operation == 'inv':
        return vmap(jnp.linalg.inv)(matrices)
    elif operation == 'det':
        return vmap(jnp.linalg.det)(matrices)
    elif operation == 'eig':
        return vmap(jnp.linalg.eig)(matrices)
    elif operation == 'eigvals':
        return vmap(jnp.linalg.eigvals)(matrices)
    elif operation == 'svd':
        full_matrices = kwargs.get('full_matrices', True)
        return vmap(lambda m: jnp.linalg.svd(m, full_matrices=full_matrices))(matrices)
    elif operation == 'cholesky':
        return vmap(jnp.linalg.cholesky)(matrices)
    elif operation == 'qr':
        return vmap(jnp.linalg.qr)(matrices)
    else:
        raise ValueError(f"Unsupported matrix operation: {operation}")


def batch_solve(A: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Solve batched linear systems Ax = b.
    
    Args:
        A: Batch of coefficient matrices (batch_size, n, n)
        b: Batch of right-hand sides (batch_size, n) or (batch_size, n, k)
        
    Returns:
        Solutions x for each system
    """
    return vmap(jnp.linalg.solve)(A, b)


def batch_apply_along_axis(fun: Callable, 
                          axis: int,
                          arr: jnp.ndarray,
                          keepdims: bool = False) -> jnp.ndarray:
    """Apply function along specific axis using vmap.
    
    Args:
        fun: Function to apply
        axis: Axis along which to apply function
        arr: Input array
        keepdims: Whether to keep reduced dimensions
        
    Returns:
        Result of applying function along axis
    """
    # Move the target axis to position 0
    arr_moved = jnp.moveaxis(arr, axis, 0)
    
    # Apply function to each slice
    result = vmap(fun)(arr_moved)
    
    if keepdims:
        # Add back the reduced dimension
        result = jnp.expand_dims(result, axis=0)
        # Move back to original axis position
        result = jnp.moveaxis(result, 0, axis)
    
    return result


def nested_vmap(fun: Callable, 
               in_axes_list: list,
               out_axes_list: list) -> Callable:
    """Apply nested vmaps for multi-dimensional batching.
    
    Args:
        fun: Function to vectorize
        in_axes_list: List of input axes for each vmap level
        out_axes_list: List of output axes for each vmap level
        
    Returns:
        Nested-vectorized function
    """
    result_fun = fun
    
    # Apply vmaps from innermost to outermost
    for in_axes, out_axes in zip(reversed(in_axes_list), reversed(out_axes_list)):
        result_fun = vmap(result_fun, in_axes=in_axes, out_axes=out_axes)
    
    return result_fun


def batch_gradient(fun: Callable, 
                  argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """Compute gradients for batched inputs.
    
    Args:
        fun: Function to differentiate (should handle single examples)
        argnums: Arguments to differentiate with respect to
        
    Returns:
        Function that computes batched gradients
    """
    grad_fun = jax.grad(fun, argnums=argnums)
    return vmap(grad_fun)


def batch_jacobian(fun: Callable,
                  argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """Compute Jacobians for batched inputs.
    
    Args:
        fun: Function to differentiate (should handle single examples)
        argnums: Arguments to differentiate with respect to
        
    Returns:
        Function that computes batched Jacobians
    """
    jac_fun = jax.jacobian(fun, argnums=argnums)
    return vmap(jac_fun)


def selective_vmap(fun: Callable,
                  condition_fn: Callable,
                  in_axes: Any = 0,
                  out_axes: Any = 0) -> Callable:
    """Conditionally apply vmap based on input properties.
    
    Args:
        fun: Function to potentially vectorize
        condition_fn: Function that determines whether to use vmap
        in_axes: Input axes for vmap
        out_axes: Output axes for vmap
        
    Returns:
        Function that conditionally uses vmap
    """
    vmapped_fun = vmap(fun, in_axes=in_axes, out_axes=out_axes)
    
    @functools.wraps(fun)
    def wrapper(*args, **kwargs):
        if condition_fn(*args, **kwargs):
            return vmapped_fun(*args, **kwargs)
        else:
            return fun(*args, **kwargs)
    
    return wrapper


def vmap_with_signature(signature: str) -> Callable:
    """Create vmap with Einstein-like signature specification.
    
    Args:
        signature: Signature like 'bi,bj->bij' for batched operations
        
    Returns:
        vmap decorator with inferred axes
    """
    def decorator(fun: Callable) -> Callable:
        # Parse signature to determine input/output axes
        # This is a simplified implementation
        parts = signature.split('->')
        input_sigs = parts[0].split(',')
        output_sig = parts[1] if len(parts) > 1 else None
        
        # Determine which axes are batched (appear in all inputs/outputs)
        # For now, assume 'b' represents batch dimension at axis 0
        in_axes = []
        for sig in input_sigs:
            in_axes.append(0 if 'b' in sig else None)
        
        out_axes = 0 if output_sig and 'b' in output_sig else None
        
        return vmap(fun, in_axes=tuple(in_axes), out_axes=out_axes)
    
    return decorator