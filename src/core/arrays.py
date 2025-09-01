# File location: jax-nsl/src/core/arrays.py

"""
DeviceArray utilities, dtype handling, and pytree operations.

This module provides essential utilities for working with JAX arrays,
including dtype information, safe casting, and pytree analysis tools.
"""

import jax
import jax.numpy as jnp
from jax import tree_util
from typing import Any, Dict, Tuple, Optional, Union
import numpy as np


def get_dtype_info(dtype: jnp.dtype) -> Dict[str, Any]:
    """Get comprehensive information about a JAX dtype.
    
    Args:
        dtype: JAX/NumPy dtype to analyze
        
    Returns:
        Dictionary containing dtype information including:
        - name, itemsize, kind
        - For float: eps, max, min, precision
        - For int: max, min
    """
    info = {
        'name': dtype.name,
        'itemsize': dtype.itemsize,
        'kind': dtype.kind,
    }
    
    if jnp.issubdtype(dtype, jnp.floating):
        finfo = jnp.finfo(dtype)
        info.update({
            'eps': finfo.eps,
            'max': finfo.max,
            'min': finfo.min,
            'precision': finfo.precision,
            'resolution': finfo.resolution
        })
    elif jnp.issubdtype(dtype, jnp.integer):
        iinfo = jnp.iinfo(dtype)
        info.update({
            'max': iinfo.max,
            'min': iinfo.min
        })
    
    return info


def safe_cast(x: jnp.ndarray, dtype: jnp.dtype, clip: bool = True) -> jnp.ndarray:
    """Safely cast array to target dtype with optional clipping.
    
    Args:
        x: Input array
        dtype: Target dtype
        clip: Whether to clip values to dtype range
        
    Returns:
        Array cast to target dtype
    """
    if x.dtype == dtype:
        return x
    
    if clip and jnp.issubdtype(dtype, jnp.integer):
        iinfo = jnp.iinfo(dtype)
        x = jnp.clip(x, iinfo.min, iinfo.max)
    elif clip and jnp.issubdtype(dtype, jnp.floating):
        finfo = jnp.finfo(dtype)
        x = jnp.clip(x, finfo.min, finfo.max)
    
    return x.astype(dtype)


def tree_size(tree: Any) -> int:
    """Count total number of elements in a pytree.
    
    Args:
        tree: PyTree to analyze
        
    Returns:
        Total number of array elements
    """
    leaves = tree_util.tree_leaves(tree)
    return sum(leaf.size for leaf in leaves if hasattr(leaf, 'size'))


def tree_bytes(tree: Any) -> int:
    """Calculate total memory usage of a pytree in bytes.
    
    Args:
        tree: PyTree to analyze
        
    Returns:
        Total memory usage in bytes
    """
    leaves = tree_util.tree_leaves(tree)
    return sum(leaf.nbytes for leaf in leaves if hasattr(leaf, 'nbytes'))


def tree_summary(tree: Any, name: str = "Tree") -> Dict[str, Any]:
    """Generate comprehensive summary of a pytree.
    
    Args:
        tree: PyTree to analyze
        name: Name for the tree in summary
        
    Returns:
        Dictionary with size, memory, shapes, and dtype info
    """
    leaves = tree_util.tree_leaves(tree)
    arrays = [leaf for leaf in leaves if hasattr(leaf, 'shape')]
    
    if not arrays:
        return {'name': name, 'empty': True}
    
    shapes = [arr.shape for arr in arrays]
    dtypes = [arr.dtype for arr in arrays]
    devices = [arr.device() if hasattr(arr, 'device') else 'cpu' for arr in arrays]
    
    return {
        'name': name,
        'num_arrays': len(arrays),
        'total_elements': sum(arr.size for arr in arrays),
        'total_bytes': sum(arr.nbytes for arr in arrays),
        'shapes': shapes,
        'dtypes': list(set(str(dt) for dt in dtypes)),
        'devices': list(set(str(dev) for dev in devices)),
        'tree_structure': tree_util.tree_structure(tree)
    }


def tree_map_with_path(f, tree: Any) -> Any:
    """Map function over pytree leaves with access to their paths.
    
    Args:
        f: Function that takes (path, leaf) and returns new leaf
        tree: Input pytree
        
    Returns:
        New pytree with function applied to each leaf
    """
    def _map_with_path(path, leaf):
        return f(path, leaf)
    
    return tree_util.tree_map_with_path(_map_with_path, tree)


# Utility functions for common array patterns
def create_mesh_grid(shape: Tuple[int, ...], 
                     bounds: Optional[Tuple[Tuple[float, float], ...]] = None) -> jnp.ndarray:
    """Create coordinate mesh grid for given shape and bounds.
    
    Args:
        shape: Grid dimensions
        bounds: Optional bounds for each dimension as (min, max) tuples
        
    Returns:
        Coordinate arrays for mesh grid
    """
    if bounds is None:
        bounds = [(0., float(s-1)) for s in shape]
    
    coords = []
    for i, (s, (min_val, max_val)) in enumerate(zip(shape, bounds)):
        coords.append(jnp.linspace(min_val, max_val, s))
    
    return jnp.meshgrid(*coords, indexing='ij')


def sliding_window(x: jnp.ndarray, window_size: int, stride: int = 1) -> jnp.ndarray:
    """Create sliding windows over the last axis of an array.
    
    Args:
        x: Input array
        window_size: Size of sliding window
        stride: Stride between windows
        
    Returns:
        Array with sliding windows as new dimension
    """
    if window_size > x.shape[-1]:
        raise ValueError(f"Window size {window_size} larger than array size {x.shape[-1]}")
    
    # Number of windows
    num_windows = (x.shape[-1] - window_size) // stride + 1
    
    # Create indices for windows
    indices = jnp.arange(window_size)[None, :] + jnp.arange(num_windows)[:, None] * stride
    
    # Index the array
    return x[..., indices]


def pad_to_shape(x: jnp.ndarray, target_shape: Tuple[int, ...], 
                 mode: str = 'constant', constant_values: float = 0.0) -> jnp.ndarray:
    """Pad array to target shape.
    
    Args:
        x: Input array
        target_shape: Desired shape
        mode: Padding mode
        constant_values: Value for constant padding
        
    Returns:
        Padded array
    """
    if len(x.shape) != len(target_shape):
        raise ValueError(f"Shape mismatch: {x.shape} vs {target_shape}")
    
    pad_widths = []
    for current, target in zip(x.shape, target_shape):
        if current > target:
            raise ValueError(f"Current dimension {current} larger than target {target}")
        pad_width = target - current
        pad_widths.append((0, pad_width))
    
    return jnp.pad(x, pad_widths, mode=mode, constant_values=constant_values)