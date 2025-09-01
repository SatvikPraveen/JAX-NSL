# File location: jax-nsl/src/utils/tree_utils.py

"""
PyTree manipulation utilities.

This module provides advanced utilities for working with JAX PyTrees,
including path-aware operations and tree analysis tools.
"""

import jax
import jax.numpy as jnp
from jax import tree_util
from typing import Any, Callable, Dict, List, Tuple, Optional, Union
import functools


def tree_flatten_with_path(tree: Any) -> Tuple[List[Tuple[Any, Any]], Any]:
    """Flatten tree while preserving paths to each leaf.
    
    Args:
        tree: PyTree to flatten
        
    Returns:
        ([(path, leaf), ...], tree_def) tuple
    """
    leaves, tree_def = tree_util.tree_flatten(tree)
    paths = tree_util.tree_flatten(tree_util.tree_map(lambda _: None, tree))[1]
    
    # Get paths using tree structure
    def get_paths(tree_structure):
        paths = []
        
        def collect_paths(subtree, current_path=[]):
            if tree_util.tree_leaves(subtree):
                if isinstance(subtree, dict):
                    for key, value in subtree.items():
                        collect_paths(value, current_path + [key])
                elif isinstance(subtree, (list, tuple)):
                    for i, value in enumerate(subtree):
                        collect_paths(value, current_path + [i])
                else:
                    paths.append(tuple(current_path))
            else:
                paths.append(tuple(current_path))
        
        collect_paths(tree_structure)
        return paths
    
    actual_paths = get_paths(tree)
    return list(zip(actual_paths, leaves)), tree_def


def tree_unflatten_with_path(path_leaf_pairs: List[Tuple[Any, Any]], tree_def: Any) -> Any:
    """Unflatten tree from path-leaf pairs.
    
    Args:
        path_leaf_pairs: List of (path, leaf) pairs
        tree_def: Tree definition from tree_flatten_with_path
        
    Returns:
        Reconstructed tree
    """
    leaves = [leaf for _, leaf in path_leaf_pairs]
    return tree_util.tree_unflatten(tree_def, leaves)


def tree_reduce(tree: Any, 
               reduce_fn: Callable,
               initializer: Any = None) -> Any:
    """Reduce all leaves in a tree to a single value.
    
    Args:
        tree: PyTree to reduce
        reduce_fn: Binary reduction function
        initializer: Initial value for reduction
        
    Returns:
        Reduced value
    """
    leaves = tree_util.tree_leaves(tree)
    
    if not leaves:
        return initializer
    
    result = leaves[0] if initializer is None else initializer
    start_idx = 1 if initializer is None else 0
    
    for leaf in leaves[start_idx:]:
        result = reduce_fn(result, leaf)
    
    return result


def tree_select(tree: Any, condition_fn: Callable) -> Any:
    """Select subtrees based on condition function.
    
    Args:
        tree: PyTree to filter
        condition_fn: Function that returns True for leaves to keep
        
    Returns:
        Filtered tree
    """
    def select_leaf(leaf):
        return leaf if condition_fn(leaf) else None
    
    return tree_util.tree_map(select_leaf, tree)


def tree_update_at_path(tree: Any, path: Tuple, new_value: Any) -> Any:
    """Update tree at specific path.
    
    Args:
        tree: PyTree to update
        path: Path to update (tuple of keys/indices)
        new_value: New value to set
        
    Returns:
        Updated tree
    """
    def update_recursive(current_tree, remaining_path):
        if not remaining_path:
            return new_value
        
        key = remaining_path[0]
        rest_path = remaining_path[1:]
        
        if isinstance(current_tree, dict):
            updated_subtree = update_recursive(current_tree[key], rest_path)
            return {**current_tree, key: updated_subtree}
        elif isinstance(current_tree, list):
            new_list = list(current_tree)
            new_list[key] = update_recursive(current_tree[key], rest_path)
            return new_list
        elif isinstance(current_tree, tuple):
            new_tuple = list(current_tree)
            new_tuple[key] = update_recursive(current_tree[key], rest_path)
            return tuple(new_tuple)
        else:
            raise TypeError(f"Cannot index into type {type(current_tree)}")
    
    return update_recursive(tree, path)


def tree_diff(tree1: Any, tree2: Any, tolerance: float = 1e-8) -> Dict[str, Any]:
    """Compare two trees and return differences.
    
    Args:
        tree1: First tree
        tree2: Second tree
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary containing difference information
    """
    path_leaf_pairs1, treedef1 = tree_flatten_with_path(tree1)
    path_leaf_pairs2, treedef2 = tree_flatten_with_path(tree2)
    
    # Check structure differences
    if treedef1 != treedef2:
        return {'structure_differs': True, 'tree1_structure': treedef1, 'tree2_structure': treedef2}
    
    # Check value differences
    differences = []
    max_diff = 0.0
    
    for (path1, leaf1), (path2, leaf2) in zip(path_leaf_pairs1, path_leaf_pairs2):
        if path1 != path2:
            differences.append({'path': path1, 'error': 'Path mismatch'})
            continue
        
        try:
            if hasattr(leaf1, 'shape') and hasattr(leaf2, 'shape'):
                # Array comparison
                diff = jnp.abs(leaf1 - leaf2)
                max_leaf_diff = float(jnp.max(diff))
                mean_diff = float(jnp.mean(diff))
                
                if max_leaf_diff > tolerance:
                    differences.append({
                        'path': path1,
                        'max_diff': max_leaf_diff,
                        'mean_diff': mean_diff,
                        'shape1': leaf1.shape,
                        'shape2': leaf2.shape
                    })
                
                max_diff = max(max_diff, max_leaf_diff)
            else:
                # Non-array comparison
                if leaf1 != leaf2:
                    differences.append({
                        'path': path1,
                        'value1': leaf1,
                        'value2': leaf2
                    })
        except Exception as e:
            differences.append({
                'path': path1,
                'error': f'Comparison failed: {e}'
            })
    
    return {
        'structure_differs': False,
        'num_differences': len(differences),
        'max_difference': max_diff,
        'differences': differences,
        'trees_equal': len(differences) == 0 and max_diff <= tolerance
    }


def tree_statistics(tree: Any) -> Dict[str, Any]:
    """Compute statistics about a PyTree.
    
    Args:
        tree: PyTree to analyze
        
    Returns:
        Dictionary of tree statistics
    """
    leaves = tree_util.tree_leaves(tree)
    
    if not leaves:
        return {'empty': True}
    
    # Basic counts
    num_leaves = len(leaves)
    num_arrays = sum(1 for leaf in leaves if hasattr(leaf, 'shape'))
    num_scalars = num_leaves - num_arrays
    
    # Array statistics
    if num_arrays > 0:
        array_leaves = [leaf for leaf in leaves if hasattr(leaf, 'shape')]
        total_elements = sum(leaf.size for leaf in array_leaves)
        total_bytes = sum(leaf.nbytes for leaf in array_leaves)
        
        shapes = [leaf.shape for leaf in array_leaves]
        dtypes = [leaf.dtype for leaf in array_leaves]
        ndims = [leaf.ndim for leaf in array_leaves]
        
        array_stats = {
            'total_elements': total_elements,
            'total_bytes': total_bytes,
            'total_mb': total_bytes / (1024 * 1024),
            'shapes': shapes,
            'unique_dtypes': list(set(str(dt) for dt in dtypes)),
            'min_ndim': min(ndims),
            'max_ndim': max(ndims),
            'mean_ndim': sum(ndims) / len(ndims)
        }
    else:
        array_stats = {}
    
    # Tree structure analysis
    path_leaf_pairs, tree_def = tree_flatten_with_path(tree)
    max_depth = max(len(path) for path, _ in path_leaf_pairs) if path_leaf_pairs else 0
    
    return {
        'empty': False,
        'num_leaves': num_leaves,
        'num_arrays': num_arrays,
        'num_scalars': num_scalars,
        'max_depth': max_depth,
        'tree_structure': tree_def,
        'array_statistics': array_stats
    }


def tree_map_with_key(f: Callable, tree: Any, *rest_trees) -> Any:
    """Map function with access to the key/path of each element.
    
    Args:
        f: Function that takes (key_path, leaf, *rest_leaves)
        tree: Primary tree
        *rest_trees: Additional trees to map over
        
    Returns:
        Mapped tree
    """
    def map_fn(key_path, *leaves):
        return f(key_path, *leaves)
    
    return tree_util.tree_map_with_path(map_fn, tree, *rest_trees)


def tree_apply_mask(tree: Any, mask_tree: Any, default_value: Any = None) -> Any:
    """Apply boolean mask to tree leaves.
    
    Args:
        tree: Tree to mask
        mask_tree: Boolean mask tree (same structure)
        default_value: Value to use where mask is False
        
    Returns:
        Masked tree
    """
    def apply_mask(leaf, mask):
        if hasattr(leaf, 'shape') and hasattr(mask, 'shape'):
            return jnp.where(mask, leaf, default_value)
        elif mask:
            return leaf
        else:
            return default_value
    
    return tree_util.tree_map(apply_mask, tree, mask_tree)


def tree_stack(trees: List[Any], axis: int = 0) -> Any:
    """Stack multiple trees along new axis.
    
    Args:
        trees: List of trees with same structure
        axis: Axis to stack along
        
    Returns:
        Stacked tree
    """
    if not trees:
        raise ValueError("Cannot stack empty list of trees")
    
    def stack_leaves(*leaves):
        return jnp.stack(leaves, axis=axis)
    
    return tree_util.tree_map(stack_leaves, *trees)


def tree_unstack(tree: Any, axis: int = 0) -> List[Any]:
    """Unstack tree along specified axis.
    
    Args:
        tree: Tree to unstack
        axis: Axis to unstack along
        
    Returns:
        List of unstacked trees
    """
    # Get size of axis to unstack
    first_leaf = tree_util.tree_leaves(tree)[0]
    if not hasattr(first_leaf, 'shape'):
        raise ValueError("Cannot unstack tree with non-array leaves")
    
    axis_size = first_leaf.shape[axis]
    
    def unstack_leaf(leaf):
        return [jnp.take(leaf, i, axis=axis) for i in range(axis_size)]
    
    unstacked_leaves_list = tree_util.tree_map(unstack_leaf, tree)
    
    # Reorganize to list of trees
    result_trees = []
    for i in range(axis_size):
        tree_i = tree_util.tree_map(lambda leaves_list: leaves_list[i], unstacked_leaves_list)
        result_trees.append(tree_i)
    
    return result_trees


def tree_take(tree: Any, indices: jnp.ndarray, axis: int = 0) -> Any:
    """Take elements from tree along specified axis.
    
    Args:
        tree: Tree to index
        indices: Indices to take
        axis: Axis to take along
        
    Returns:
        Tree with selected elements
    """
    def take_leaf(leaf):
        if hasattr(leaf, 'shape'):
            return jnp.take(leaf, indices, axis=axis)
        else:
            return leaf
    
    return tree_util.tree_map(take_leaf, tree)


def tree_concatenate(trees: List[Any], axis: int = 0) -> Any:
    """Concatenate trees along specified axis.
    
    Args:
        trees: List of trees to concatenate
        axis: Axis to concatenate along
        
    Returns:
        Concatenated tree
    """
    if not trees:
        raise ValueError("Cannot concatenate empty list of trees")
    
    def concat_leaves(*leaves):
        array_leaves = [leaf for leaf in leaves if hasattr(leaf, 'shape')]
        if array_leaves:
            return jnp.concatenate(array_leaves, axis=axis)
        else:
            return leaves[0]  # Return first non-array leaf
    
    return tree_util.tree_map(concat_leaves, *trees)


def tree_slice(tree: Any, slice_obj: Union[slice, Tuple[slice, ...]], axis: int = 0) -> Any:
    """Slice tree along specified axis.
    
    Args:
        tree: Tree to slice
        slice_obj: Slice object or tuple of slices
        axis: Primary axis for slicing (if slice_obj is just a slice)
        
    Returns:
        Sliced tree
    """
    def slice_leaf(leaf):
        if hasattr(leaf, 'shape'):
            if isinstance(slice_obj, slice):
                slices = [slice(None)] * leaf.ndim
                slices[axis] = slice_obj
                return leaf[tuple(slices)]
            else:
                return leaf[slice_obj]
        else:
            return leaf
    
    return tree_util.tree_map(slice_leaf, tree)