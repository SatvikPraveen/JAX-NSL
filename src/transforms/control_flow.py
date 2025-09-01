# File location: jax-nsl/src/transforms/control_flow.py

"""
Control flow utilities: lax.cond, switch, while_loop patterns.

This module provides utilities for JAX control flow operations
with better error handling and common usage patterns.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Any, Optional, Union, Tuple, List
import functools


def safe_cond(pred: Union[bool, jnp.ndarray],
             true_fun: Callable,
             false_fun: Callable,
             *operands,
             linear: Tuple[bool, ...] = None) -> Any:
    """Safe conditional execution with error handling.
    
    Args:
        pred: Boolean condition
        true_fun: Function to execute if pred is True
        false_fun: Function to execute if pred is False
        *operands: Arguments for the functions
        linear: Linearity specification for operands
        
    Returns:
        Result of conditional execution
    """
    try:
        return lax.cond(pred, true_fun, false_fun, *operands, linear=linear)
    except Exception as e:
        print(f"Conditional execution failed: {e}")
        # Try to execute both branches to check for errors
        try:
            true_result = true_fun(*operands)
            false_result = false_fun(*operands)
            # Return based on pred if possible
            if isinstance(pred, bool):
                return true_result if pred else false_result
            else:
                # For array pred, use where
                return jnp.where(pred, true_result, false_result)
        except:
            raise e


def switch_case(index: Union[int, jnp.ndarray],
               branches: List[Callable],
               *operands,
               linear: Tuple[bool, ...] = None) -> Any:
    """Multi-way conditional execution.
    
    Args:
        index: Branch index to execute
        branches: List of functions for each branch
        *operands: Arguments for the functions
        linear: Linearity specification
        
    Returns:
        Result of executing selected branch
    """
    try:
        return lax.switch(index, branches, *operands, linear=linear)
    except Exception as e:
        print(f"Switch execution failed: {e}")
        # Fallback to manual selection
        if isinstance(index, int):
            if 0 <= index < len(branches):
                return branches[index](*operands)
            else:
                raise IndexError(f"Branch index {index} out of range")
        else:
            # For array indices, more complex fallback needed
            raise e


def while_loop_safe(cond_fun: Callable,
                   body_fun: Callable,
                   init_val: Any,
                   max_iterations: Optional[int] = None) -> Any:
    """Safe while loop with optional iteration limit.
    
    Args:
        cond_fun: Condition function
        body_fun: Loop body function
        init_val: Initial value
        max_iterations: Maximum iterations to prevent infinite loops
        
    Returns:
        Final loop value
    """
    if max_iterations is None:
        return lax.while_loop(cond_fun, body_fun, init_val)
    
    # Add iteration counter to prevent infinite loops
    def augmented_cond(state):
        val, count = state
        return jnp.logical_and(cond_fun(val), count < max_iterations)
    
    def augmented_body(state):
        val, count = state
        new_val = body_fun(val)
        return new_val, count + 1
    
    augmented_init = (init_val, 0)
    final_state, _ = lax.while_loop(augmented_cond, augmented_body, augmented_init)
    
    return final_state


def for_loop(lower: int,
            upper: int,
            body_fun: Callable,
            init_val: Any,
            unroll: int = 1) -> Any:
    """For loop using lax.fori_loop.
    
    Args:
        lower: Loop start (inclusive)
        upper: Loop end (exclusive)
        body_fun: Function (i, val) -> new_val
        init_val: Initial accumulator value
        unroll: Number of iterations to unroll
        
    Returns:
        Final accumulator value
    """
    return lax.fori_loop(lower, upper, body_fun, init_val, unroll=unroll)


def dynamic_slice_safe(operand: jnp.ndarray,
                      start_indices: Union[List[int], jnp.ndarray],
                      slice_sizes: Union[List[int], Tuple[int, ...]]) -> jnp.ndarray:
    """Safe dynamic slicing with bounds checking.
    
    Args:
        operand: Array to slice
        start_indices: Starting indices for slice
        slice_sizes: Size of slice in each dimension
        
    Returns:
        Dynamically sliced array
    """
    # Ensure start_indices are within bounds
    start_indices = jnp.asarray(start_indices)
    operand_shape = jnp.array(operand.shape)
    slice_sizes = jnp.array(slice_sizes)
    
    # Clamp start indices to valid range
    max_start = operand_shape - slice_sizes
    start_indices = jnp.maximum(0, jnp.minimum(start_indices, max_start))
    
    return lax.dynamic_slice(operand, start_indices, slice_sizes)


def conditional_update(condition: jnp.ndarray,
                      x: jnp.ndarray,
                      update_fun: Callable,
                      *args) -> jnp.ndarray:
    """Conditionally update array elements.
    
    Args:
        condition: Boolean mask for updates
        x: Array to update
        update_fun: Function to compute new values
        *args: Additional arguments for update_fun
        
    Returns:
        Array with conditional updates applied
    """
    def true_branch(*operands):
        x_val = operands[0]
        return update_fun(x_val, *operands[1:])
    
    def false_branch(*operands):
        return operands[0]  # Return unchanged
    
    return jnp.where(
        condition,
        lax.cond(
            jnp.any(condition),
            true_branch,
            false_branch,
            x, *args
        ),
        x
    )


def binary_search(f: Callable,
                 target: float,
                 low: float,
                 high: float,
                 tolerance: float = 1e-6,
                 max_iterations: int = 100) -> float:
    """Binary search using while_loop.
    
    Args:
        f: Function to search over (must be monotonic)
        target: Target value to find
        low: Lower search bound
        high: Upper search bound
        tolerance: Convergence tolerance
        max_iterations: Maximum search iterations
        
    Returns:
        Input value where f(x) ≈ target
    """
    def cond_fun(state):
        low_val, high_val, iterations = state
        converged = jnp.abs(high_val - low_val) < tolerance
        max_iters_reached = iterations >= max_iterations
        return jnp.logical_not(jnp.logical_or(converged, max_iters_reached))
    
    def body_fun(state):
        low_val, high_val, iterations = state
        mid = (low_val + high_val) / 2
        f_mid = f(mid)
        
        # Update bounds based on comparison
        new_low = lax.cond(f_mid < target, lambda: mid, lambda: low_val)
        new_high = lax.cond(f_mid < target, lambda: high_val, lambda: mid)
        
        return new_low, new_high, iterations + 1
    
    initial_state = (low, high, 0)
    final_low, final_high, _ = lax.while_loop(cond_fun, body_fun, initial_state)
    
    return (final_low + final_high) / 2


def iterative_solver(f: Callable,
                    x0: jnp.ndarray,
                    tolerance: float = 1e-6,
                    max_iterations: int = 100,
                    damping: float = 1.0) -> Tuple[jnp.ndarray, bool]:
    """Generic iterative solver using while_loop.
    
    Args:
        f: Update function x -> x_new
        x0: Initial guess
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        damping: Damping factor for updates
        
    Returns:
        (solution, converged) tuple
    """
    def cond_fun(state):
        x, x_prev, iterations, converged = state
        not_converged = jnp.logical_not(converged)
        not_max_iters = iterations < max_iterations
        return jnp.logical_and(not_converged, not_max_iters)
    
    def body_fun(state):
        x, x_prev, iterations, _ = state
        x_new = f(x)
        
        # Apply damping
        x_damped = x + damping * (x_new - x)
        
        # Check convergence
        diff = jnp.linalg.norm(x_damped - x)
        converged = diff < tolerance
        
        return x_damped, x, iterations + 1, converged
    
    initial_state = (x0, x0, 0, False)
    final_x, _, iterations, converged = lax.while_loop(cond_fun, body_fun, initial_state)
    
    return final_x, converged


def select_n(pred: jnp.ndarray,
            on_true: jnp.ndarray,
            on_false: jnp.ndarray) -> jnp.ndarray:
    """Generalized select operation.
    
    Args:
        pred: Boolean condition array
        on_true: Values to select when pred is True
        on_false: Values to select when pred is False
        
    Returns:
        Selected values
    """
    return lax.select(pred, on_true, on_false)


def gather_nd(params: jnp.ndarray,
             indices: jnp.ndarray,
             batch_dims: int = 0) -> jnp.ndarray:
    """N-dimensional gather operation.
    
    Args:
        params: Parameter array to gather from
        indices: Indices for gathering
        batch_dims: Number of batch dimensions
        
    Returns:
        Gathered values
    """
    return lax.gather(
        params,
        indices,
        lax.GatherDimensionNumbers(
            offset_dims=tuple(range(batch_dims, len(params.shape))),
            collapsed_slice_dims=tuple(range(len(indices.shape) - 1)),
            start_index_map=tuple(range(len(indices.shape) - 1))
        ),
        slice_sizes=(1,) * (len(indices.shape) - 1) + params.shape[len(indices.shape) - 1:]
    )


def scatter_add_nd(operand: jnp.ndarray,
                  indices: jnp.ndarray,
                  updates: jnp.ndarray) -> jnp.ndarray:
    """N-dimensional scatter-add operation.
    
    Args:
        operand: Base array to scatter into
        indices: Indices where to scatter
        updates: Values to add
        
    Returns:
        Array with scattered updates added
    """
    return lax.scatter_add(
        operand,
        indices,
        updates,
        lax.ScatterDimensionNumbers(
            update_window_dims=tuple(range(1, len(updates.shape))),
            inserted_window_dims=(0,),
            scatter_dims_to_operand_dims=(0,)
        )
    )