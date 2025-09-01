# File location: jax-nsl/src/transforms/scan_utils.py

"""
Scan utilities for loops, RNNs, and sequential operations.

This module provides utilities for using JAX's scan transformation
for efficient sequential computations and recurrent patterns.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Any, Tuple, Optional, Union
import functools


def cumulative_op(op: Callable, 
                 xs: jnp.ndarray, 
                 init: Optional[jnp.ndarray] = None,
                 axis: int = 0,
                 reverse: bool = False) -> jnp.ndarray:
    """Cumulative operation using scan.
    
    Args:
        op: Binary operation (e.g., jnp.add, jnp.multiply)
        xs: Input array
        init: Initial value (defaults to first element)
        axis: Axis along which to compute cumulative operation
        reverse: Whether to scan in reverse order
        
    Returns:
        Cumulative results
    """
    if axis != 0:
        # Move target axis to front
        xs = jnp.moveaxis(xs, axis, 0)
    
    def scan_fn(carry, x):
        result = op(carry, x)
        return result, result
    
    if init is None:
        init = xs[0]
        xs_to_scan = xs[1:]
    else:
        xs_to_scan = xs
    
    if reverse:
        xs_to_scan = xs_to_scan[::-1]
    
    _, results = lax.scan(scan_fn, init, xs_to_scan)
    
    if init is None:
        results = jnp.concatenate([init[None], results], axis=0)
    
    if reverse:
        results = results[::-1]
    
    if axis != 0:
        # Move axis back to original position
        results = jnp.moveaxis(results, 0, axis)
    
    return results


def running_statistics(xs: jnp.ndarray, 
                      axis: int = 0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute running mean and variance using scan.
    
    Args:
        xs: Input array
        axis: Axis along which to compute statistics
        
    Returns:
        (running_means, running_vars) tuple
    """
    if axis != 0:
        xs = jnp.moveaxis(xs, axis, 0)
    
    def scan_fn(state, x):
        count, mean, var = state
        count += 1
        
        # Online update formulas (Welford's algorithm)
        delta = x - mean
        mean = mean + delta / count
        delta2 = x - mean
        var = var + delta * delta2
        
        return (count, mean, var), (mean, var / jnp.maximum(count - 1, 1))
    
    # Initialize with first element
    init_state = (1.0, xs[0], jnp.zeros_like(xs[0]))
    xs_to_scan = xs[1:]
    
    _, (running_means, running_vars) = lax.scan(scan_fn, init_state, xs_to_scan)
    
    # Prepend initial values
    initial_mean = xs[0][None]
    initial_var = jnp.zeros_like(initial_mean)
    
    running_means = jnp.concatenate([initial_mean, running_means], axis=0)
    running_vars = jnp.concatenate([initial_var, running_vars], axis=0)
    
    if axis != 0:
        running_means = jnp.moveaxis(running_means, 0, axis)
        running_vars = jnp.moveaxis(running_vars, 0, axis)
    
    return running_means, running_vars


def sequential_apply(funs: list, 
                    init_state: Any,
                    inputs: Optional[jnp.ndarray] = None) -> Tuple[Any, list]:
    """Apply sequence of functions using scan.
    
    Args:
        funs: List of functions to apply sequentially
        init_state: Initial state
        inputs: Optional inputs for each function
        
    Returns:
        (final_state, outputs) tuple
    """
    if inputs is None:
        inputs = [None] * len(funs)
    
    def scan_fn(state, fun_input_pair):
        fun, inp = fun_input_pair
        if inp is None:
            new_state, output = fun(state)
        else:
            new_state, output = fun(state, inp)
        return new_state, output
    
    fun_input_pairs = list(zip(funs, inputs))
    final_state, outputs = lax.scan(scan_fn, init_state, fun_input_pairs)
    
    return final_state, outputs


def rnn_scan(rnn_cell: Callable,
            inputs: jnp.ndarray,
            init_state: Any,
            reverse: bool = False,
            unroll: int = 1) -> Tuple[Any, jnp.ndarray]:
    """Apply RNN cell over sequence using scan.
    
    Args:
        rnn_cell: RNN cell function (state, input) -> (new_state, output)
        inputs: Input sequence with shape (seq_len, ...)
        init_state: Initial hidden state
        reverse: Whether to process sequence in reverse
        unroll: Number of scan iterations to unroll
        
    Returns:
        (final_state, outputs) tuple
    """
    if reverse:
        inputs = inputs[::-1]
    
    final_state, outputs = lax.scan(
        rnn_cell, 
        init_state, 
        inputs,
        unroll=unroll
    )
    
    if reverse:
        outputs = outputs[::-1]
    
    return final_state, outputs


def ode_solve_scan(ode_fn: Callable,
                  y0: jnp.ndarray,
                  t: jnp.ndarray,
                  dt: Optional[float] = None) -> jnp.ndarray:
    """Simple ODE solver using scan (Euler method).
    
    Args:
        ode_fn: ODE function dy/dt = f(y, t)
        y0: Initial condition
        t: Time points
        dt: Time step (computed from t if not provided)
        
    Returns:
        Solution trajectory
    """
    if dt is None:
        dt = t[1] - t[0]
    
    def euler_step(y, t_curr):
        dydt = ode_fn(y, t_curr)
        y_next = y + dt * dydt
        return y_next, y
    
    # Skip first time point since it's the initial condition
    times = t[1:]
    
    _, trajectory = lax.scan(euler_step, y0, times)
    
    # Prepend initial condition
    trajectory = jnp.concatenate([y0[None], trajectory], axis=0)
    
    return trajectory


def associative_scan(op: Callable, elems: jnp.ndarray) -> jnp.ndarray:
    """Associative scan for parallel prefix computation.
    
    Args:
        op: Associative binary operation
        elems: Elements to scan over
        
    Returns:
        Prefix scan results
    """
    def scan_fn(a, b):
        return op(a, b)
    
    return lax.associative_scan(scan_fn, elems)


def windowed_scan(fun: Callable,
                 inputs: jnp.ndarray,
                 window_size: int,
                 stride: int = 1,
                 init: Optional[Any] = None) -> Tuple[Any, jnp.ndarray]:
    """Apply function to sliding windows using scan.
    
    Args:
        fun: Function to apply to each window
        inputs: Input sequence
        window_size: Size of sliding window
        stride: Stride between windows
        init: Initial state for scan
        
    Returns:
        (final_state, outputs) from windowed processing
    """
    # Create sliding windows
    num_windows = (len(inputs) - window_size) // stride + 1
    windows = []
    
    for i in range(num_windows):
        start_idx = i * stride
        end_idx = start_idx + window_size
        windows.append(inputs[start_idx:end_idx])
    
    windows = jnp.stack(windows)
    
    if init is None:
        # Function doesn't use state
        def scan_fn(_, window):
            result = fun(window)
            return None, result
        
        _, outputs = lax.scan(scan_fn, None, windows)
        return None, outputs
    else:
        # Function uses state
        def scan_fn(state, window):
            new_state, output = fun(state, window)
            return new_state, output
        
        return lax.scan(scan_fn, init, windows)


def dynamic_rnn(cell: Callable,
               inputs: jnp.ndarray,
               sequence_lengths: jnp.ndarray,
               init_state: Any,
               time_major: bool = True) -> Tuple[Any, jnp.ndarray]:
    """Dynamic RNN with variable sequence lengths.
    
    Args:
        cell: RNN cell function
        inputs: Input sequences
        sequence_lengths: Length of each sequence
        init_state: Initial state
        time_major: Whether time is the first axis
        
    Returns:
        (final_states, outputs) for each sequence
    """
    if not time_major:
        inputs = jnp.transpose(inputs, (1, 0, 2))  # (batch, time, ...) -> (time, batch, ...)
    
    max_time = inputs.shape[0]
    batch_size = inputs.shape[1]
    
    def scan_fn(state, time_input):
        time_step, inp = time_input
        
        # Create mask for sequences that are still active
        mask = time_step < sequence_lengths
        
        # Apply cell only to active sequences
        def apply_cell(s, x):
            return cell(s, x)
        
        def identity(s, x):
            return s, jnp.zeros_like(s)  # Assume output has same shape as state
        
        new_state, output = jax.lax.cond(
            jnp.any(mask),
            apply_cell,
            identity,
            state, inp
        )
        
        # Mask outputs and states
        output = jnp.where(mask[:, None], output, 0.0)
        new_state = jax.tree_util.tree_map(
            lambda s_new, s_old: jnp.where(mask[:, None], s_new, s_old),
            new_state, state
        )
        
        return new_state, output
    
    time_steps = jnp.arange(max_time)
    time_inputs = list(zip(time_steps, inputs))
    
    final_state, outputs = lax.scan(scan_fn, init_state, time_inputs)
    
    if not time_major:
        outputs = jnp.transpose(outputs, (1, 0, 2))  # (time, batch, ...) -> (batch, time, ...)
    
    return final_state, outputs


def scan_with_checkpointing(fun: Callable,
                          init: Any,
                          xs: jnp.ndarray,
                          checkpoint_every: int = 1) -> Tuple[Any, jnp.ndarray]:
    """Scan with gradient checkpointing for memory efficiency.
    
    Args:
        fun: Scan function
        init: Initial carry
        xs: Sequence to scan over
        checkpoint_every: How often to checkpoint gradients
        
    Returns:
        (final_carry, outputs) tuple
    """
    if checkpoint_every == 1:
        # No checkpointing
        return lax.scan(fun, init, xs)
    
    # Split sequence into chunks
    seq_len = len(xs)
    num_chunks = (seq_len + checkpoint_every - 1) // checkpoint_every
    
    # Pad sequence if needed
    pad_len = num_chunks * checkpoint_every - seq_len
    if pad_len > 0:
        xs = jnp.concatenate([xs, jnp.zeros_like(xs[:pad_len])], axis=0)
    
    xs_chunks = xs.reshape(num_chunks, checkpoint_every, *xs.shape[1:])
    
    def chunk_scan(carry, chunk):
        return lax.scan(fun, carry, chunk)
    
    # Apply checkpointing to chunk processing
    checkpointed_chunk_scan = jax.checkpoint(chunk_scan)
    
    final_carry, chunk_outputs = lax.scan(checkpointed_chunk_scan, init, xs_chunks)
    
    # Reshape outputs back to original sequence format
    outputs = chunk_outputs.reshape(-1, *chunk_outputs.shape[2:])
    
    # Remove padding if any
    if pad_len > 0:
        outputs = outputs[:seq_len]
    
    return final_carry, outputs