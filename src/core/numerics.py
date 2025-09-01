# File location: jax-nsl/src/core/numerics.py

"""
Numerically stable operations: logsumexp, clipping, and safe math.

This module provides numerically stable implementations of common
operations that are prone to overflow, underflow, or precision issues.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Union, Tuple, Any
import math


def safe_log(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Numerically stable logarithm.
    
    Args:
        x: Input array
        eps: Small epsilon to prevent log(0)
        
    Returns:
        log(max(x, eps))
    """
    return jnp.log(jnp.maximum(x, eps))


def safe_exp(x: jnp.ndarray, max_val: Optional[float] = None) -> jnp.ndarray:
    """Numerically stable exponential with optional clipping.
    
    Args:
        x: Input array
        max_val: Maximum value before exp (default: log of float32 max)
        
    Returns:
        exp(min(x, max_val))
    """
    if max_val is None:
        max_val = jnp.log(jnp.finfo(x.dtype).max) - 1.0
    return jnp.exp(jnp.minimum(x, max_val))


def logsumexp_stable(x: jnp.ndarray, 
                    axis: Optional[Union[int, Tuple[int, ...]]] = None,
                    keepdims: bool = False,
                    return_max: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Numerically stable log-sum-exp computation.
    
    Computes log(sum(exp(x), axis)) in a numerically stable way by
    factoring out the maximum value before exponentiation.
    
    Args:
        x: Input array
        axis: Axis or axes along which to sum
        keepdims: Whether to keep dimensions
        return_max: Whether to return the max value used for stability
        
    Returns:
        log-sum-exp result, optionally with max value
    """
    x_max = jnp.max(x, axis=axis, keepdims=True)
    
    # Handle case where all values are -inf
    x_max = jnp.where(jnp.isfinite(x_max), x_max, 0.0)
    
    result = x_max + jnp.log(jnp.sum(jnp.exp(x - x_max), axis=axis, keepdims=True))
    
    if not keepdims:
        result = jnp.squeeze(result, axis=axis)
        x_max = jnp.squeeze(x_max, axis=axis)
    
    if return_max:
        return result, x_max
    return result


def softmax_stable(x: jnp.ndarray, 
                  axis: int = -1,
                  temperature: float = 1.0) -> jnp.ndarray:
    """Numerically stable softmax computation.
    
    Args:
        x: Input logits
        axis: Axis along which to compute softmax
        temperature: Temperature parameter (higher = more uniform)
        
    Returns:
        Softmax probabilities
    """
    x = x / temperature
    x_max = jnp.max(x, axis=axis, keepdims=True)
    x_shifted = x - x_max
    exp_x = jnp.exp(x_shifted)
    return exp_x / jnp.sum(exp_x, axis=axis, keepdims=True)


def log_softmax_stable(x: jnp.ndarray, 
                      axis: int = -1,
                      temperature: float = 1.0) -> jnp.ndarray:
    """Numerically stable log-softmax computation.
    
    Args:
        x: Input logits
        axis: Axis along which to compute log-softmax
        temperature: Temperature parameter
        
    Returns:
        Log-softmax values
    """
    x = x / temperature
    return x - logsumexp_stable(x, axis=axis, keepdims=True)


def clip_gradients(grads: Any, 
                  max_norm: Optional[float] = None,
                  max_value: Optional[float] = None) -> Any:
    """Clip gradients by global norm or value.
    
    Args:
        grads: Gradient pytree
        max_norm: Maximum gradient norm (global clipping)
        max_value: Maximum gradient value (element-wise clipping)
        
    Returns:
        Clipped gradients with same structure
    """
    if max_norm is not None:
        # Global norm clipping
        global_norm = safe_norm(grads)
        clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
        grads = jax.tree_util.tree_map(lambda g: g * clip_factor, grads)
    
    if max_value is not None:
        # Element-wise value clipping
        grads = jax.tree_util.tree_map(
            lambda g: jnp.clip(g, -max_value, max_value), 
            grads
        )
    
    return grads


def safe_norm(tree: Any, ord: Optional[Union[int, float, str]] = None) -> jnp.ndarray:
    """Compute norm of a pytree of arrays.
    
    Args:
        tree: PyTree of arrays
        ord: Order of the norm (2 for L2, 1 for L1, etc.)
        
    Returns:
        Scalar norm value
    """
    if ord is None or ord == 2 or ord == 'fro':
        # L2 norm (default)
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in leaves))
    
    elif ord == 1:
        # L1 norm
        leaves = jax.tree_util.tree_leaves(tree)
        return sum(jnp.sum(jnp.abs(leaf)) for leaf in leaves)
    
    elif ord == jnp.inf or ord == 'inf':
        # L-infinity norm
        leaves = jax.tree_util.tree_leaves(tree)
        return max(jnp.max(jnp.abs(leaf)) for leaf in leaves)
    
    else:
        raise ValueError(f"Unsupported norm order: {ord}")


def safe_divide(x: jnp.ndarray, 
               y: jnp.ndarray, 
               eps: float = 1e-8,
               replace_nan: bool = True) -> jnp.ndarray:
    """Numerically stable division with optional NaN replacement.
    
    Args:
        x: Numerator
        y: Denominator
        eps: Small epsilon added to denominator
        replace_nan: Whether to replace NaN results with 0
        
    Returns:
        x / (y + eps) with optional NaN handling
    """
    result = x / (y + eps)
    
    if replace_nan:
        result = jnp.where(jnp.isfinite(result), result, 0.0)
    
    return result


def stable_sigmoid(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable sigmoid computation.
    
    Uses the identity: sigmoid(x) = exp(x) / (1 + exp(x)) for x >= 0
                                 = 1 / (1 + exp(-x)) for x < 0
    
    Args:
        x: Input array
        
    Returns:
        Sigmoid of input
    """
    return jnp.where(
        x >= 0,
        1.0 / (1.0 + jnp.exp(-x)),
        jnp.exp(x) / (1.0 + jnp.exp(x))
    )


def stable_tanh(x: jnp.ndarray) -> jnp.ndarray:
    """Numerically stable tanh computation.
    
    Args:
        x: Input array
        
    Returns:
        Tanh of input
    """
    # Use the identity: tanh(x) = 2 * sigmoid(2x) - 1
    return 2.0 * stable_sigmoid(2.0 * x) - 1.0


def smooth_max(x: jnp.ndarray, 
               axis: Optional[int] = None,
               alpha: float = 1.0) -> jnp.ndarray:
    """Smooth approximation to max function.
    
    Uses the smooth maximum: smooth_max(x) = log(sum(exp(alpha * x))) / alpha
    
    Args:
        x: Input array
        axis: Axis along which to compute smooth max
        alpha: Smoothness parameter (higher = closer to true max)
        
    Returns:
        Smooth maximum
    """
    return logsumexp_stable(alpha * x, axis=axis) / alpha


def smooth_min(x: jnp.ndarray,
               axis: Optional[int] = None, 
               alpha: float = 1.0) -> jnp.ndarray:
    """Smooth approximation to min function.
    
    Args:
        x: Input array
        axis: Axis along which to compute smooth min
        alpha: Smoothness parameter (higher = closer to true min)
        
    Returns:
        Smooth minimum
    """
    return -smooth_max(-x, axis=axis, alpha=alpha)


def gumbel_softmax(logits: jnp.ndarray,
                  temperature: float,
                  key: jax.Array,
                  axis: int = -1,
                  hard: bool = False) -> jnp.ndarray:
    """Gumbel-Softmax sampling for differentiable discrete sampling.
    
    Args:
        logits: Input logits
        temperature: Gumbel softmax temperature
        key: Random key for Gumbel noise
        axis: Axis along which to apply softmax
        hard: Whether to use straight-through estimator
        
    Returns:
        Gumbel-softmax samples
    """
    # Sample Gumbel noise
    gumbel_noise = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape, minval=1e-10)))
    
    # Add noise to logits and apply softmax
    y = softmax_stable((logits + gumbel_noise) / temperature, axis=axis)
    
    if hard:
        # Straight-through estimator
        y_hard = jnp.eye(logits.shape[axis])[jnp.argmax(y, axis=axis)]
        y = y_hard - jax.lax.stop_gradient(y) + y
    
    return y