# File location: jax-nsl/src/autodiff/custom_jvp.py

"""
Custom JVP (Jacobian-Vector Product) implementations.

This module provides examples of custom forward-mode differentiation
for operations that need specialized derivative behavior.
"""

import jax
import jax.numpy as jnp
from jax import custom_jvp
from typing import Tuple


@custom_jvp
def clip_gradient_jvp(x: jnp.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> jnp.ndarray:
    """Clip gradients during forward-mode differentiation.
    
    Args:
        x: Input tensor
        min_val: Minimum gradient value
        max_val: Maximum gradient value
        
    Returns:
        x (unchanged)
    """
    return x


@clip_gradient_jvp.defjvp
def clip_gradient_jvp_jvp(primals, tangents):
    """JVP rule for gradient clipping."""
    x, min_val, max_val = primals
    dx, _, _ = tangents
    
    # Clip the tangent vector
    clipped_dx = jnp.clip(dx, min_val, max_val)
    
    return x, clipped_dx


@custom_jvp
def custom_sqrt_jvp(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Square root with stable JVP.
    
    Args:
        x: Input tensor
        eps: Stabilization epsilon
        
    Returns:
        sqrt(max(x, 0))
    """
    return jnp.sqrt(jnp.maximum(x, 0.0))


@custom_sqrt_jvp.defjvp
def custom_sqrt_jvp_jvp(primals, tangents):
    """JVP rule for stable square root."""
    x, eps = primals
    dx, _ = tangents
    
    y = jnp.sqrt(jnp.maximum(x, 0.0))
    # dy/dx = 1/(2*sqrt(x)), but stabilized
    dy = dx / (2.0 * jnp.sqrt(jnp.maximum(x, eps)))
    
    return y, dy


@custom_jvp
def piecewise_linear_jvp(x: jnp.ndarray, breakpoints: jnp.ndarray, slopes: jnp.ndarray) -> jnp.ndarray:
    """Piecewise linear function with custom JVP.
    
    Args:
        x: Input values
        breakpoints: Breakpoint locations (sorted)
        slopes: Slopes for each segment
        
    Returns:
        Piecewise linear function output
    """
    # Find which segment each x belongs to
    segment_idx = jnp.searchsorted(breakpoints, x, side='right') - 1
    segment_idx = jnp.clip(segment_idx, 0, len(slopes) - 1)
    
    # Compute output using appropriate slopes
    if len(breakpoints) == 0:
        return x * slopes[0]
    
    # For simplicity, assume breakpoints start at 0
    y = jnp.take(slopes, segment_idx) * x
    return y


@piecewise_linear_jvp.defjvp
def piecewise_linear_jvp_jvp(primals, tangents):
    """JVP rule for piecewise linear function."""
    x, breakpoints, slopes = primals
    dx, _, _ = tangents
    
    # Find segments
    segment_idx = jnp.searchsorted(breakpoints, x, side='right') - 1
    segment_idx = jnp.clip(segment_idx, 0, len(slopes) - 1)
    
    # Forward pass
    y = jnp.take(slopes, segment_idx) * x
    
    # JVP: derivative is just the slope in each segment
    dy = jnp.take(slopes, segment_idx) * dx
    
    return y, dy


@custom_jvp
def saturated_activation_jvp(x: jnp.ndarray, 
                           lower_bound: float = -1.0, 
                           upper_bound: float = 1.0,
                           smoothness: float = 1.0) -> jnp.ndarray:
    """Smooth saturated activation function.
    
    Uses a smooth approximation to a clipped linear function.
    
    Args:
        x: Input tensor
        lower_bound: Lower saturation bound
        upper_bound: Upper saturation bound
        smoothness: Smoothness parameter (higher = smoother)
        
    Returns:
        Saturated activation output
    """
    # Use smooth clipping with tanh
    center = (upper_bound + lower_bound) / 2
    width = (upper_bound - lower_bound) / 2
    
    scaled_x = (x - center) / width * smoothness
    y = center + width * jnp.tanh(scaled_x)
    
    return y


@saturated_activation_jvp.defjvp
def saturated_activation_jvp_jvp(primals, tangents):
    """JVP rule for saturated activation."""
    x, lower_bound, upper_bound, smoothness = primals
    dx, _, _, _ = tangents
    
    center = (upper_bound + lower_bound) / 2
    width = (upper_bound - lower_bound) / 2
    
    scaled_x = (x - center) / width * smoothness
    tanh_scaled_x = jnp.tanh(scaled_x)
    
    # Forward pass
    y = center + width * tanh_scaled_x
    
    # JVP: derivative of tanh is sech^2 = 1 - tanh^2
    sech_squared = 1 - tanh_scaled_x ** 2
    dy = smoothness * sech_squared * dx
    
    return y, dy


@custom_jvp
def soft_sign_jvp(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    """Soft sign activation: x / (alpha + |x|).
    
    Args:
        x: Input tensor
        alpha: Smoothness parameter
        
    Returns:
        Soft sign activation output
    """
    return x / (alpha + jnp.abs(x))


@soft_sign_jvp.defjvp
def soft_sign_jvp_jvp(primals, tangents):
    """JVP rule for soft sign activation."""
    x, alpha = primals
    dx, _ = tangents
    
    abs_x = jnp.abs(x)
    denominator = alpha + abs_x
    
    # Forward pass
    y = x / denominator
    
    # Derivative: d/dx [x / (alpha + |x|)] = alpha / (alpha + |x|)^2
    dy = alpha * dx / (denominator ** 2)
    
    return y, dy


@custom_jvp
def gaussian_activation_jvp(x: jnp.ndarray, sigma: float = 1.0) -> jnp.ndarray:
    """Gaussian activation function: exp(-x^2 / (2 * sigma^2)).
    
    Args:
        x: Input tensor
        sigma: Standard deviation parameter
        
    Returns:
        Gaussian activation output
    """
    return jnp.exp(-x**2 / (2 * sigma**2))


@gaussian_activation_jvp.defjvp
def gaussian_activation_jvp_jvp(primals, tangents):
    """JVP rule for Gaussian activation."""
    x, sigma = primals
    dx, _ = tangents
    
    # Forward pass
    y = jnp.exp(-x**2 / (2 * sigma**2))
    
    # Derivative: d/dx [exp(-x^2/(2*sigma^2))] = -x/sigma^2 * exp(-x^2/(2*sigma^2))
    dy = (-x / sigma**2) * y * dx
    
    return y, dy


@custom_jvp
def learnable_activation_jvp(x: jnp.ndarray, params: jnp.ndarray) -> jnp.ndarray:
    """Learnable activation function using polynomial approximation.
    
    Args:
        x: Input tensor
        params: Polynomial coefficients [c0, c1, c2, ...]
        
    Returns:
        Polynomial activation output: sum(ci * x^i)
    """
    # Compute polynomial using Horner's method for stability
    result = params[-1]
    for coeff in reversed(params[:-1]):
        result = result * x + coeff
    
    return result


@learnable_activation_jvp.defjvp
def learnable_activation_jvp_jvp(primals, tangents):
    """JVP rule for learnable polynomial activation."""
    x, params = primals
    dx, _ = tangents
    
    # Forward pass
    result = params[-1]
    for coeff in reversed(params[:-1]):
        result = result * x + coeff
    
    # Derivative: sum(i * ci * x^(i-1))
    if len(params) == 1:
        dy = jnp.zeros_like(dx)
    else:
        # Compute derivative coefficients
        deriv_params = params[1:] * jnp.arange(1, len(params))
        
        # Evaluate derivative polynomial
        deriv_result = deriv_params[-1] if len(deriv_params) > 0 else 0.0
        for coeff in reversed(deriv_params[:-1]):
            deriv_result = deriv_result * x + coeff
        
        dy = deriv_result * dx
    
    return result, dy