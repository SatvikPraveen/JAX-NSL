# File location: jax-nsl/src/autodiff/custom_vjp.py

"""
Custom VJP (Vector-Jacobian Product) implementations.

This module provides examples of custom backward-mode differentiation
for operations that need specialized gradient behavior.
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp
from typing import Tuple


@custom_vjp
def clip_gradient_vjp(x: jnp.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> jnp.ndarray:
    """Clip gradients during backprop while preserving forward values.
    
    Forward: returns x unchanged
    Backward: clips gradients to [min_val, max_val]
    
    Args:
        x: Input tensor
        min_val: Minimum gradient value
        max_val: Maximum gradient value
        
    Returns:
        x (unchanged in forward pass)
    """
    return x


def clip_gradient_vjp_fwd(x: jnp.ndarray, min_val: float, max_val: float) -> Tuple[jnp.ndarray, Tuple]:
    """Forward pass for gradient clipping VJP."""
    return x, (min_val, max_val)


def clip_gradient_vjp_bwd(res: Tuple, g: jnp.ndarray) -> Tuple[jnp.ndarray, None, None]:
    """Backward pass for gradient clipping VJP."""
    min_val, max_val = res
    clipped_g = jnp.clip(g, min_val, max_val)
    return clipped_g, None, None


clip_gradient_vjp.defvjp(clip_gradient_vjp_fwd, clip_gradient_vjp_bwd)


@custom_vjp
def straight_through_estimator(x: jnp.ndarray, threshold: float = 0.5) -> jnp.ndarray:
    """Straight-through estimator for binary activations.
    
    Forward: applies hard threshold
    Backward: passes gradients through unchanged
    
    Args:
        x: Input tensor
        threshold: Threshold for binarization
        
    Returns:
        Binarized output (0 or 1)
    """
    return (x > threshold).astype(jnp.float32)


def straight_through_estimator_fwd(x: jnp.ndarray, threshold: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward pass for straight-through estimator."""
    output = (x > threshold).astype(jnp.float32)
    return output, x


def straight_through_estimator_bwd(x: jnp.ndarray, g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    """Backward pass for straight-through estimator."""
    # Pass gradients through unchanged
    return g, None


straight_through_estimator.defvjp(straight_through_estimator_fwd, straight_through_estimator_bwd)


@custom_vjp
def custom_sqrt(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Square root with custom gradient to avoid NaN at x=0.
    
    Args:
        x: Input tensor (should be non-negative)
        eps: Small epsilon to stabilize gradient
        
    Returns:
        sqrt(x) with stable gradients
    """
    return jnp.sqrt(jnp.maximum(x, 0.0))


def custom_sqrt_fwd(x: jnp.ndarray, eps: float) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, float]]:
    """Forward pass for custom sqrt."""
    y = jnp.sqrt(jnp.maximum(x, 0.0))
    return y, (x, eps)


def custom_sqrt_bwd(res: Tuple, g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    """Backward pass for custom sqrt with stable gradient."""
    x, eps = res
    # Gradient of sqrt(x) is 1/(2*sqrt(x)), but we stabilize it
    grad_x = g / (2.0 * jnp.sqrt(jnp.maximum(x, eps)))
    return grad_x, None


custom_sqrt.defvjp(custom_sqrt_fwd, custom_sqrt_bwd)


@custom_vjp
def safe_log_vjp(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    """Logarithm with safe gradient computation.
    
    Args:
        x: Input tensor
        eps: Small epsilon to prevent log(0)
        
    Returns:
        log(max(x, eps))
    """
    return jnp.log(jnp.maximum(x, eps))


def safe_log_vjp_fwd(x: jnp.ndarray, eps: float) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, float]]:
    """Forward pass for safe log."""
    y = jnp.log(jnp.maximum(x, eps))
    return y, (x, eps)


def safe_log_vjp_bwd(res: Tuple, g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    """Backward pass for safe log."""
    x, eps = res
    # Gradient is 1/x, but we use max(x, eps) for stability
    grad_x = g / jnp.maximum(x, eps)
    return grad_x, None


safe_log_vjp.defvjp(safe_log_vjp_fwd, safe_log_vjp_bwd)


@custom_vjp
def huber_loss_vjp(x: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Huber loss with custom VJP for efficiency.
    
    Huber loss: 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
    
    Args:
        x: Input residuals
        delta: Threshold for switching between quadratic and linear
        
    Returns:
        Huber loss values
    """
    abs_x = jnp.abs(x)
    quadratic = 0.5 * x * x
    linear = delta * (abs_x - 0.5 * delta)
    return jnp.where(abs_x <= delta, quadratic, linear)


def huber_loss_vjp_fwd(x: jnp.ndarray, delta: float) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, float]]:
    """Forward pass for Huber loss."""
    abs_x = jnp.abs(x)
    quadratic = 0.5 * x * x
    linear = delta * (abs_x - 0.5 * delta)
    loss = jnp.where(abs_x <= delta, quadratic, linear)
    return loss, (x, delta)


def huber_loss_vjp_bwd(res: Tuple, g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    """Backward pass for Huber loss."""
    x, delta = res
    abs_x = jnp.abs(x)
    # Gradient: x if |x| <= delta, else delta * sign(x)
    grad_quadratic = x
    grad_linear = delta * jnp.sign(x)
    grad_x = jnp.where(abs_x <= delta, grad_quadratic, grad_linear)
    return g * grad_x, None


huber_loss_vjp.defvjp(huber_loss_vjp_fwd, huber_loss_vjp_bwd)


@custom_vjp
def gated_linear_unit_vjp(x: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Gated Linear Unit (GLU) with efficient custom VJP.
    
    GLU(x) = x[:d] * sigmoid(x[d:]) where x is split along axis
    
    Args:
        x: Input tensor
        axis: Axis along which to split and gate
        
    Returns:
        GLU output
    """
    d = x.shape[axis] // 2
    if axis == -1:
        a, b = x[..., :d], x[..., d:]
    else:
        a = jnp.take(x, jnp.arange(d), axis=axis)
        b = jnp.take(x, jnp.arange(d, 2*d), axis=axis)
    
    sigmoid_b = jax.nn.sigmoid(b)
    return a * sigmoid_b


def gated_linear_unit_vjp_fwd(x: jnp.ndarray, axis: int) -> Tuple[jnp.ndarray, Tuple]:
    """Forward pass for GLU."""
    d = x.shape[axis] // 2
    if axis == -1:
        a, b = x[..., :d], x[..., d:]
    else:
        a = jnp.take(x, jnp.arange(d), axis=axis)
        b = jnp.take(x, jnp.arange(d, 2*d), axis=axis)
    
    sigmoid_b = jax.nn.sigmoid(b)
    output = a * sigmoid_b
    return output, (a, b, sigmoid_b, axis, d)


def gated_linear_unit_vjp_bwd(res: Tuple, g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    """Backward pass for GLU."""
    a, b, sigmoid_b, axis, d = res
    
    # Gradient w.r.t. a: sigmoid(b)
    grad_a = g * sigmoid_b
    
    # Gradient w.r.t. b: a * sigmoid(b) * (1 - sigmoid(b))
    grad_b = g * a * sigmoid_b * (1 - sigmoid_b)
    
    # Concatenate gradients
    if axis == -1:
        grad_x = jnp.concatenate([grad_a, grad_b], axis=-1)
    else:
        grad_x = jnp.concatenate([grad_a, grad_b], axis=axis)
    
    return grad_x, None


gated_linear_unit_vjp.defvjp(gated_linear_unit_vjp_fwd, gated_linear_unit_vjp_bwd)


@custom_vjp
def swish_vjp(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray:
    """Swish activation with custom VJP.
    
    Swish(x) = x * sigmoid(beta * x)
    
    Args:
        x: Input tensor
        beta: Scaling factor for sigmoid
        
    Returns:
        Swish activation output
    """
    return x * jax.nn.sigmoid(beta * x)


def swish_vjp_fwd(x: jnp.ndarray, beta: float) -> Tuple[jnp.ndarray, Tuple]:
    """Forward pass for Swish."""
    sigmoid_beta_x = jax.nn.sigmoid(beta * x)
    output = x * sigmoid_beta_x
    return output, (x, beta, sigmoid_beta_x)


def swish_vjp_bwd(res: Tuple, g: jnp.ndarray) -> Tuple[jnp.ndarray, None]:
    """Backward pass for Swish."""
    x, beta, sigmoid_beta_x = res
    
    # Derivative: sigmoid(beta*x) + x * beta * sigmoid(beta*x) * (1 - sigmoid(beta*x))
    grad_x = sigmoid_beta_x + beta * x * sigmoid_beta_x * (1 - sigmoid_beta_x)
    
    return g * grad_x, None


swish_vjp.defvjp(swish_vjp_fwd, swish_vjp_bwd)