# File location: jax-nsl/src/autodiff/grad_jac_hess.py

"""
Gradient, Jacobian, and Hessian computations with safety and batching.

This module provides robust implementations of automatic differentiation
operations with error handling and numerical stability considerations.
"""

import jax
import jax.numpy as jnp
from jax import grad, jacobian, hessian
from typing import Callable, Any, Optional, Union, Tuple
import functools


def safe_grad(fun: Callable, 
              argnums: Union[int, Tuple[int, ...]] = 0,
              has_aux: bool = False,
              holomorphic: bool = False,
              allow_int: bool = False) -> Callable:
    """Safe gradient computation with better error handling.
    
    Args:
        fun: Function to differentiate
        argnums: Arguments to differentiate with respect to
        has_aux: Whether function returns auxiliary data
        holomorphic: Whether function is holomorphic
        allow_int: Whether to allow integer inputs
        
    Returns:
        Gradient function with error handling
    """
    grad_fun = grad(fun, argnums=argnums, has_aux=has_aux, 
                   holomorphic=holomorphic, allow_int=allow_int)
    
    @functools.wraps(grad_fun)
    def wrapped_grad(*args, **kwargs):
        try:
            return grad_fun(*args, **kwargs)
        except Exception as e:
            print(f"Gradient computation failed: {e}")
            if has_aux:
                # Return zero gradients and None auxiliary data
                if isinstance(argnums, int):
                    zero_grad = jnp.zeros_like(args[argnums])
                else:
                    zero_grad = tuple(jnp.zeros_like(args[i]) for i in argnums)
                return zero_grad, None
            else:
                if isinstance(argnums, int):
                    return jnp.zeros_like(args[argnums])
                else:
                    return tuple(jnp.zeros_like(args[i]) for i in argnums)
    
    return wrapped_grad


def safe_jacobian(fun: Callable,
                 argnums: Union[int, Tuple[int, ...]] = 0,
                 holomorphic: bool = False,
                 allow_int: bool = False) -> Callable:
    """Safe Jacobian computation with error handling.
    
    Args:
        fun: Function to differentiate
        argnums: Arguments to differentiate with respect to
        holomorphic: Whether function is holomorphic
        allow_int: Whether to allow integer inputs
        
    Returns:
        Jacobian function with error handling
    """
    jac_fun = jacobian(fun, argnums=argnums, holomorphic=holomorphic, allow_int=allow_int)
    
    @functools.wraps(jac_fun)
    def wrapped_jacobian(*args, **kwargs):
        try:
            return jac_fun(*args, **kwargs)
        except Exception as e:
            print(f"Jacobian computation failed: {e}")
            # Estimate output shape by running function once
            output = fun(*args, **kwargs)
            if isinstance(argnums, int):
                input_shape = args[argnums].shape
                return jnp.zeros(output.shape + input_shape, dtype=output.dtype)
            else:
                # Multiple arguments case is more complex
                return None
    
    return wrapped_jacobian


def safe_hessian(fun: Callable,
                argnums: Union[int, Tuple[int, ...]] = 0,
                holomorphic: bool = False) -> Callable:
    """Safe Hessian computation with error handling.
    
    Args:
        fun: Scalar-valued function to differentiate
        argnums: Arguments to differentiate with respect to
        holomorphic: Whether function is holomorphic
        
    Returns:
        Hessian function with error handling
    """
    hess_fun = hessian(fun, argnums=argnums, holomorphic=holomorphic)
    
    @functools.wraps(hess_fun)
    def wrapped_hessian(*args, **kwargs):
        try:
            return hess_fun(*args, **kwargs)
        except Exception as e:
            print(f"Hessian computation failed: {e}")
            if isinstance(argnums, int):
                input_shape = args[argnums].shape
                return jnp.zeros(input_shape + input_shape, dtype=args[argnums].dtype)
            else:
                return None
    
    return wrapped_hessian


def grad_and_value(fun: Callable,
                  argnums: Union[int, Tuple[int, ...]] = 0,
                  has_aux: bool = False) -> Callable:
    """Compute gradient and function value simultaneously.
    
    More efficient than computing them separately.
    
    Args:
        fun: Function to differentiate
        argnums: Arguments to differentiate with respect to
        has_aux: Whether function returns auxiliary data
        
    Returns:
        Function that returns (gradient, value) tuple
    """
    @functools.wraps(fun)
    def grad_and_val_fun(*args, **kwargs):
        def value_fun(*args, **kwargs):
            if has_aux:
                value, aux = fun(*args, **kwargs)
                return value
            else:
                return fun(*args, **kwargs)
        
        grad_fun = grad(value_fun, argnums=argnums)
        
        if has_aux:
            value, aux = fun(*args, **kwargs)
            grad_val = grad_fun(*args, **kwargs)
            return grad_val, value, aux
        else:
            value = fun(*args, **kwargs)
            grad_val = grad_fun(*args, **kwargs)
            return grad_val, value
    
    return grad_and_val_fun


def jacobian_and_value(fun: Callable,
                      argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """Compute Jacobian and function value simultaneously.
    
    Args:
        fun: Function to differentiate
        argnums: Arguments to differentiate with respect to
        
    Returns:
        Function that returns (jacobian, value) tuple
    """
    jac_fun = jacobian(fun, argnums=argnums)
    
    @functools.wraps(fun)
    def jac_and_val_fun(*args, **kwargs):
        value = fun(*args, **kwargs)
        jac_val = jac_fun(*args, **kwargs)
        return jac_val, value
    
    return jac_and_val_fun


def hessian_and_value(fun: Callable,
                     argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """Compute Hessian and function value simultaneously.
    
    Args:
        fun: Scalar-valued function to differentiate
        argnums: Arguments to differentiate with respect to
        
    Returns:
        Function that returns (hessian, value) tuple
    """
    hess_fun = hessian(fun, argnums=argnums)
    
    @functools.wraps(fun)
    def hess_and_val_fun(*args, **kwargs):
        value = fun(*args, **kwargs)
        hess_val = hess_fun(*args, **kwargs)
        return hess_val, value
    
    return hess_and_val_fun


def batch_jacobian(fun: Callable,
                  argnums: int = 0) -> Callable:
    """Compute Jacobian for batched inputs efficiently.
    
    Uses vmap to compute Jacobians for a batch of inputs.
    
    Args:
        fun: Function to differentiate (should handle single examples)
        argnums: Argument to differentiate with respect to
        
    Returns:
        Function that computes batched Jacobians
    """
    jac_fun = jacobian(fun, argnums=argnums)
    batched_jac = jax.vmap(jac_fun)
    return batched_jac


def batch_hessian(fun: Callable,
                 argnums: int = 0) -> Callable:
    """Compute Hessian for batched inputs efficiently.
    
    Args:
        fun: Scalar-valued function (should handle single examples)
        argnums: Argument to differentiate with respect to
        
    Returns:
        Function that computes batched Hessians
    """
    hess_fun = hessian(fun, argnums=argnums)
    batched_hess = jax.vmap(hess_fun)
    return batched_hess


def directional_derivative(fun: Callable,
                          x: jnp.ndarray,
                          v: jnp.ndarray) -> jnp.ndarray:
    """Compute directional derivative of function at point x in direction v.
    
    Uses JVP (Jacobian-vector product) for efficiency.
    
    Args:
        fun: Function to differentiate
        x: Point at which to compute derivative
        v: Direction vector
        
    Returns:
        Directional derivative fun'(x) · v
    """
    _, jvp_result = jax.jvp(fun, (x,), (v,))
    return jvp_result


def finite_diff_grad(fun: Callable,
                    x: jnp.ndarray,
                    eps: float = 1e-5,
                    method: str = 'central') -> jnp.ndarray:
    """Compute gradient using finite differences (for testing/debugging).
    
    Args:
        fun: Scalar-valued function
        x: Point at which to compute gradient
        eps: Step size for finite differences
        method: 'forward', 'backward', or 'central'
        
    Returns:
        Finite difference approximation of gradient
    """
    f0 = fun(x)
    grad_approx = jnp.zeros_like(x)
    
    def compute_partial(i):
        ei = jnp.zeros_like(x)
        ei = ei.at[i].set(1.0)
        
        if method == 'forward':
            return (fun(x + eps * ei) - f0) / eps
        elif method == 'backward':
            return (f0 - fun(x - eps * ei)) / eps
        elif method == 'central':
            return (fun(x + eps * ei) - fun(x - eps * ei)) / (2 * eps)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Vectorize over all dimensions
    grad_approx = jax.vmap(compute_partial)(jnp.arange(x.size))
    return grad_approx.reshape(x.shape)


def gradient_checker(fun: Callable,
                    x: jnp.ndarray,
                    eps: float = 1e-5,
                    rtol: float = 1e-3,
                    atol: float = 1e-6) -> Tuple[bool, float]:
    """Check gradient computation against finite differences.
    
    Args:
        fun: Function to test
        x: Input point
        eps: Finite difference step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        (is_correct, max_error) tuple
    """
    # Compute analytical gradient
    grad_analytical = grad(fun)(x)
    
    # Compute finite difference gradient
    grad_fd = finite_diff_grad(fun, x, eps=eps, method='central')
    
    # Compute error
    error = jnp.abs(grad_analytical - grad_fd)
    max_error = jnp.max(error)
    
    # Check tolerance
    is_close = jnp.allclose(grad_analytical, grad_fd, rtol=rtol, atol=atol)
    
    return bool(is_close), float(max_error)