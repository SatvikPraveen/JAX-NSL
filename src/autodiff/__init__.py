# File location: jax-nsl/src/autodiff/__init__.py

"""
Automatic differentiation utilities and custom derivatives.

This module provides utilities for working with JAX's automatic differentiation,
including gradient, Jacobian, and Hessian computations, as well as custom VJP/JVP
implementations for specialized operations.
"""

from .grad_jac_hess import *
from .custom_vjp import *
from .custom_jvp import *

__all__ = [
    # grad_jac_hess.py
    "safe_grad",
    "safe_jacobian", 
    "safe_hessian",
    "grad_and_value",
    "jacobian_and_value",
    "hessian_and_value",
    "batch_jacobian",
    "batch_hessian",
    "directional_derivative",
    "finite_diff_grad",
    
    # custom_vjp.py
    "clip_gradient_vjp",
    "straight_through_estimator",
    "custom_sqrt",
    "safe_log_vjp",
    "huber_loss_vjp",
    
    # custom_jvp.py  
    "clip_gradient_jvp",
    "custom_sqrt_jvp",
    "piecewise_linear_jvp",
    "saturated_activation_jvp"
]