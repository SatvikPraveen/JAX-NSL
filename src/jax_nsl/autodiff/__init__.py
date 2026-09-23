# File location: src/jax_nsl/autodiff/__init__.py

"""
Automatic differentiation: derivative utilities, custom VJP/JVP rules, and
implicit differentiation through solvers.
"""

from .custom_jvp import (
    clip_gradient_jvp,
    custom_sqrt_jvp,
    gaussian_activation_jvp,
    learnable_activation_jvp,
    piecewise_linear_jvp,
    saturated_activation_jvp,
    smooth_abs_jvp,
    soft_sign_jvp,
)
from .custom_vjp import (
    clip_gradient_vjp,
    custom_sqrt,
    custom_sqrt_vjp,
    gated_linear_unit_vjp,
    gradient_reversal,
    huber_loss_vjp,
    safe_log_vjp,
    smooth_abs_vjp,
    ste_round,
    straight_through_estimator,
    swish_vjp,
)
from .grad_jac_hess import (
    auto_jacobian,
    batch_hessian,
    batch_jacobian,
    checked_grad,
    compute_gradient,
    compute_hessian,
    compute_jacobian,
    directional_derivative,
    finite_diff_grad,
    gauss_newton_vp,
    grad_and_value,
    gradient_check_report,
    gradient_checker,
    hessian_and_value,
    hessian_diagonal,
    hessian_trace_hutchinson,
    hvp,
    hvp_reverse_over_reverse,
    jacobian_and_value,
    safe_grad,
    safe_hessian,
    safe_jacobian,
)
from .implicit import fixed_point, fixed_point_unrolled, implicit_newton_solve

__all__ = [
    # grad_jac_hess.py
    "safe_grad", "checked_grad", "safe_jacobian", "safe_hessian", "grad_and_value",
    "jacobian_and_value", "hessian_and_value", "auto_jacobian", "batch_jacobian",
    "batch_hessian", "directional_derivative", "hvp", "hvp_reverse_over_reverse",
    "gauss_newton_vp", "hessian_diagonal", "hessian_trace_hutchinson", "finite_diff_grad",
    "gradient_checker", "gradient_check_report", "compute_gradient", "compute_jacobian",
    "compute_hessian",
    # custom_vjp.py
    "clip_gradient_vjp", "straight_through_estimator", "ste_round", "gradient_reversal",
    "custom_sqrt", "custom_sqrt_vjp", "safe_log_vjp", "huber_loss_vjp",
    "gated_linear_unit_vjp", "swish_vjp", "smooth_abs_vjp",
    # custom_jvp.py
    "clip_gradient_jvp", "custom_sqrt_jvp", "piecewise_linear_jvp", "saturated_activation_jvp",
    "soft_sign_jvp", "gaussian_activation_jvp", "learnable_activation_jvp", "smooth_abs_jvp",
    # implicit.py
    "fixed_point", "fixed_point_unrolled", "implicit_newton_solve",
]
