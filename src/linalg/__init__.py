# File location: jax-nsl/src/linalg/__init__.py

"""
Linear algebra operations and iterative solvers.

This module provides matrix operations, decompositions, and iterative
algorithms for solving linear systems and optimization problems.
"""

from .ops import *
from .solvers import *

__all__ = [
    # ops.py
    "safe_matmul",
    "batched_matmul",
    "einsum_path_optimize",
    "stable_svd",
    "stable_eigh",
    "qr_decomposition",
    "cholesky_safe",
    "matrix_power",
    "trace_product",
    "frobenius_norm",
    "spectral_norm",
    "condition_number",
    
    # solvers.py  
    "conjugate_gradient",
    "gradient_descent",
    "nesterov_momentum",
    "lbfgs_solver",
    "linear_solve_iterative",
    "least_squares_solver",
    "eigenvalue_power_method",
    "lanczos_algorithm"
]