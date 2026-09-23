# File location: src/jax_nsl/linalg/__init__.py

"""
Linear algebra: decompositions, norms, and jittable iterative solvers.
"""

from .ops import (
    batched_matmul,
    cholesky_safe,
    condition_number,
    einsum_path_optimize,
    frobenius_norm,
    gram_schmidt,
    matrix_logarithm,
    matrix_power,
    matrix_sqrt,
    pseudoinverse_stable,
    qr_decomposition,
    safe_matmul,
    spectral_norm,
    stable_eigh,
    stable_svd,
    trace_product,
)
from .solvers import (
    SolverState,
    conjugate_gradient,
    eigenvalue_power_method,
    gradient_descent,
    jacobi_method,
    lanczos_algorithm,
    lbfgs_solver,
    least_squares_solver,
    linear_solve_iterative,
    nesterov_momentum,
)

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
    "matrix_sqrt",
    "matrix_logarithm",
    "pseudoinverse_stable",
    "gram_schmidt",
    "trace_product",
    "frobenius_norm",
    "spectral_norm",
    "condition_number",
    # solvers.py
    "SolverState",
    "conjugate_gradient",
    "jacobi_method",
    "linear_solve_iterative",
    "least_squares_solver",
    "gradient_descent",
    "nesterov_momentum",
    "lbfgs_solver",
    "eigenvalue_power_method",
    "lanczos_algorithm",
]
