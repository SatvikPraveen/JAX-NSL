# File location: jax-nsl/src/linalg/solvers.py

"""
Iterative solvers: CG, gradient descent, Nesterov, LBFGS.

This module provides iterative algorithms for solving linear systems
and optimization problems with numerical stability.
"""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Optional, Tuple, NamedTuple
from collections import deque
import functools


class SolverState(NamedTuple):
    """State for iterative solvers."""
    x: jnp.ndarray
    residual: jnp.ndarray
    iteration: int
    converged: bool
    error: float


def conjugate_gradient(A: jnp.ndarray,
                      b: jnp.ndarray,
                      x0: Optional[jnp.ndarray] = None,
                      tolerance: float = 1e-6,
                      max_iterations: Optional[int] = None) -> Tuple[jnp.ndarray, SolverState]:
    """Conjugate gradient solver for symmetric positive definite systems.
    
    Args:
        A: Coefficient matrix (must be SPD)
        b: Right-hand side vector
        x0: Initial guess (defaults to zero)
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        
    Returns:
        (solution, final_state) tuple
    """
    n = A.shape[0]
    if x0 is None:
        x0 = jnp.zeros(n)
    if max_iterations is None:
        max_iterations = n
    
    def cg_step(state):
        x, r, p, iteration = state
        
        # Compute step size
        Ap = A @ p
        r_dot_r = jnp.dot(r, r)
        alpha = r_dot_r / jnp.dot(p, Ap)
        
        # Update solution and residual
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        
        # Compute conjugate direction
        beta = jnp.dot(r_new, r_new) / r_dot_r
        p_new = r_new + beta * p
        
        return x_new, r_new, p_new, iteration + 1
    
    def cg_cond(state):
        _, r, _, iteration = state
        error = jnp.linalg.norm(r)
        converged = error < tolerance
        max_iters = iteration >= max_iterations
        return jnp.logical_not(jnp.logical_or(converged, max_iters))
    
    # Initialize
    r0 = b - A @ x0
    p0 = r0
    initial_state = (x0, r0, p0, 0)
    
    # Run CG iterations
    final_state = lax.while_loop(cg_cond, cg_step, initial_state)
    x_final, r_final, _, iterations = final_state
    
    error = jnp.linalg.norm(r_final)
    converged = error < tolerance
    
    solver_state = SolverState(
        x=x_final,
        residual=r_final,
        iteration=iterations,
        converged=converged,
        error=error
    )
    
    return x_final, solver_state


def gradient_descent(objective_fn: Callable,
                    x0: jnp.ndarray,
                    learning_rate: float = 0.01,
                    tolerance: float = 1e-6,
                    max_iterations: int = 1000) -> Tuple[jnp.ndarray, SolverState]:
    """Gradient descent optimizer.
    
    Args:
        objective_fn: Function to minimize
        x0: Initial point
        learning_rate: Step size
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        
    Returns:
        (solution, final_state) tuple
    """
    grad_fn = jax.grad(objective_fn)
    
    def gd_step(state):
        x, grad, iteration = state
        x_new = x - learning_rate * grad
        grad_new = grad_fn(x_new)
        return x_new, grad_new, iteration + 1
    
    def gd_cond(state):
        _, grad, iteration = state
        grad_norm = jnp.linalg.norm(grad)
        converged = grad_norm < tolerance
        max_iters = iteration >= max_iterations
        return jnp.logical_not(jnp.logical_or(converged, max_iters))
    
    # Initialize
    grad0 = grad_fn(x0)
    initial_state = (x0, grad0, 0)
    
    # Run GD iterations
    x_final, grad_final, iterations = lax.while_loop(gd_cond, gd_step, initial_state)
    
    error = jnp.linalg.norm(grad_final)
    converged = error < tolerance
    
    solver_state = SolverState(
        x=x_final,
        residual=grad_final,
        iteration=iterations,
        converged=converged,
        error=error
    )
    
    return x_final, solver_state


def nesterov_momentum(objective_fn: Callable,
                     x0: jnp.ndarray,
                     learning_rate: float = 0.01,
                     momentum: float = 0.9,
                     tolerance: float = 1e-6,
                     max_iterations: int = 1000) -> Tuple[jnp.ndarray, SolverState]:
    """Nesterov accelerated gradient descent.
    
    Args:
        objective_fn: Function to minimize
        x0: Initial point
        learning_rate: Step size
        momentum: Momentum coefficient
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        
    Returns:
        (solution, final_state) tuple
    """
    grad_fn = jax.grad(objective_fn)
    
    def nesterov_step(state):
        x, v, iteration = state
        
        # Look-ahead point
        x_lookahead = x + momentum * v
        grad_lookahead = grad_fn(x_lookahead)
        
        # Update velocity and position
        v_new = momentum * v - learning_rate * grad_lookahead
        x_new = x + v_new
        
        return x_new, v_new, iteration + 1
    
    def nesterov_cond(state):
        x, v, iteration = state
        grad_current = grad_fn(x)
        grad_norm = jnp.linalg.norm(grad_current)
        converged = grad_norm < tolerance
        max_iters = iteration >= max_iterations
        return jnp.logical_not(jnp.logical_or(converged, max_iters))
    
    # Initialize
    v0 = jnp.zeros_like(x0)
    initial_state = (x0, v0, 0)
    
    # Run Nesterov iterations
    x_final, _, iterations = lax.while_loop(nesterov_cond, nesterov_step, initial_state)
    
    grad_final = grad_fn(x_final)
    error = jnp.linalg.norm(grad_final)
    converged = error < tolerance
    
    solver_state = SolverState(
        x=x_final,
        residual=grad_final,
        iteration=iterations,
        converged=converged,
        error=error
    )
    
    return x_final, solver_state


def lbfgs_solver(objective_fn: Callable,
                x0: jnp.ndarray,
                memory_size: int = 10,
                tolerance: float = 1e-6,
                max_iterations: int = 1000,
                line_search_steps: int = 20) -> Tuple[jnp.ndarray, SolverState]:
    """Limited-memory BFGS optimization.
    
    Args:
        objective_fn: Function to minimize
        x0: Initial point
        memory_size: Number of previous iterations to remember
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        line_search_steps: Steps for line search
        
    Returns:
        (solution, final_state) tuple
    """
    grad_fn = jax.grad(objective_fn)
    
    def lbfgs_direction(grad, s_history, y_history, rho_history):
        """Compute L-BFGS search direction."""
        q = grad
        alpha_history = []
        
        # First loop (backward)
        for i in range(len(s_history) - 1, -1, -1):
            rho_i = rho_history[i]
            alpha_i = rho_i * jnp.dot(s_history[i], q)
            q = q - alpha_i * y_history[i]
            alpha_history.append(alpha_i)
        
        alpha_history = alpha_history[::-1]  # Reverse for second loop
        
        # Initial Hessian approximation (identity)
        r = q
        
        # Second loop (forward)
        for i in range(len(s_history)):
            beta = rho_history[i] * jnp.dot(y_history[i], r)
            r = r + s_history[i] * (alpha_history[i] - beta)
        
        return -r  # Negative for descent direction
    
    def backtrack_line_search(x, direction, grad, f_val):
        """Simple backtracking line search."""
        alpha = 1.0
        c1 = 1e-4  # Armijo constant
        
        for _ in range(line_search_steps):
            x_new = x + alpha * direction
            f_new = objective_fn(x_new)
            
            # Armijo condition
            if f_new <= f_val + c1 * alpha * jnp.dot(grad, direction):
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    # Initialize history
    s_history = []
    y_history = []
    rho_history = []
    
    x = x0
    grad = grad_fn(x)
    f_val = objective_fn(x)
    
    for iteration in range(max_iterations):
        if jnp.linalg.norm(grad) < tolerance:
            break
        
        # Compute search direction
        if len(s_history) == 0:
            direction = -grad  # First iteration: steepest descent
        else:
            direction = lbfgs_direction(grad, s_history, y_history, rho_history)
        
        # Line search
        step_size = backtrack_line_search(x, direction, grad, f_val)
        
        # Update
        x_new = x + step_size * direction
        grad_new = grad_fn(x_new)
        f_new = objective_fn(x_new)
        
        # Update history
        s = x_new - x
        y = grad_new - grad
        rho = 1.0 / jnp.dot(y, s)
        
        if len(s_history) >= memory_size:
            s_history.pop(0)
            y_history.pop(0)
            rho_history.pop(0)
        
        s_history.append(s)
        y_history.append(y)
        rho_history.append(rho)
        
        x, grad, f_val = x_new, grad_new, f_new
    
    error = jnp.linalg.norm(grad)
    converged = error < tolerance
    
    solver_state = SolverState(
        x=x,
        residual=grad,
        iteration=iteration + 1,
        converged=converged,
        error=error
    )
    
    return x, solver_state


def linear_solve_iterative(A: jnp.ndarray,
                          b: jnp.ndarray,
                          method: str = 'cg',
                          **kwargs) -> Tuple[jnp.ndarray, SolverState]:
    """Unified interface for iterative linear solvers.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side
        method: Solver method ('cg', 'gmres', 'jacobi')
        **kwargs: Additional solver parameters
        
    Returns:
        (solution, solver_state) tuple
    """
    if method == 'cg':
        return conjugate_gradient(A, b, **kwargs)
    elif method == 'jacobi':
        return jacobi_method(A, b, **kwargs)
    else:
        raise ValueError(f"Unknown solver method: {method}")


def jacobi_method(A: jnp.ndarray,
                 b: jnp.ndarray,
                 x0: Optional[jnp.ndarray] = None,
                 tolerance: float = 1e-6,
                 max_iterations: int = 1000) -> Tuple[jnp.ndarray, SolverState]:
    """Jacobi iterative method for linear systems.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        x0: Initial guess
        tolerance: Convergence tolerance
        max_iterations: Maximum iterations
        
    Returns:
        (solution, solver_state) tuple
    """
    n = A.shape[0]
    if x0 is None:
        x0 = jnp.zeros(n)
    
    # Extract diagonal and off-diagonal parts
    D = jnp.diag(jnp.diag(A))
    R = A - D
    D_inv = jnp.diag(1.0 / jnp.diag(A))
    
    def jacobi_step(state):
        x, iteration = state
        x_new = D_inv @ (b - R @ x)
        return x_new, iteration + 1
    
    def jacobi_cond(state):
        x, iteration = state
        residual = b - A @ x
        error = jnp.linalg.norm(residual)
        converged = error < tolerance
        max_iters = iteration >= max_iterations
        return jnp.logical_not(jnp.logical_or(converged, max_iters))
    
    initial_state = (x0, 0)
    x_final, iterations = lax.while_loop(jacobi_cond, jacobi_step, initial_state)
    
    residual = b - A @ x_final
    error = jnp.linalg.norm(residual)
    converged = error < tolerance
    
    solver_state = SolverState(
        x=x_final,
        residual=residual,
        iteration=iterations,
        converged=converged,
        error=error
    )
    
    return x_final, solver_state


def least_squares_solver(A: jnp.ndarray,
                        b: jnp.ndarray,
                        regularization: float = 0.0) -> jnp.ndarray:
    """Solve least squares problem min ||Ax - b||^2 + λ||x||^2.
    
    Args:
        A: Design matrix
        b: Target vector
        regularization: L2 regularization parameter
        
    Returns:
        Least squares solution
    """
    m, n = A.shape
    
    if m >= n:  # Overdetermined system
        # Normal equations: (A^T A + λI) x = A^T b
        AtA = A.T @ A
        if regularization > 0:
            AtA = AtA + regularization * jnp.eye(n)
        Atb = A.T @ b
        return jnp.linalg.solve(AtA, Atb)
    else:  # Underdetermined system
        # Use SVD for minimum norm solution
        u, s, vt = jnp.linalg.svd(A, full_matrices=False)
        s_reg = s / (s**2 + regularization)
        return vt.T @ (s_reg * (u.T @ b))


def eigenvalue_power_method(A: jnp.ndarray,
                          max_iterations: int = 100,
                          tolerance: float = 1e-6) -> Tuple[float, jnp.ndarray]:
    """Power method for dominant eigenvalue and eigenvector.
    
    Args:
        A: Square matrix
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        (eigenvalue, eigenvector) tuple
    """
    n = A.shape[0]
    v = jnp.ones(n)
    v = v / jnp.linalg.norm(v)
    
    eigenvalue = 0.0
    
    for i in range(max_iterations):
        Av = A @ v
        eigenvalue_new = jnp.dot(v, Av)
        v_new = Av / jnp.linalg.norm(Av)
        
        if jnp.abs(eigenvalue_new - eigenvalue) < tolerance:
            break
        
        eigenvalue = eigenvalue_new
        v = v_new
    
    return eigenvalue, v


def lanczos_algorithm(A: jnp.ndarray,
                     num_iterations: int,
                     starting_vector: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Lanczos algorithm for tridiagonalization.
    
    Args:
        A: Symmetric matrix
        num_iterations: Number of Lanczos iterations
        starting_vector: Initial vector (random if None)
        
    Returns:
        (tridiagonal_matrix, Q_matrix) tuple
    """
    n = A.shape[0]
    
    if starting_vector is None:
        q = jax.random.normal(jax.random.PRNGKey(0), (n,))
        q = q / jnp.linalg.norm(q)
    else:
        q = starting_vector / jnp.linalg.norm(starting_vector)
    
    Q = jnp.zeros((n, num_iterations))
    Q = Q.at[:, 0].set(q)
    
    alpha = jnp.zeros(num_iterations)
    beta = jnp.zeros(num_iterations - 1)
    
    for j in range(num_iterations):
        v = A @ Q[:, j]
        
        if j > 0:
            v = v - beta[j-1] * Q[:, j-1]
        
        alpha = alpha.at[j].set(jnp.dot(Q[:, j], v))
        v = v - alpha[j] * Q[:, j]
        
        if j < num_iterations - 1:
            beta = beta.at[j].set(jnp.linalg.norm(v))
            if beta[j] > 0:
                Q = Q.at[:, j+1].set(v / beta[j])
    
    # Construct tridiagonal matrix
    T = jnp.diag(alpha) + jnp.diag(beta, 1) + jnp.diag(beta, -1)
    
    return T, Q