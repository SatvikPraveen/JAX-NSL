# File location: src/jax_nsl/linalg/solvers.py

"""
Iterative solvers written as JAX control flow so they can be ``jit``-ed,
``vmap``-ed and differentiated through.

* :func:`conjugate_gradient` - (preconditioned) CG for SPD systems, matrix-free.
* :func:`jacobi_method` - classic stationary iteration.
* :func:`gradient_descent`, :func:`nesterov_momentum` - first-order optimisers.
* :func:`lbfgs_solver` - limited-memory BFGS with a fixed-size circular history
  and Armijo backtracking, entirely inside ``lax.while_loop``.
* :func:`eigenvalue_power_method`, :func:`lanczos_algorithm` - Krylov methods.

All solvers return ``(solution, SolverState)`` where ``SolverState`` carries
the final residual, iteration count and a convergence flag.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array
LinearOperator = Array | Callable[[Array], Array]


class SolverState(NamedTuple):
    """Final state of an iterative solver."""

    x: Array
    residual: Array
    iteration: Array
    converged: Array
    error: Array


def _as_matvec(A: LinearOperator) -> Callable[[Array], Array]:
    """Accept a dense matrix or a callable ``v -> A @ v``."""
    if callable(A):
        return A
    return lambda v: A @ v


# ---------------------------------------------------------------------------
# Linear systems
# ---------------------------------------------------------------------------


def _build_minv(preconditioner: LinearOperator | None, b: Array):
    """Return ``(minv_fn, consts)`` with ``minv_fn(r, *consts)``."""
    if preconditioner is None:
        return (lambda r: r), ()
    if callable(preconditioner):
        return jax.closure_convert(preconditioner, b)
    if preconditioner.ndim == 1:
        return (lambda r, d: d * r), (preconditioner,)
    return (lambda r, M: M @ r), (preconditioner,)


def _cg_loop(matvec, minv, tolerance, max_iterations, b, x0, a_consts, m_consts):
    b_norm = jnp.maximum(jnp.linalg.norm(b), jnp.finfo(b.dtype).tiny)
    threshold = tolerance * b_norm

    def cond(state):
        _, r, _, _, k = state
        return jnp.logical_and(jnp.linalg.norm(r) > threshold, k < max_iterations)

    def body(state):
        x, r, z, p, k = state
        Ap = matvec(p, *a_consts)
        rz = jnp.vdot(r, z)
        alpha = rz / jnp.vdot(p, Ap)
        x_new = x + alpha * p
        r_new = r - alpha * Ap
        z_new = minv(r_new, *m_consts)
        beta = jnp.vdot(r_new, z_new) / rz
        return x_new, r_new, z_new, z_new + beta * p, k + 1

    r0 = b - matvec(x0, *a_consts)
    z0 = minv(r0, *m_consts)
    x, r, _, _, k = lax.while_loop(cond, body, (x0, r0, z0, z0, jnp.int32(0)))
    err = jnp.linalg.norm(r)
    return x, SolverState(x=x, residual=r, iteration=k, converged=err <= threshold, error=err)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3))
def _cg(matvec, minv, tolerance, max_iterations, b, x0, a_consts, m_consts):
    return _cg_loop(matvec, minv, tolerance, max_iterations, b, x0, a_consts, m_consts)


def _cg_fwd(matvec, minv, tolerance, max_iterations, b, x0, a_consts, m_consts):
    out = _cg_loop(matvec, minv, tolerance, max_iterations, b, x0, a_consts, m_consts)
    return out, (out[0], a_consts, m_consts)


def _cg_bwd(matvec, minv, tolerance, max_iterations, residuals, cotangents):
    """Implicit differentiation of ``x = A^{-1} b``.

    ``while_loop`` cannot be reverse-differentiated (the trip count is
    dynamic), and unrolling would be wasteful anyway.  Instead we use the
    implicit function theorem: with ``A x = b``,

        dL/db = A^{-T} dL/dx =: lam   (one more CG solve, A is symmetric)
        dL/dA = -lam x^T             (obtained as a VJP through the matvec)

    so the gradient costs one extra linear solve regardless of how many CG
    iterations the forward pass took.
    """
    x, a_consts, m_consts = residuals
    g_x = cotangents[0]
    lam, _ = _cg_loop(
        matvec, minv, tolerance, max_iterations, g_x, jnp.zeros_like(g_x), a_consts, m_consts
    )
    # dL/dconsts = vjp of (consts -> A(consts) x) at cotangent -lam.
    _, vjp_fn = jax.vjp(lambda c: matvec(x, *c), a_consts)
    (g_a_consts,) = vjp_fn(-lam)
    g_m_consts = jax.tree_util.tree_map(jnp.zeros_like, m_consts)
    return lam, jnp.zeros_like(x), g_a_consts, g_m_consts


_cg.defvjp(_cg_fwd, _cg_bwd)


def conjugate_gradient(
    A: LinearOperator,
    b: Array,
    x0: Array | None = None,
    tolerance: float = 1e-6,
    max_iterations: int | None = None,
    preconditioner: LinearOperator | None = None,
) -> tuple[Array, SolverState]:
    """Preconditioned conjugate gradient for symmetric positive-definite ``A``.

    The solver is ``jit``-able and *reverse-mode differentiable* with respect
    to ``b`` and to ``A`` (dense entries, or arrays closed over by a matrix-free
    ``A``) via implicit differentiation - see :func:`_cg_bwd`.

    Args:
        A: Dense SPD matrix or a matrix-free callable ``v -> A @ v``.
        b: Right-hand side.
        x0: Initial guess (zeros by default).
        tolerance: Stop when ``||r|| <= tolerance * ||b||``.
        max_iterations: Defaults to ``len(b)`` (CG terminates in ``n`` steps in
            exact arithmetic; in float32 badly conditioned systems need more).
        preconditioner: Approximation of ``A^{-1}``: a 1-D array is treated as
            the diagonal of a Jacobi preconditioner, a 2-D array as a dense
            matrix, a callable as ``v -> M^{-1} v``.  Preconditioning replaces
            the condition number ``kappa(A)`` by ``kappa(M^{-1} A)``; CG's error
            contracts by roughly ``(sqrt(kappa) - 1) / (sqrt(kappa) + 1)`` per step.

    Returns:
        ``(x, SolverState)``.
    """
    if x0 is None:
        x0 = jnp.zeros_like(b)
    if max_iterations is None:
        max_iterations = b.shape[0]
    if callable(A):
        matvec, a_consts = jax.closure_convert(A, b)
        a_consts = tuple(a_consts)
    else:
        matvec, a_consts = (lambda v, M: M @ v), (A,)
    minv, m_consts = _build_minv(preconditioner, b)
    return _cg(
        matvec, minv, float(tolerance), int(max_iterations), b, x0, a_consts, tuple(m_consts)
    )


def jacobi_method(
    A: Array, b: Array, x0: Array | None = None, tolerance: float = 1e-6, max_iterations: int = 1000
) -> tuple[Array, SolverState]:
    """Jacobi iteration ``x <- D^{-1}(b - R x)``; converges for diagonally dominant ``A``."""
    if x0 is None:
        x0 = jnp.zeros_like(b)
    d_inv = 1.0 / jnp.diag(A)
    R = A - jnp.diag(jnp.diag(A))
    threshold = tolerance * jnp.maximum(jnp.linalg.norm(b), jnp.finfo(b.dtype).tiny)

    def cond(state):
        x, k = state
        return jnp.logical_and(jnp.linalg.norm(b - A @ x) > threshold, k < max_iterations)

    def body(state):
        x, k = state
        return d_inv * (b - R @ x), k + 1

    x, k = lax.while_loop(cond, body, (x0, jnp.int32(0)))
    r = b - A @ x
    err = jnp.linalg.norm(r)
    return x, SolverState(x=x, residual=r, iteration=k, converged=err <= threshold, error=err)


def linear_solve_iterative(
    A: Array, b: Array, method: str = "cg", **kwargs
) -> tuple[Array, SolverState]:
    """Dispatch to :func:`conjugate_gradient` (``'cg'``) or :func:`jacobi_method` (``'jacobi'``)."""
    if method == "cg":
        return conjugate_gradient(A, b, **kwargs)
    if method == "jacobi":
        return jacobi_method(A, b, **kwargs)
    raise ValueError(f"Unknown solver method: {method}")


def least_squares_solver(A: Array, b: Array, regularization: float = 0.0) -> Array:
    """Minimise ``||Ax - b||^2 + lambda ||x||^2``.

    * ``lambda = 0``, tall ``A``: QR (``R x = Q^T b``), which avoids squaring
      the condition number the way the normal equations do.
    * ``lambda > 0``: Tikhonov via the normal equations ``(A^T A + lambda I) x = A^T b``.
    * Wide ``A``: minimum-norm solution through the SVD.
    """
    m, n = A.shape
    if m >= n and regularization == 0.0:
        q, r = jnp.linalg.qr(A, mode="reduced")
        return jax.scipy.linalg.solve_triangular(r, q.T @ b, lower=False)
    if m >= n:
        AtA = A.T @ A + regularization * jnp.eye(n, dtype=A.dtype)
        return jnp.linalg.solve(AtA, A.T @ b)
    u, s, vt = jnp.linalg.svd(A, full_matrices=False)
    s_reg = s / (s**2 + regularization)
    return vt.T @ (s_reg * (u.T @ b))


# ---------------------------------------------------------------------------
# Unconstrained minimisation
# ---------------------------------------------------------------------------


def gradient_descent(
    objective_fn: Callable[[Array], Array],
    x0: Array,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[Array, SolverState]:
    """Plain gradient descent until ``||grad|| < tolerance`` or ``max_iterations``."""
    grad_fn = jax.grad(objective_fn)

    def cond(state):
        _, g, k = state
        return jnp.logical_and(jnp.linalg.norm(g) >= tolerance, k < max_iterations)

    def body(state):
        x, g, k = state
        x_new = x - learning_rate * g
        return x_new, grad_fn(x_new), k + 1

    x, g, k = lax.while_loop(cond, body, (x0, grad_fn(x0), jnp.int32(0)))
    err = jnp.linalg.norm(g)
    return x, SolverState(x=x, residual=g, iteration=k, converged=err < tolerance, error=err)


def nesterov_momentum(
    objective_fn: Callable[[Array], Array],
    x0: Array,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
) -> tuple[Array, SolverState]:
    """Nesterov accelerated gradient: gradient evaluated at the look-ahead point."""
    grad_fn = jax.grad(objective_fn)

    def cond(state):
        x, _, k = state
        return jnp.logical_and(jnp.linalg.norm(grad_fn(x)) >= tolerance, k < max_iterations)

    def body(state):
        x, v, k = state
        g = grad_fn(x + momentum * v)
        v_new = momentum * v - learning_rate * g
        return x + v_new, v_new, k + 1

    x, _, k = lax.while_loop(cond, body, (x0, jnp.zeros_like(x0), jnp.int32(0)))
    g = grad_fn(x)
    err = jnp.linalg.norm(g)
    return x, SolverState(x=x, residual=g, iteration=k, converged=err < tolerance, error=err)


def lbfgs_solver(
    objective_fn: Callable[[Array], Array],
    x0: Array,
    memory_size: int = 10,
    tolerance: float = 1e-6,
    max_iterations: int = 1000,
    line_search_steps: int = 20,
    armijo_c1: float = 1e-4,
) -> tuple[Array, SolverState]:
    """Limited-memory BFGS, fully expressed in ``lax`` control flow.

    The last ``memory_size`` curvature pairs ``(s, y)`` live in fixed-size
    circular buffers so the whole solver has static shapes and can be
    ``jit``-ed.  The search direction comes from the standard two-loop
    recursion; the step length from Armijo backtracking.  Pairs with
    ``s . y <= 0`` (no positive curvature) are skipped, which keeps the
    inverse-Hessian approximation positive definite.

    Args:
        objective_fn: Scalar function to minimise.
        x0: Starting point (1-D array).
        memory_size: Number of curvature pairs to remember.
        tolerance: Stop when ``||grad|| < tolerance``.
        max_iterations: Iteration cap.
        line_search_steps: Maximum halvings in the backtracking search.
        armijo_c1: Sufficient-decrease constant.
    """
    value_and_grad = jax.value_and_grad(objective_fn)
    n = x0.shape[0]
    m = memory_size

    def two_loop(g, S, Y, rho, count, head):
        """Compute ``-H g`` from the circular history (most recent pair last)."""

        # Iterate over the `count` valid slots from newest to oldest.
        def backward(i, carry):
            q, alphas = carry
            idx = (head - 1 - i) % m
            valid = i < count
            a = rho[idx] * jnp.dot(S[idx], q)
            a = jnp.where(valid, a, 0.0)
            q = q - a * Y[idx]
            return q, alphas.at[idx].set(a)

        q, alphas = lax.fori_loop(0, m, backward, (g, jnp.zeros(m, g.dtype)))

        # Initial Hessian scaling gamma = s.y / y.y from the newest pair.
        newest = (head - 1) % m
        gamma = jnp.where(
            count > 0,
            jnp.dot(S[newest], Y[newest]) / jnp.maximum(jnp.dot(Y[newest], Y[newest]), 1e-30),
            1.0,
        )
        r = gamma * q

        def forward(i, r):
            idx = (head - count + i) % m
            valid = i < count
            beta = rho[idx] * jnp.dot(Y[idx], r)
            beta = jnp.where(valid, beta, 0.0)
            return r + S[idx] * (alphas[idx] - beta)

        r = lax.fori_loop(0, m, forward, r)
        return -r

    def line_search(x, f, g, d):
        slope = jnp.dot(g, d)

        def cond(state):
            alpha, k = state
            f_new = objective_fn(x + alpha * d)
            armijo = f_new <= f + armijo_c1 * alpha * slope
            return jnp.logical_and(jnp.logical_not(armijo), k < line_search_steps)

        def body(state):
            alpha, k = state
            return alpha * 0.5, k + 1

        alpha, _ = lax.while_loop(cond, body, (jnp.asarray(1.0, x.dtype), jnp.int32(0)))
        return alpha

    def cond(state):
        x, f, g, S, Y, rho, count, head, k = state
        return jnp.logical_and(jnp.linalg.norm(g) >= tolerance, k < max_iterations)

    def body(state):
        x, f, g, S, Y, rho, count, head, k = state
        d = two_loop(g, S, Y, rho, count, head)
        # Fall back to steepest descent if the direction is not a descent direction.
        d = jnp.where(jnp.dot(g, d) < 0, d, -g)
        alpha = line_search(x, f, g, d)
        x_new = x + alpha * d
        f_new, g_new = value_and_grad(x_new)
        s = x_new - x
        y = g_new - g
        sy = jnp.dot(s, y)
        accept = sy > 1e-10

        def push(args):
            S, Y, rho, count, head = args
            return (
                S.at[head].set(s),
                Y.at[head].set(y),
                rho.at[head].set(1.0 / sy),
                jnp.minimum(count + 1, m),
                (head + 1) % m,
            )

        S, Y, rho, count, head = lax.cond(accept, push, lambda a: a, (S, Y, rho, count, head))
        return x_new, f_new, g_new, S, Y, rho, count, head, k + 1

    f0, g0 = value_and_grad(x0)
    init = (
        x0,
        f0,
        g0,
        jnp.zeros((m, n), x0.dtype),
        jnp.zeros((m, n), x0.dtype),
        jnp.zeros(m, x0.dtype),
        jnp.int32(0),
        jnp.int32(0),
        jnp.int32(0),
    )
    x, _, g, *_, k = lax.while_loop(cond, body, init)
    err = jnp.linalg.norm(g)
    return x, SolverState(x=x, residual=g, iteration=k, converged=err < tolerance, error=err)


# ---------------------------------------------------------------------------
# Eigenvalue methods
# ---------------------------------------------------------------------------


def eigenvalue_power_method(
    A: LinearOperator,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    v0: Array | None = None,
    n: int | None = None,
) -> tuple[Array, Array]:
    """Power iteration for the dominant eigenpair.

    Converges at rate ``|lambda_2 / lambda_1|``; the Rayleigh quotient
    ``v^T A v`` gives the eigenvalue estimate.

    Args:
        A: Matrix or matvec callable (pass ``n`` for the callable case).
        max_iterations: Iteration cap.
        tolerance: Stop when the eigenvalue estimate changes by less than this.
        v0: Starting vector (ones by default).
        n: Dimension, required when ``A`` is a callable and ``v0`` is None.
    """
    matvec = _as_matvec(A)
    if v0 is None:
        dim = n if callable(A) else A.shape[0]
        if dim is None:
            raise ValueError("n must be given for a matrix-free operator")
        v0 = jnp.ones(dim)
    v0 = v0 / jnp.linalg.norm(v0)

    def cond(state):
        _, lam, lam_prev, k = state
        return jnp.logical_and(jnp.abs(lam - lam_prev) >= tolerance, k < max_iterations)

    def body(state):
        v, lam, _, k = state
        Av = matvec(v)
        v_new = Av / jnp.maximum(jnp.linalg.norm(Av), jnp.finfo(Av.dtype).tiny)
        lam_new = jnp.dot(v_new, matvec(v_new))
        return v_new, lam_new, lam, k + 1

    lam0 = jnp.dot(v0, matvec(v0))
    v, lam, _, _ = lax.while_loop(cond, body, (v0, lam0, lam0 + 2 * tolerance + 1.0, jnp.int32(0)))
    return lam, v


def lanczos_algorithm(
    A: LinearOperator,
    num_iterations: int,
    starting_vector: Array | None = None,
    n: int | None = None,
    reorthogonalize: bool = True,
) -> tuple[Array, Array]:
    """Lanczos tridiagonalisation ``Q^T A Q = T`` for symmetric ``A``.

    The eigenvalues of the small tridiagonal ``T`` (Ritz values) approximate
    the extreme eigenvalues of ``A`` after far fewer than ``n`` steps.  In
    floating point the Lanczos vectors lose orthogonality; ``reorthogonalize``
    applies full Gram-Schmidt against all previous vectors each step (O(k n)
    extra work) which is the simplest robust fix.

    Returns:
        ``(T, Q)`` with ``T`` of shape ``(k, k)`` and ``Q`` of shape ``(n, k)``.
    """
    matvec = _as_matvec(A)
    dim = n if callable(A) else A.shape[0]
    k = num_iterations
    if starting_vector is None:
        if dim is None:
            raise ValueError("n must be given for a matrix-free operator")
        q0 = jax.random.normal(jax.random.PRNGKey(0), (dim,))
    else:
        q0 = starting_vector
    q0 = q0 / jnp.linalg.norm(q0)

    Q0 = jnp.zeros((dim, k), q0.dtype).at[:, 0].set(q0)
    alpha0 = jnp.zeros(k, q0.dtype)
    beta0 = jnp.zeros(k, q0.dtype)  # beta[j] couples column j and j+1

    def step(j, state):
        Q, alpha, beta = state
        qj = Q[:, j]
        v = matvec(qj)
        v = v - jnp.where(j > 0, beta[jnp.maximum(j - 1, 0)], 0.0) * Q[:, jnp.maximum(j - 1, 0)]
        a = jnp.dot(qj, v)
        v = v - a * qj
        if reorthogonalize:
            mask = (jnp.arange(k) <= j).astype(v.dtype)
            v = v - Q @ ((Q.T @ v) * mask)
        b = jnp.linalg.norm(v)
        q_next = v / jnp.maximum(b, jnp.finfo(v.dtype).tiny)
        Q = lax.cond(
            j + 1 < k, lambda Q: Q.at[:, jnp.minimum(j + 1, k - 1)].set(q_next), lambda Q: Q, Q
        )
        return Q, alpha.at[j].set(a), beta.at[j].set(b)

    Q, alpha, beta = lax.fori_loop(0, k, step, (Q0, alpha0, beta0))
    T = jnp.diag(alpha) + jnp.diag(beta[:-1], 1) + jnp.diag(beta[:-1], -1)
    return T, Q
