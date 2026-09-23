# File location: src/jax_nsl/linalg/ops.py

"""
Matrix operations with attention to precision and conditioning.

Notes on JAX linear algebra:

* ``jnp.linalg`` never raises ``LinAlgError`` inside traced code - a singular
  or non-PD input yields ``nan`` instead.  Check the *output* if you care.
* On TPU (and GPU with TF32) the default matmul precision is reduced; pass
  ``precision=jax.lax.Precision.HIGHEST`` for genuinely float32 results.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


def safe_matmul(
    a: Array, b: Array, precision: lax.Precision | None = None, check_shapes: bool = True
) -> Array:
    """``a @ b`` with eager shape validation and an explicit precision setting."""
    if check_shapes:
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError(f"Inputs must be at least 2-D: {a.shape}, {b.shape}")
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(f"Inner dimensions must match: {a.shape}, {b.shape}")
    return jnp.matmul(a, b, precision=precision)


def batched_matmul(a: Array, b: Array) -> Array:
    """``(..., m, k) @ (..., k, n)`` over a leading batch axis via ``vmap``."""
    return jax.vmap(jnp.matmul)(a, b)


def einsum_path_optimize(subscripts: str, *operands: Array, optimize: str = "optimal") -> Array:
    """``jnp.einsum`` with contraction-path optimisation enabled."""
    return jnp.einsum(subscripts, *operands, optimize=optimize)


# ---------------------------------------------------------------------------
# Decompositions
# ---------------------------------------------------------------------------


def stable_svd(
    matrix: Array, full_matrices: bool = True, compute_uv: bool = True, hermitian: bool = False
) -> Array | tuple[Array, Array, Array]:
    """SVD with singular values clamped to be non-negative (round-off can give -1e-8)."""
    if compute_uv:
        u, s, vt = jnp.linalg.svd(matrix, full_matrices=full_matrices, hermitian=hermitian)
        return u, jnp.maximum(s, 0.0), vt
    return jnp.maximum(jnp.linalg.svd(matrix, compute_uv=False, hermitian=hermitian), 0.0)


def stable_eigh(
    matrix: Array, UPLO: str = "L", symmetrize_input: bool = True
) -> tuple[Array, Array]:
    """Eigendecomposition of a Hermitian matrix, eigenvalues in ascending order.

    ``eigh`` only reads one triangle, so a slightly asymmetric input (from
    ``A @ A.T`` in float32, say) silently produces the decomposition of a
    *different* matrix.  Symmetrising first removes that ambiguity.
    """
    if symmetrize_input:
        matrix = (matrix + jnp.swapaxes(matrix, -1, -2).conj()) / 2
    return jnp.linalg.eigh(matrix, UPLO=UPLO)


def qr_decomposition(matrix: Array, mode: str = "reduced") -> Array | tuple[Array, Array]:
    """Householder QR; ``mode`` in ``{'reduced', 'complete', 'r'}``."""
    return jnp.linalg.qr(matrix, mode=mode)


def cholesky_safe(matrix: Array, regularization: float = 1e-8) -> Array:
    """Cholesky of ``matrix + regularization * I``.

    The jitter makes a positive *semi*-definite matrix (a Gram matrix, a
    covariance with a zero-variance direction) strictly positive definite.
    Note that inside ``jit`` a failed factorisation yields ``nan``, not an
    exception.
    """
    n = matrix.shape[-1]
    return jnp.linalg.cholesky(matrix + regularization * jnp.eye(n, dtype=matrix.dtype))


def matrix_power(matrix: Array, power: int) -> Array:
    """Integer matrix power by binary exponentiation (negative powers invert first)."""
    if power < 0:
        return matrix_power(jnp.linalg.inv(matrix), -power)
    result = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    base = matrix
    while power > 0:
        if power & 1:
            result = result @ base
        base = base @ base
        power >>= 1
    return result


def matrix_sqrt(matrix: Array, hermitian: bool = True) -> Array:
    """Principal square root via eigendecomposition (Hermitian PSD) or SVD."""
    if hermitian:
        w, v = stable_eigh(matrix)
        return (v * jnp.sqrt(jnp.maximum(w, 0.0))) @ v.T
    u, s, vt = stable_svd(matrix)
    return (u * jnp.sqrt(s)) @ vt


def matrix_logarithm(matrix: Array) -> Array:
    """Matrix logarithm of a Hermitian positive-definite matrix."""
    w, v = stable_eigh(matrix)
    log_w = jnp.log(jnp.maximum(w, jnp.finfo(w.dtype).tiny))
    return (v * log_w) @ v.T


def pseudoinverse_stable(
    matrix: Array, rcond: float | None = None, hermitian: bool = False
) -> Array:
    """Moore-Penrose pseudoinverse with an explicit singular-value cutoff."""
    if rcond is None:
        rcond = max(matrix.shape[-2:]) * float(jnp.finfo(matrix.dtype).eps)
    u, s, vt = stable_svd(matrix, full_matrices=False, hermitian=hermitian)
    cutoff = rcond * jnp.max(s)
    s_inv = jnp.where(s > cutoff, 1.0 / jnp.where(s > cutoff, s, 1.0), 0.0)
    return (vt.T * s_inv) @ u.T


def gram_schmidt(vectors: Array, normalize: bool = True) -> Array:
    """Modified Gram-Schmidt on the columns of ``vectors``.

    *Modified* GS subtracts each projection from the running vector (rather
    than from the original), which is far more stable than classical GS.  It
    is O(n^2 m) sequential work - for anything beyond teaching, use
    ``jnp.linalg.qr``.
    """
    m, n = vectors.shape

    def outer(i, Q):
        v = vectors[:, i]

        def inner(j, v):
            qj = Q[:, j]
            return v - jnp.where(j < i, jnp.dot(qj, v), 0.0) * qj

        v = lax.fori_loop(0, n, inner, v)
        if normalize:
            v = v / jnp.maximum(jnp.linalg.norm(v), jnp.finfo(v.dtype).tiny)
        return Q.at[:, i].set(v)

    return lax.fori_loop(0, n, outer, jnp.zeros_like(vectors))


# ---------------------------------------------------------------------------
# Norms and conditioning
# ---------------------------------------------------------------------------


def trace_product(a: Array, b: Array) -> Array:
    """``trace(a @ b)`` without forming the product: ``sum(a * b.T)``."""
    return jnp.sum(a * jnp.swapaxes(b, -1, -2))


def frobenius_norm(matrix: Array, axis: tuple[int, int] | None = None) -> Array:
    """Frobenius norm over the last two axes (or the given pair)."""
    if axis is None:
        axis = (-2, -1)
    return jnp.sqrt(jnp.sum(jnp.abs(matrix) ** 2, axis=axis))


def spectral_norm(matrix: Array, max_iterations: int = 50, v0: Array | None = None) -> Array:
    """Largest singular value by power iteration on ``A^T A``.

    Each iteration multiplies by ``A`` and ``A^T``; the estimate converges as
    ``(sigma_2 / sigma_1) ** (2k)``.  This is the estimator used for spectral
    normalisation in GANs, where a single iteration per training step suffices
    because the weights change slowly.
    """
    _, n = matrix.shape[-2:]
    v = jnp.ones(n, dtype=matrix.dtype) if v0 is None else v0
    v = v / jnp.linalg.norm(v)

    def body(_, carry):
        v, _ = carry
        u = matrix @ v
        u = u / jnp.maximum(jnp.linalg.norm(u), jnp.finfo(u.dtype).tiny)
        w = matrix.T @ u
        sigma = jnp.linalg.norm(w)
        return w / jnp.maximum(sigma, jnp.finfo(w.dtype).tiny), sigma

    _, sigma = lax.fori_loop(0, max_iterations, body, (v, jnp.asarray(0.0, matrix.dtype)))
    return sigma


def condition_number(matrix: Array, p: int | str | None = None) -> Array:
    """``kappa_p(A) = ||A||_p ||A^{-1}||_p`` (2-norm by default, via the SVD)."""
    return jnp.linalg.cond(matrix, p=p)
