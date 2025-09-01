# File location: jax-nsl/src/linalg/ops.py

"""
Matrix operations: matmul, einsum, SVD/QR/eigendecomposition.

This module provides enhanced linear algebra operations with
numerical stability and batching support.
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union, List
import warnings


def safe_matmul(a: jnp.ndarray, 
               b: jnp.ndarray,
               precision: Optional[jax.lax.Precision] = None,
               check_shapes: bool = True) -> jnp.ndarray:
    """Safe matrix multiplication with shape checking.
    
    Args:
        a: Left matrix
        b: Right matrix  
        precision: Numerical precision (DEFAULT, HIGH, HIGHEST)
        check_shapes: Whether to validate shapes
        
    Returns:
        Matrix product a @ b
    """
    if check_shapes:
        if a.ndim < 2 or b.ndim < 2:
            raise ValueError(f"Inputs must be at least 2D: {a.shape}, {b.shape}")
        if a.shape[-1] != b.shape[-2]:
            raise ValueError(f"Inner dimensions must match: {a.shape}, {b.shape}")
    
    return jnp.matmul(a, b, precision=precision)


def batched_matmul(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Efficient batched matrix multiplication.
    
    Args:
        a: Batched matrices (..., m, k)
        b: Batched matrices (..., k, n)
        
    Returns:
        Batched products (..., m, n)
    """
    return jax.vmap(jnp.matmul)(a, b)


def einsum_path_optimize(subscripts: str, *operands, optimize: str = 'optimal') -> jnp.ndarray:
    """Einstein summation with path optimization.
    
    Args:
        subscripts: Einstein summation subscripts
        *operands: Input arrays
        optimize: Optimization strategy
        
    Returns:
        Einsum result with optimized contraction path
    """
    return jnp.einsum(subscripts, *operands, optimize=optimize)


def stable_svd(matrix: jnp.ndarray, 
              full_matrices: bool = True,
              compute_uv: bool = True,
              hermitian: bool = False) -> Union[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    """Numerically stable SVD computation.
    
    Args:
        matrix: Input matrix
        full_matrices: Whether to compute full U and Vt matrices
        compute_uv: Whether to compute U and Vt
        hermitian: Whether matrix is Hermitian
        
    Returns:
        SVD components (U, s, Vt) or just s
    """
    try:
        if compute_uv:
            u, s, vt = jnp.linalg.svd(matrix, full_matrices=full_matrices, hermitian=hermitian)
            # Ensure singular values are non-negative and sorted
            s = jnp.maximum(s, 0.0)
            return u, s, vt
        else:
            s = jnp.linalg.svd(matrix, compute_uv=False, hermitian=hermitian)
            s = jnp.maximum(s, 0.0)
            return s
    except jnp.linalg.LinAlgError as e:
        warnings.warn(f"SVD failed, using pseudoinverse fallback: {e}")
        # Fallback to eigendecomposition for square matrices
        if matrix.shape[-1] == matrix.shape[-2]:
            return stable_eigh(matrix @ matrix.T)
        else:
            raise e


def stable_eigh(matrix: jnp.ndarray, 
               UPLO: str = 'L',
               symmetrize_input: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Stable eigendecomposition for Hermitian matrices.
    
    Args:
        matrix: Hermitian matrix
        UPLO: Whether to use upper ('U') or lower ('L') triangle
        symmetrize_input: Whether to symmetrize input matrix
        
    Returns:
        (eigenvalues, eigenvectors) tuple
    """
    if symmetrize_input:
        # Ensure exact Hermitian symmetry
        matrix = (matrix + matrix.T.conj()) / 2
    
    try:
        eigenvals, eigenvecs = jnp.linalg.eigh(matrix, UPLO=UPLO)
        # Sort eigenvalues and eigenvectors in descending order
        idx = jnp.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        return eigenvals, eigenvecs
    except jnp.linalg.LinAlgError as e:
        warnings.warn(f"Eigendecomposition failed: {e}")
        raise e


def qr_decomposition(matrix: jnp.ndarray, 
                    mode: str = 'reduced',
                    pivoting: bool = False) -> Union[Tuple[jnp.ndarray, jnp.ndarray], 
                                                   Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """QR decomposition with optional pivoting.
    
    Args:
        matrix: Input matrix
        mode: 'reduced', 'complete', or 'r'
        pivoting: Whether to use column pivoting
        
    Returns:
        QR decomposition components
    """
    if pivoting:
        # JAX doesn't have built-in pivoting, use standard QR
        warnings.warn("Pivoting not supported, using standard QR")
    
    if mode == 'r':
        return jnp.linalg.qr(matrix, mode='r')
    else:
        q, r = jnp.linalg.qr(matrix, mode=mode)
        return q, r


def cholesky_safe(matrix: jnp.ndarray, 
                 regularization: float = 1e-8) -> jnp.ndarray:
    """Safe Cholesky decomposition with regularization.
    
    Args:
        matrix: Positive definite matrix
        regularization: Diagonal regularization to ensure positive definiteness
        
    Returns:
        Lower triangular Cholesky factor
    """
    # Add regularization to diagonal
    regularized = matrix + regularization * jnp.eye(matrix.shape[-1])
    
    try:
        return jnp.linalg.cholesky(regularized)
    except jnp.linalg.LinAlgError:
        # Try with larger regularization
        regularized = matrix + 1e-6 * jnp.eye(matrix.shape[-1])
        return jnp.linalg.cholesky(regularized)


def matrix_power(matrix: jnp.ndarray, power: int) -> jnp.ndarray:
    """Compute integer powers of matrices efficiently.
    
    Args:
        matrix: Square matrix
        power: Integer power
        
    Returns:
        Matrix raised to the given power
    """
    if power == 0:
        return jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    elif power == 1:
        return matrix
    elif power < 0:
        return matrix_power(jnp.linalg.inv(matrix), -power)
    else:
        # Use binary exponentiation
        result = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
        base = matrix
        
        while power > 0:
            if power % 2 == 1:
                result = result @ base
            base = base @ base
            power //= 2
        
        return result


def trace_product(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Compute trace(A @ B) efficiently without forming the product.
    
    Args:
        a: First matrix
        b: Second matrix
        
    Returns:
        Trace of the matrix product
    """
    return jnp.sum(a * b.T)


def frobenius_norm(matrix: jnp.ndarray, axis: Optional[Tuple[int, int]] = None) -> jnp.ndarray:
    """Compute Frobenius norm of matrices.
    
    Args:
        matrix: Input matrix or batch of matrices
        axis: Axes over which to compute norm (default: last two)
        
    Returns:
        Frobenius norm(s)
    """
    if axis is None:
        axis = (-2, -1)
    
    return jnp.sqrt(jnp.sum(jnp.abs(matrix) ** 2, axis=axis))


def spectral_norm(matrix: jnp.ndarray, max_iterations: int = 50) -> jnp.ndarray:
    """Compute spectral norm (largest singular value) via power iteration.
    
    Args:
        matrix: Input matrix
        max_iterations: Maximum power iterations
        
    Returns:
        Spectral norm
    """
    # Power iteration to find largest singular value
    m, n = matrix.shape[-2:]
    
    # Start with random vector
    v = jnp.ones(n, dtype=matrix.dtype)
    v = v / jnp.linalg.norm(v)
    
    for _ in range(max_iterations):
        u = matrix @ v
        u = u / jnp.linalg.norm(u)
        
        v = matrix.T @ u
        sigma = jnp.linalg.norm(v)
        v = v / sigma
    
    return sigma


def condition_number(matrix: jnp.ndarray, 
                    p: Optional[Union[None, int, str]] = None) -> jnp.ndarray:
    """Compute condition number of matrix.
    
    Args:
        matrix: Input matrix
        p: Norm type (None, 1, -1, 2, -2, 'fro')
        
    Returns:
        Condition number
    """
    return jnp.linalg.cond(matrix, p=p)


def pseudoinverse_stable(matrix: jnp.ndarray, 
                        rcond: Optional[float] = None,
                        hermitian: bool = False) -> jnp.ndarray:
    """Stable computation of Moore-Penrose pseudoinverse.
    
    Args:
        matrix: Input matrix
        rcond: Cutoff for small singular values
        hermitian: Whether matrix is Hermitian
        
    Returns:
        Pseudoinverse matrix
    """
    if rcond is None:
        rcond = max(matrix.shape[-2:]) * jnp.finfo(matrix.dtype).eps
    
    try:
        u, s, vt = stable_svd(matrix, full_matrices=False, hermitian=hermitian)
        
        # Cutoff small singular values
        cutoff = rcond * jnp.max(s)
        s_inv = jnp.where(s > cutoff, 1.0 / s, 0.0)
        
        return (vt.T * s_inv) @ u.T
    
    except Exception:
        # Fallback to direct computation
        return jnp.linalg.pinv(matrix, rcond=rcond, hermitian=hermitian)


def matrix_sqrt(matrix: jnp.ndarray, 
               hermitian: bool = True) -> jnp.ndarray:
    """Compute matrix square root.
    
    Args:
        matrix: Input matrix (should be positive definite if hermitian=True)
        hermitian: Whether to use eigendecomposition (for Hermitian matrices)
        
    Returns:
        Matrix square root
    """
    if hermitian:
        eigenvals, eigenvecs = stable_eigh(matrix)
        sqrt_eigenvals = jnp.sqrt(jnp.maximum(eigenvals, 0.0))
        return (eigenvecs * sqrt_eigenvals) @ eigenvecs.T
    else:
        # Use SVD for general matrices
        u, s, vt = stable_svd(matrix)
        sqrt_s = jnp.sqrt(s)
        return (u * sqrt_s) @ vt


def matrix_logarithm(matrix: jnp.ndarray) -> jnp.ndarray:
    """Compute matrix logarithm for positive definite matrices.
    
    Args:
        matrix: Positive definite matrix
        
    Returns:
        Matrix logarithm
    """
    eigenvals, eigenvecs = stable_eigh(matrix)
    log_eigenvals = jnp.log(jnp.maximum(eigenvals, jnp.finfo(eigenvals.dtype).tiny))
    return (eigenvecs * log_eigenvals) @ eigenvecs.T


def gram_schmidt(vectors: jnp.ndarray, normalize: bool = True) -> jnp.ndarray:
    """Gram-Schmidt orthogonalization.
    
    Args:
        vectors: Matrix with vectors as columns
        normalize: Whether to normalize the result
        
    Returns:
        Orthogonalized vectors
    """
    m, n = vectors.shape
    orthogonal = jnp.zeros_like(vectors)
    
    for i in range(n):
        v = vectors[:, i]
        
        # Subtract projections onto previous vectors
        for j in range(i):
            proj = jnp.dot(v, orthogonal[:, j]) * orthogonal[:, j]
            v = v - proj
        
        if normalize:
            v = v / jnp.linalg.norm(v)
        
        orthogonal = orthogonal.at[:, i].set(v)
    
    return orthogonal