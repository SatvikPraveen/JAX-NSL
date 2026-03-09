Linear Algebra API
==================

The ``linalg`` package provides numerically stable matrix operations and iterative linear solvers.

.. contents:: Modules
   :local:
   :depth: 1

linalg.ops
----------

Matrix operations including batched routines, decompositions, and norms.

.. automodule:: linalg.ops
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``safe_matmul(A, B)``
     - Batched matrix multiply with shape validation.
   * - ``batched_matmul(A, B)``
     - ``vmap``-based batch matmul.
   * - ``stable_svd(A)``
     - SVD with sorted singular values.
   * - ``stable_eigh(A)``
     - Symmetric eigendecomposition with sorted eigenvalues.
   * - ``stable_qr(A)``
     - QR factorisation with positive diagonal R.
   * - ``stable_cholesky(A, eps)``
     - Cholesky with jitter for near-singular matrices.
   * - ``matrix_norm(A, ord)``
     - Frobenius and operator norms.

linalg.solvers
--------------

Iterative linear system solvers.

.. automodule:: linalg.solvers
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``conjugate_gradient(A, b, tol, max_iter)``
     - Solves *Ax = b* using the conjugate gradient method.
   * - ``gradient_descent_solver(A, b, lr, max_iter)``
     - Gradient descent for quadratic systems.
   * - ``power_method(A, num_iter)``
     - Dominant eigenvalue/vector via power iteration.
