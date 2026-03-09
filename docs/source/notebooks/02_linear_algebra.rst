Linear Algebra Notebooks
=========================

Notebooks 05–07 cover efficient numerical linear algebra with JAX.

.. toctree::
   :maxdepth: 1

Notebook 05 – Matrix Operations
---------------------------------

**File**: ``notebooks/02_linear_algebra/05_matrix_ops.ipynb``

Topics covered:

* Batched matrix multiplication with ``vmap``.
* Singular Value Decomposition (SVD) and low-rank approximations.
* Eigendecomposition: symmetric (``eigh``) and general (``eig``).
* QR factorisation and orthogonalisation.
* Cholesky decomposition for positive-definite systems.

Notebook 06 – Iterative Solvers
---------------------------------

**File**: ``notebooks/02_linear_algebra/06_iterative_solvers.ipynb``

Topics covered:

* Conjugate Gradient method for symmetric PD systems.
* Gradient descent as a simple linear solver.
* Power iteration for dominant eigenvalues.
* Convergence analysis and preconditioning.

Notebook 07 – Numerical Stability
-----------------------------------

**File**: ``notebooks/02_linear_algebra/07_numerical_stability.ipynb``

Topics covered:

* Log-sum-exp trick for numerical stability.
* Stable softmax and cross-entropy computation.
* Condition numbers and ill-conditioned matrices.
* Safe operations: ``safe_sqrt``, ``safe_divide``.
