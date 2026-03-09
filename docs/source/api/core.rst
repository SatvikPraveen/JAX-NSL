Core API
========

The ``core`` package provides fundamental array utilities, PRNG management, and numerically stable math operations.

.. contents:: Modules
   :local:
   :depth: 1

core.arrays
-----------

Array dtype inspection, safe casting, and pytree-level helpers.

.. automodule:: core.arrays
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - ``get_dtype_info(x)``
     - Returns dtype, shape, and device of an array.
   * - ``safe_cast(x, dtype)``
     - Casts array to target dtype with overflow checking.
   * - ``tree_size(tree)``
     - Total number of elements across a pytree.
   * - ``tree_bytes(tree)``
     - Total bytes consumed by a pytree.
   * - ``tree_summary(tree)``
     - Human-readable summary of pytree shapes and dtypes.
   * - ``check_finite(x)``
     - Returns ``True`` iff all elements of *x* are finite.

core.prng
---------

Stateful PRNG sequence and random array primitives.

.. automodule:: core.prng
   :members:
   :undoc-members:
   :show-inheritance:

core.numerics
-------------

Numerically stable implementations of common operations.

.. automodule:: core.numerics
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``logsumexp_stable`` / ``stable_logsumexp``
     - Log-sum-exp with max subtraction for numerical stability.
   * - ``softmax_stable`` / ``stable_softmax``
     - Numerically stable softmax.
   * - ``safe_sqrt(x, eps)``
     - ``sqrt(max(x, eps))`` – avoids gradient NaNs at zero.
   * - ``safe_divide(a, b, eps)``
     - Division with epsilon denominator guard.
   * - ``numerical_gradient(fun, x, h)``
     - Central finite-difference gradient (for gradient checking).
