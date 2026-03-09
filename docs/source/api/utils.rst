Utils API
=========

The ``utils`` package provides benchmarking and pytree utility functions.

.. contents:: Modules
   :local:
   :depth: 1

utils.benchmarking
------------------

Utilities for timing JAX computations.

.. automodule:: utils.benchmarking
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``timer(fun, *args)``
     - Times a single function call (after device sync).
   * - ``benchmark(fun, *args, n_runs)``
     - Runs *fun* multiple times and returns mean/std timing.
   * - ``profile_memory(fun, *args)``
     - Estimates peak device memory used by *fun*.

utils.tree_utils
----------------

Helpers for manipulating JAX pytrees.

.. automodule:: utils.tree_utils
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``tree_zeros_like(tree)``
     - Returns a pytree of zeros matching the structure of *tree*.
   * - ``tree_l2_norm(tree)``
     - Global L2 norm across all leaves.
   * - ``tree_add(tree_a, tree_b)``
     - Element-wise addition of two pytrees.
   * - ``tree_scale(tree, scalar)``
     - Scales all leaves of a pytree by *scalar*.
