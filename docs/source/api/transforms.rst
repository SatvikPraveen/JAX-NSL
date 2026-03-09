Transforms API
==============

The ``transforms`` package wraps JAX's functional transformations (``jit``, ``vmap``, ``scan``, and control-flow primitives) with convenience utilities.

.. contents:: Modules
   :local:
   :depth: 1

transforms.jit_utils
---------------------

JIT compilation helpers.

.. automodule:: transforms.jit_utils
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``smart_jit`` / ``efficient_jit``
     - JIT with automatic static argument detection.
   * - ``jit_with_static(fun, static_argnums)``
     - Thin wrapper around ``jax.jit`` with explicit static args.
   * - ``warmup_jit(fun, *args)``
     - Runs one warmup compilation step.
   * - ``benchmark_jit(fun, *args)``
     - Times both warmup and steady-state execution.

transforms.vmap_utils
---------------------

Vectorisation helpers.

.. automodule:: transforms.vmap_utils
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``batch_apply(fun, xs)``
     - Applies *fun* over a batch axis.
   * - ``batch_gradient(fun, xs)``
     - Per-sample gradients via ``vmap(grad(fun))``.
   * - ``batched_matmul(A, B)``
     - Vectorised ``jnp.matmul`` over a batch dimension.
   * - ``vmap_with_signature``
     - ``vmap`` with explicit in/out-axes specification.

transforms.scan_utils
---------------------

``lax.scan``-based sequential computation utilities.

.. automodule:: transforms.scan_utils
   :members:
   :undoc-members:
   :show-inheritance:

transforms.control_flow
------------------------

JAX control-flow utilities (``cond``, ``while_loop``, ``switch``).

.. automodule:: transforms.control_flow
   :members:
   :undoc-members:
   :show-inheritance:
