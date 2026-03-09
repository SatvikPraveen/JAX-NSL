Parallel API
============

The ``parallel`` package provides utilities for data parallelism (``pmap``) and model/tensor parallelism (``pjit``).

.. contents:: Modules
   :local:
   :depth: 1

parallel.pmap_utils
-------------------

Data-parallel utilities built on ``jax.pmap``.

.. automodule:: parallel.pmap_utils
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``replicate(x)``
     - Replicates arrays across all local devices.
   * - ``unreplicate(x)``
     - Takes the first-device shard of a replicated array.
   * - ``pmapped_train_step(state, batch)``
     - Data-parallel training step using ``pmap``.
   * - ``sync_gradients(gradients)``
     - All-reduce (mean) gradients across devices via ``lax.pmean``.

parallel.pjit_utils
-------------------

Model/tensor parallelism utilities (JAX's ``pjit`` / ``jit`` with ``Mesh``).

.. automodule:: parallel.pjit_utils
   :members:
   :undoc-members:
   :show-inheritance:

parallel.collectives
--------------------

Low-level collective communication primitives.

.. automodule:: parallel.collectives
   :members:
   :undoc-members:
   :show-inheritance:

**Key functions**:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Function
     - Description
   * - ``all_reduce_sum(x, axis_name)``
     - ``lax.psum`` wrapper.
   * - ``all_reduce_mean(x, axis_name)``
     - ``lax.pmean`` wrapper.
   * - ``distributed_dot(x, y, axis_name)``
     - Distributed matrix–vector dot product.
   * - ``sync_batch_stats(batch_stats, axis_name)``
     - Synchronises batch normalisation statistics across devices.
