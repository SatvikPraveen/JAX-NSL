Parallel API
============

``jax_nsl.parallel``: ``pmap`` data parallelism, ``jit`` + ``NamedSharding`` / ``shard_map`` model parallelism, and collectives.

Every function's docstring explains *why* it is written the way it is (which numerical
pitfall it avoids, which JAX rule it works around), so the API reference doubles as notes.

.. contents:: Modules
   :local:
   :depth: 1

jax_nsl.parallel.pmap_utils
---------------------------

.. automodule:: jax_nsl.parallel.pmap_utils
   :members:
   :undoc-members:
   :show-inheritance:

jax_nsl.parallel.pjit_utils
---------------------------

.. automodule:: jax_nsl.parallel.pjit_utils
   :members:
   :undoc-members:
   :show-inheritance:

jax_nsl.parallel.collectives
----------------------------

.. automodule:: jax_nsl.parallel.collectives
   :members:
   :undoc-members:
   :show-inheritance:
