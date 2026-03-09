Parallelism Notebooks
======================

Notebooks 14–16 explore JAX's device-parallel execution model.

.. toctree::
   :maxdepth: 1

Notebook 14 – pmap Basics
---------------------------

**File**: ``notebooks/05_parallelism/14_pmap_basics.ipynb``

Topics covered:

* Data parallelism with ``jax.pmap``.
* Replicating model parameters across devices.
* Sharding a data batch across devices.
* Collective all-reduce for gradient synchronisation.
* ``replicate`` / ``unreplicate`` utilities.

Notebook 15 – pjit and Sharding
---------------------------------

**File**: ``notebooks/05_parallelism/15_pjit_and_sharding.ipynb``

Topics covered:

* Tensor and model parallelism with ``jax.jit`` + ``Mesh`` and ``PartitionSpec``.
* Sharding annotations: ``None``, named mesh axes.
* Inspecting shard layouts.
* Mixed data- and model-parallel training.

Notebook 16 – Collectives
---------------------------

**File**: ``notebooks/05_parallelism/16_collectives.ipynb``

Topics covered:

* ``lax.psum``, ``lax.pmean``, ``lax.pmax``.
* ``lax.all_gather``, ``lax.axis_index``.
* Ring all-reduce pattern.
* Batch-normalisation statistics synchronisation.
