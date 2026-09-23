JAX-NSL: Numerics and Systems Lab in JAX
=========================================

JAX-NSL is a learning resource and reference library for JAX. It pairs 21
executable notebooks with a tested package, ``jax_nsl``, whose code is
written to be read: every module documents the numerical or systems
reasoning behind it, from why ``x - logsumexp(x)`` loses digits to how a
ring all-reduce is assembled from point-to-point sends.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart
   tutorials

.. toctree::
   :maxdepth: 1
   :caption: Notebooks

   notebooks/01_fundamentals
   notebooks/02_linear_algebra
   notebooks/03_neural_networks
   notebooks/04_training_optimization
   notebooks/05_parallelism
   notebooks/06_special_topics
   notebooks/capstone_projects

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api/core
   api/autodiff
   api/transforms
   api/linalg
   api/models
   api/training
   api/parallel
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic_usage
   examples/advanced_patterns
   examples/performance_tips

.. toctree::
   :maxdepth: 1
   :caption: Development

   contributing
   changelog

What is in the package
----------------------

* **core** - stable ``logsumexp``/``softmax``/``sigmoid`` (with the gradient
  pitfalls explained), dtype-aware finite differences, initialisers with
  correct fan computation for convolutions.
* **autodiff** - Hessian-vector products, Hutchinson trace estimates,
  complex-step gradient checking, ``checkify``-based NaN detection, and
  implicit differentiation of fixed points and Newton solves.
* **transforms** - retrace counting and AOT compile inspection, per-example
  gradients, chunked ``vmap``, RK4 integration as a ``scan``, remat scans,
  parallel linear recurrences with ``associative_scan``.
* **linalg** - preconditioned CG with an implicit (adjoint-solve) gradient,
  a fully ``jit``-able L-BFGS, Lanczos, power iteration.
* **models** - MLP, CNN (NHWC, no transposes) and a Transformer whose layer
  stack runs as a single ``scan`` with optional rematerialisation, RoPE,
  pre-/post-LN.
* **training** - optimisers with schedules, EMA, gradient accumulation,
  bf16 mixed precision, checkpoints that survive typed PRNG keys.
* **parallel** - ``pmap`` steps, ``jit`` + ``NamedSharding`` partitioning
  rules (FSDP, Megatron-style), ``shard_map`` examples, and a real ring
  all-reduce; all tested on 8 virtual CPU devices.

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
