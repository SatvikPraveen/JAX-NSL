Tutorials (the notebooks)
=========================

The 21 notebooks are self-contained (they do not import ``jax_nsl``) and
are executed in CI on every push, so they always run on current JAX.

Fundamentals - ``notebooks/01_fundamentals/``
---------------------------------------------

* **01 - Arrays and PRNG** - immutability, ``.at[]`` updates, key splitting.
* **02 - Autodiff basics** - ``grad``, ``value_and_grad``, ``jacfwd``/``jacrev``, higher order.
* **03 - Custom VJP/JVP** - one ``custom_jvp`` rule serves both modes; when you need ``custom_vjp``.
* **04 - Control flow and scan** - ``cond``, ``switch``, ``while_loop``, ``scan``, and what traces.

Linear algebra - ``notebooks/02_linear_algebra/``
-------------------------------------------------

* **05 - Matrix operations** - einsum, decompositions, batching.
* **06 - Iterative solvers** - CG, preconditioning, GMRES with a masked Arnoldi step inside ``fori_loop``.
* **07 - Numerical stability** - log-sum-exp, condition numbers, float32 limits.

Neural networks - ``notebooks/03_neural_networks/``
---------------------------------------------------

* **08 - MLP from scratch** - pytree parameters, initialisation, training.
* **09 - Minimal CNN** - ``conv_general_dilated`` with explicit dimension numbers.
* **10 - Attention from scratch** - scaled dot-product and multi-head attention.

Training - ``notebooks/04_training_optimization/``
--------------------------------------------------

* **11 - Optimisers** - SGD/momentum/Adam written as pure functions.
* **12 - Loss functions** - cross-entropy, focal, regularisers.
* **13 - Training loops** - pytree-registered state, checkpointing, early stopping.

Parallelism - ``notebooks/05_parallelism/``
-------------------------------------------

* **14 - pmap basics** - replicate, shard, ``pmean``.
* **15 - jit + sharding** - meshes, ``PartitionSpec``, ``NamedSharding`` (the modern ``pjit``).
* **16 - Collectives** - ``psum``/``pmean``/``all_gather``, synchronised training steps.

Special topics - ``notebooks/06_special_topics/``
-------------------------------------------------

* **17 - Differentiable ODEs** - RK4 as a ``scan``, neural ODEs, continuous normalising flows.
* **18 - Probabilistic gradients** - reparameterisation, REINFORCE, natural gradient.
* **19 - Research tricks** - gradient surgery, rematerialisation, mixed precision.

Capstones - ``notebooks/capstone_projects/``
--------------------------------------------

* **20 - Physics-informed neural networks** - heat, Poisson and Navier-Stokes residual losses.
* **21 - Large-scale training** - sharded transformer training with remat, bf16 and gradient accumulation.
