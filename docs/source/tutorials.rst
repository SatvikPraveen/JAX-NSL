Tutorials
=========

Structured tutorials that build up JAX and neural-network skills step by step.

.. contents:: Table of Contents
   :local:
   :depth: 1

Tutorial 1 – JAX Fundamentals
------------------------------

Covers JAX arrays, PRNG, JIT compilation, and basic autodiff.
See the companion notebook: ``notebooks/01_fundamentals/``.

* **01 – Arrays and PRNG** – creating arrays, key splitting, random sampling.
* **02 – Autodiff basics** – ``jax.grad``, ``jax.value_and_grad``, higher-order derivatives.
* **03 – Custom VJP/JVP** – registering custom backward and forward rules.
* **04 – Control flow and scan** – ``jax.lax.cond``, ``jax.lax.scan``, compiled loops.

Tutorial 2 – Linear Algebra with JAX
-------------------------------------

Efficient matrix operations and iterative solvers.
See the companion notebook: ``notebooks/02_linear_algebra/``.

* **05 – Matrix operations** – batched matmul, SVD, eigendecomposition, QR.
* **06 – Iterative solvers** – conjugate gradient, Jacobi iterations.
* **07 – Numerical stability** – log-sum-exp, softmax, condition numbers.

Tutorial 3 – Neural Networks from Scratch
------------------------------------------

Build MLP, CNN, and Transformer models using Flax.
See the companion notebook: ``notebooks/03_neural_networks/``.

* **08 – MLP from scratch** – Flax ``nn.Module``, parameter initialisation, forward pass.
* **09 – Minimal CNN** – convolutions, pooling, feature maps.
* **10 – Attention from scratch** – scaled dot-product attention, multi-head attention.

Tutorial 4 – Training and Optimisation
---------------------------------------

Complete training loops with Optax optimisers and loss functions.
See the companion notebook: ``notebooks/04_training_optimization/``.

* **11 – Optimisers in JAX** – SGD, Adam, AdamW, learning-rate schedules.
* **12 – Loss functions** – cross-entropy, MSE, focal, Huber, KL divergence.
* **13 – Training loops** – ``TrainState``, gradient clipping, checkpointing, early stopping.

Tutorial 5 – Parallelism
-------------------------

Scale training across devices with ``pmap`` and ``pjit``.
See the companion notebook: ``notebooks/05_parallelism/``.

* **14 – pmap basics** – data parallelism, replicated variables.
* **15 – pjit and sharding** – model and tensor parallelism, ``PartitionSpec``.
* **16 – Collectives** – ``lax.psum``, ``lax.pmean``, ``lax.all_gather``.

Tutorial 6 – Special Topics
-----------------------------

Advanced JAX techniques for research.
See the companion notebook: ``notebooks/06_special_topics/``.

* **17 – Differentiable ODEs** – adjoint methods, neural ODEs.
* **18 – Probabilistic gradients** – stochastic computation graphs, REINFORCE.
* **19 – Research tricks** – gradient checkpointing, mixed precision, custom kernels.

Capstone Projects
-----------------

End-to-end projects that integrate all concepts.
See the companion notebook: ``notebooks/capstone_projects/``.

* **20 – Physics-Informed Neural Networks** – solving PDEs with neural networks.
* **21 – Large-scale training** – multi-device training with pjit and dataset pipelines.
