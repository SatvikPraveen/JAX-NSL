Special Topics Notebooks
=========================

Notebooks 17–19 cover advanced research techniques in JAX.

.. toctree::
   :maxdepth: 1

Notebook 17 – Differentiable ODEs
-----------------------------------

**File**: ``notebooks/06_special_topics/17_differentiable_odes.ipynb``

Topics covered:

* ODE integration via ``lax.scan`` (Euler, RK4).
* Adjoint sensitivity method for memory-efficient backprop through ODEs.
* Neural ODEs: parameterising the ODE dynamics with a neural network.
* Solving boundary-value problems.

Notebook 18 – Probabilistic Gradients
---------------------------------------

**File**: ``notebooks/06_special_topics/18_probabilistic_gradients.ipynb``

Topics covered:

* REINFORCE / score-function gradient estimator.
* Straight-through estimator for discrete variables.
* Reparameterisation trick for continuous latent variables.
* Variance reduction with control variates.

Notebook 19 – Research Tricks
-------------------------------

**File**: ``notebooks/06_special_topics/19_research_tricks.ipynb``

Topics covered:

* Gradient checkpointing (``jax.checkpoint``) for memory efficiency.
* Mixed-precision training with ``bfloat16``.
* Custom kernels via ``jax.pure_callback`` and ``jax.experimental.io_callback``.
* Profiling with JAX's built-in profiler.
* Debugging with ``jax.debug.print`` and ``jax.debug.breakpoint``.
