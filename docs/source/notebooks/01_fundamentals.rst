Fundamentals Notebooks
======================

Notebooks 01–04 cover the core building blocks of JAX.

.. toctree::
   :maxdepth: 1

Notebook 01 – Arrays and PRNG
------------------------------

**File**: ``notebooks/01_fundamentals/01_arrays_and_prng.ipynb``

Topics covered:

* Creating and manipulating JAX arrays (``jnp`` API).
* Understanding JAX's functional PRNG model (splitting keys).
* Using the ``PRNGSequence`` helper for stateful key management.
* dtype promotion rules and device placement.

Notebook 02 – Autodiff Basics
------------------------------

**File**: ``notebooks/01_fundamentals/02_autodiff_basics.ipynb``

Topics covered:

* ``jax.grad`` and ``jax.value_and_grad``.
* ``jax.jacobian`` and ``jax.hessian``.
* Differentiating through control flow.
* Stopping gradients with ``jax.lax.stop_gradient``.

Notebook 03 – Custom VJP and JVP
----------------------------------

**File**: ``notebooks/01_fundamentals/03_custom_vjp_jvp.ipynb``

Topics covered:

* Registering custom backward passes (VJP) with ``@jax.custom_vjp``.
* Registering custom forward-mode rules (JVP) with ``@jax.custom_jvp``.
* Practical examples: smooth absolute value, gradient clipping.

Notebook 04 – Control Flow and Scan
-------------------------------------

**File**: ``notebooks/01_fundamentals/04_control_flow_scan.ipynb``

Topics covered:

* ``jax.lax.cond``, ``jax.lax.switch``, ``jax.lax.while_loop``.
* ``jax.lax.scan`` for compiled sequential computation.
* Checkpointed scan for memory-efficient backprop.
