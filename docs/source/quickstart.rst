Quick Start
===========

Get up and running with JAX-NSL in minutes.

Installation
------------

Install JAX-NSL and its dependencies:

.. code-block:: bash

   git clone https://github.com/SatvikPraveen/JAX-NSL.git
   cd JAX-NSL
   pip install -e ".[dev]"

Basic Usage
-----------

Arrays and PRNG
~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from core.prng import PRNGSequence

   # Reproducible PRNG sequences
   seq = PRNGSequence(seed=42)
   x = jax.random.normal(next(seq), shape=(4,))
   y = jax.random.normal(next(seq), shape=(4,))

   print(x, y)

Training a Simple MLP
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from models.mlp import MLP, create_mlp
   from training.train_loop import create_train_state

   rng = jax.random.PRNGKey(0)
   mlp = MLP(features=[64, 32, 10], activation="relu")

   # Initialise parameters
   x_dummy = jnp.ones((1, 28 * 28))
   params = mlp.init(rng, x_dummy)

   # Forward pass
   logits = mlp.apply(params, x_dummy)
   print("Output shape:", logits.shape)  # (1, 10)

Gradients
~~~~~~~~~

.. code-block:: python

   from autodiff.grad_jac_hess import compute_gradient

   loss = lambda w: jnp.sum((w - 1.0) ** 2)
   grad = compute_gradient(loss, jnp.zeros(4))
   print(grad)  # [-2. -2. -2. -2.]

Next Steps
----------

* Work through the :doc:`notebooks/01_fundamentals` notebooks.
* Explore the :doc:`api/core` API reference.
* See :doc:`examples/basic_usage` for end-to-end training examples.
