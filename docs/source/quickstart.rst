Quick start
===========

Everything in ``jax_nsl`` is a plain function over pytrees; there are no
module classes to instantiate.

Stable numerics
---------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.core import log_softmax_stable, stable_sigmoid

   logits = jnp.array([[1000.0, 999.0, 1001.0]])
   print(log_softmax_stable(logits))          # finite and precise
   print(jax.grad(lambda x: stable_sigmoid(x).sum())(jnp.array([-1000.0, 1000.0])))  # finite

Train an MLP
------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.models import create_mlp
   from jax_nsl.training import adam_optimizer, cross_entropy_loss, create_train_state, make_train_step

   params, forward, _ = create_mlp([4, 32, 3], activation="relu", seed=0)
   init, update = adam_optimizer(1e-2)
   state = create_train_state(params, init, jax.random.PRNGKey(0))
   step = make_train_step(forward, cross_entropy_loss, update, max_grad_norm=1.0)  # jitted

   x = jax.random.normal(jax.random.PRNGKey(1), (64, 4))
   y = (x[:, 0] > 0).astype(jnp.int32)
   for _ in range(100):
       state, metrics = step(state, {"inputs": x, "labels": y})
   print(metrics["accuracy"])

Differentiate through a solver
------------------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.linalg import conjugate_gradient

   a = jnp.array([[4.0, 1.0], [1.0, 3.0]])
   x, info = conjugate_gradient(a, jnp.ones(2))
   # gradient w.r.t. the matrix comes from an adjoint solve, not from unrolling CG
   g = jax.grad(lambda a: conjugate_gradient(a, jnp.ones(2))[0].sum())(a)

Shard a computation
-------------------

.. code-block:: python

   import os
   os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
   import jax, jax.numpy as jnp
   from jax_nsl.parallel import P, create_mesh, setup_model_parallelism

   mesh = create_mesh((8,), ("data",))
   f = setup_model_parallelism(lambda x, w: x @ w, mesh, in_specs=(P("data", None), P()), out_specs=P("data", None))
   out = f(jnp.ones((16, 4)), jnp.ones((4, 4)))
   print(out.sharding.spec)   # P('data', None)

Next steps
----------

* Work through :doc:`tutorials` (the notebooks).
* Browse the :doc:`api/core` pages; the docstrings carry the explanations.
