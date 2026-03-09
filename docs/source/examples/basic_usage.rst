Basic Usage Examples
====================

End-to-end examples for common JAX-NSL workflows.

.. contents::
   :local:
   :depth: 1

Linear Regression
-----------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from jax import grad, jit
   from core.prng import PRNGSequence

   # ── Data ──────────────────────────────────────────────────────────────────
   seq = PRNGSequence(0)
   X = jax.random.normal(next(seq), (100, 4))
   true_w = jnp.array([1.0, -2.0, 0.5, 3.0])
   y = X @ true_w + jax.random.normal(next(seq), (100,)) * 0.1

   # ── Model ─────────────────────────────────────────────────────────────────
   def predict(w, x):
       return x @ w

   def loss(w, x, y):
       return jnp.mean((predict(w, x) - y) ** 2)

   # ── Training ──────────────────────────────────────────────────────────────
   w = jnp.zeros(4)
   lr = 0.1
   grad_fn = jit(grad(loss))

   for step in range(200):
       g = grad_fn(w, X, y)
       w = w - lr * g
       if step % 50 == 0:
           print(f"step {step:3d}  loss={loss(w, X, y):.4f}")

   print("Recovered weights:", w)

MLP for MNIST-like Classification
----------------------------------

.. code-block:: python

   import jax
   import jax.numpy as jnp
   import optax
   from models.mlp import create_mlp
   from training.losses import cross_entropy_loss
   from training.train_loop import create_train_state, train_step

   rng = jax.random.PRNGKey(42)
   model = create_mlp(features=[256, 128, 10])

   # Dummy data (replace with real dataset)
   X = jax.random.normal(rng, (512, 784))
   y = jax.random.randint(rng, (512,), 0, 10)

   state = create_train_state(model, rng, learning_rate=1e-3, input_shape=(1, 784))

   for epoch in range(5):
       for i in range(0, 512, 32):
           batch = {"image": X[i:i+32], "label": y[i:i+32]}
           state, loss = train_step(state, batch)
       print(f"Epoch {epoch+1}  loss={loss:.4f}")

Gradient Checking
-----------------

.. code-block:: python

   from autodiff.grad_jac_hess import compute_gradient, gradient_checker
   import jax.numpy as jnp

   fun = lambda x: jnp.sum(jnp.sin(x) ** 2)
   x0 = jnp.array([0.5, 1.0, -0.3])

   analytic_g = compute_gradient(fun, x0)
   max_err = gradient_checker(fun, x0)

   print("Analytic gradient:", analytic_g)
   print("Max finite-diff error:", max_err)
