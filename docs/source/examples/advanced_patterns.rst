Advanced Patterns
=================

More sophisticated JAX-NSL patterns for research and production use.

.. contents::
   :local:
   :depth: 1

Custom VJP / JVP
----------------

Register a custom backward pass using ``@jax.custom_vjp``:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from autodiff.custom_vjp import smooth_abs_vjp

   x = jnp.array([-2.0, 0.0, 3.0])
   y = smooth_abs_vjp(x)        # ≈ |x| but smooth around 0
   g = jax.grad(lambda z: jnp.sum(smooth_abs_vjp(z)))(x)
   print(y, g)

Per-sample Gradients with vmap
-------------------------------

Compute individual-sample gradients without slow looping:

.. code-block:: python

   import jax
   from transforms.vmap_utils import batch_gradient

   def loss_per_sample(params, x, y):
       pred = params["w"] @ x + params["b"]
       return (pred - y) ** 2

   params = {"w": jnp.ones(4), "b": jnp.zeros(())}
   X = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
   Y = jax.random.normal(jax.random.PRNGKey(1), (32,))

   per_sample_grads = batch_gradient(
       lambda p: loss_per_sample(p, X[0], Y[0]),
       params,
   )

Scan-based RNN
--------------

Efficient sequential computation with ``lax.scan``:

.. code-block:: python

   from transforms.scan_utils import scan_sequence
   import jax.numpy as jnp

   def rnn_step(h, x):
       h_next = jnp.tanh(h @ W_h + x @ W_x + b)
       return h_next, h_next   # (carry, output)

   h0 = jnp.zeros(hidden_size)
   final_h, all_h = scan_sequence(rnn_step, h0, xs)

Data Parallelism with pmap
---------------------------

Replicate a training step across all available devices:

.. code-block:: python

   import jax
   from parallel.pmap_utils import replicate, unreplicate, pmapped_train_step

   state = create_train_state(model, rng, lr=1e-3, input_shape=(1, 784))
   state = replicate(state)

   for batch in dataloader:
       # Split batch: (B, ...) → (n_devices, B//n_devices, ...)
       batch = jax.tree_util.tree_map(
           lambda x: x.reshape(jax.device_count(), -1, *x.shape[1:]), batch
       )
       state = pmapped_train_step(state, batch)

   # Retrieve params from device 0
   params = unreplicate(state).params

Benchmarking JIT Compilation
-----------------------------

.. code-block:: python

   from transforms.jit_utils import benchmark_jit
   import jax.numpy as jnp

   def matmul(A, B):
       return A @ B

   A = jnp.ones((1024, 1024))
   B = jnp.ones((1024, 1024))

   warmup_t, mean_t = benchmark_jit(matmul, A, B, warmup_runs=3, benchmark_runs=20)
   print(f"Warmup: {warmup_t*1e3:.1f} ms   Steady-state: {mean_t*1e3:.2f} ms")
