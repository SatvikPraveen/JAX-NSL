Advanced patterns
=================

Implicit differentiation through a fixed point
----------------------------------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.autodiff import fixed_point

   def f(params, x):            # contraction map x -> f(params, x)
       a, b = params
       return jnp.tanh(a * x + b)

   solve = lambda p: fixed_point(f, p, jnp.array(0.0), tolerance=1e-7)
   x_star = solve((jnp.array(0.5), jnp.array(0.3)))
   g = jax.grad(lambda p: solve(p))((jnp.array(0.5), jnp.array(0.3)))   # via the adjoint fixed point

Hessian-vector products and curvature
-------------------------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.autodiff import hvp, hessian_trace_hutchinson

   loss = lambda w: jnp.sum(jnp.tanh(w) ** 2)
   w = jnp.linspace(-1, 1, 1000)
   Hv = hvp(loss, w, jnp.ones_like(w))                            # never forms the 1000x1000 Hessian
   tr = hessian_trace_hutchinson(loss, w, jax.random.PRNGKey(0), num_samples=64)

Per-example gradients and DP-style clipping
-------------------------------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.transforms import per_example_gradients, clip_per_example_gradients

   params = {"w": jnp.zeros(4), "b": jnp.zeros(())}
   loss = lambda p, x, y: (p["w"] @ x + p["b"] - y) ** 2
   xs = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
   ys = xs[:, 0]
   pe = per_example_gradients(loss, params, xs, ys)               # leaves have a leading batch axis
   g = clip_per_example_gradients(pe, max_norm=1.0)               # clip each example, then average

A transformer stack as one scan, with remat
-------------------------------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.models import create_transformer, create_causal_mask

   params, forward = create_transformer(d_model=64, num_heads=4, num_layers=6, vocab_size=100,
                                        max_seq_len=32, remat=True)
   tokens = jax.random.randint(jax.random.PRNGKey(0), (2, 16), 0, 100)
   out = jax.jit(forward)(params, tokens, mask=create_causal_mask(16))
   print(params["layers"]["attention"]["query"].shape)             # (6, 64, 64): layers stacked

Tensor-parallel partitioning of that stack
------------------------------------------

.. code-block:: python

   import os
   os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")
   import jax
   from jax_nsl.models import create_transformer
   from jax_nsl.parallel import create_mesh, partition_params, create_transformer_partition_specs, sharding_summary

   mesh = create_mesh((8,), ("model",))
   params, _ = create_transformer(d_model=64, num_heads=4, num_layers=2, vocab_size=100, max_seq_len=32)
   sharded = partition_params(params, create_transformer_partition_specs("model"), mesh)
   print(sharding_summary(sharded)["['layers']['ffn']['W1']"])   # P(None, None, 'model')

Gradient accumulation without extra memory
------------------------------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.models import create_mlp
   from jax_nsl.training import accumulate_gradients, split_into_microbatches, cross_entropy_loss

   params, forward, _ = create_mlp([4, 16, 2], seed=0)
   batch = {"inputs": jnp.ones((64, 4)), "labels": jnp.zeros(64, jnp.int32)}
   loss = lambda p, b: cross_entropy_loss(forward(p, b["inputs"]), b["labels"])
   mean_loss, grads = accumulate_gradients(loss, params, split_into_microbatches(batch, 8))
