Performance tips
================

Find accidental recompilation
-----------------------------

.. code-block:: python

   import jax.numpy as jnp
   from jax_nsl.transforms import count_compilations

   f = count_compilations(lambda x: x * 2)
   f(jnp.ones(3)); f(jnp.ones(3)); f(jnp.ones(4))
   print(f.compilation_count())     # 2: the new shape forced a retrace

Inspect the compiled program
----------------------------

.. code-block:: python

   import jax.numpy as jnp
   from jax_nsl.transforms import aot_compile, compile_info

   compiled = aot_compile(lambda a, b: a @ b, jnp.ones((256, 256)), jnp.ones((256, 256)))
   print(compile_info(compiled))    # flops, bytes accessed, temp/argument/output bytes

Benchmark correctly
-------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.utils import benchmark_function, compare_implementations, create_performance_report

   x = jnp.ones((512, 512))
   results = compare_implementations({"eager": lambda x: x @ x, "jit": jax.jit(lambda x: x @ x)}, x, num_runs=10)
   print(create_performance_report(results))   # both timings block on the result

Bound memory with chunked vmap and remat
----------------------------------------

.. code-block:: python

   import jax, jax.numpy as jnp
   from jax_nsl.transforms import chunked_vmap, scan_with_checkpointing

   f = lambda x: jnp.sum(jnp.outer(x, x))          # a large intermediate per example
   xs = jnp.ones((10_000, 256))
   out = chunked_vmap(f, xs, chunk_size=500)      # lax.map over vmap-ed chunks

   body = lambda h, x: (jnp.tanh(h + x), h)
   carry, ys = scan_with_checkpointing(body, jnp.zeros(256), xs, checkpoint_every=100)

Mixed precision
---------------

.. code-block:: python

   import jax.numpy as jnp
   from jax_nsl.models import create_mlp
   from jax_nsl.training import with_mixed_precision

   params, forward, _ = create_mlp([8, 64, 2], seed=0)
   fast_forward = with_mixed_precision(forward, jnp.bfloat16)    # f32 master params, bf16 compute
