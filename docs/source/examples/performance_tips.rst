Performance Tips
================

Practical guidelines for writing fast JAX-NSL code.

.. contents::
   :local:
   :depth: 1

Always JIT-compile Hot Loops
----------------------------

Wrap every function that is called repeatedly inside a training loop with ``jax.jit``:

.. code-block:: python

   import jax
   from transforms.jit_utils import smart_jit

   @smart_jit
   def train_step(state, batch):
       ...

Avoid Python-Side Control Flow on Traced Values
------------------------------------------------

Python ``if``/``for`` applied to JAX arrays will re-trace on every call.
Use ``jax.lax.cond``, ``jax.lax.switch``, or ``jax.lax.while_loop`` instead:

.. code-block:: python

   # BAD – triggers a re-trace each step
   def step(x):
       if x > 0:          # Python bool on JAX array
           return x * 2
       return -x

   # GOOD – compiled once
   import jax.lax as lax
   def step(x):
       return lax.cond(x > 0, lambda: x * 2, lambda: -x)

Use scan for Sequential Computation
-------------------------------------

``jax.lax.scan`` is significantly faster than Python loops:

.. code-block:: python

   # Slow Python loop
   carry = init
   for x in sequence:
       carry, out = f(carry, x)

   # Fast compiled scan
   from transforms.scan_utils import scan_sequence
   carry, outs = scan_sequence(f, init, sequence)

Numerically Stable Operations
------------------------------

Use the stable primitives in ``core.numerics`` to avoid NaNs:

.. code-block:: python

   from core.numerics import stable_logsumexp, stable_softmax, safe_sqrt

   # Stable log-sum-exp (avoids overflow for large logits)
   log_probs = stable_logsumexp(logits)

   # Safe sqrt (avoids NaN gradient at 0)
   norms = safe_sqrt(jnp.sum(x**2, axis=-1), eps=1e-8)

Profile Memory and Timing
--------------------------

Use the profiling utilities before large runs:

.. code-block:: python

   from utils.benchmarking import benchmark
   from transforms.jit_utils import benchmark_jit

   # Per-call timing
   mean_t, std_t = benchmark(fun, *args, n_runs=50)
   print(f"{mean_t*1e3:.2f} ± {std_t*1e3:.2f} ms")

   # Warmup vs steady-state
   w_t, s_t = benchmark_jit(fun, *args)

Gradient Clipping
-----------------

Clip gradients to prevent exploding gradients in deep networks:

.. code-block:: python

   from training.optimizers import create_optimizer_with_clipping

   # Or use Optax directly (recommended)
   import optax
   tx = optax.chain(
       optax.clip_by_global_norm(1.0),
       optax.adam(1e-3),
   )

Use ``float32`` by Default, ``bfloat16`` for Large Models
----------------------------------------------------------

.. code-block:: python

   import jax
   # Enable bfloat16 globally (TPUs and modern GPUs)
   jax.config.update("jax_default_matmul_precision", "bfloat16")

   # Or cast individual arrays
   x_bf16 = x.astype(jnp.bfloat16)
