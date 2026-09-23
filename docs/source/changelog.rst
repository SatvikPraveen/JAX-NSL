Changelog
=========

0.2.0 - 2026-09-23
------------------

A review of the whole project: the package now imports, every function is
tested (275 tests) and all 21 notebooks execute on current JAX.

Changed
~~~~~~~

* Package is ``jax_nsl`` under ``src/`` (was an unimportable ``src`` package).
* ``core.numerics``: precision-correct ``log_softmax_stable``; gradient-safe
  ``stable_sigmoid``; dtype-aware finite-difference steps.
* ``linalg.solvers``: preconditioned, matrix-free, implicitly differentiable
  CG; ``jit``-able L-BFGS, Lanczos and power iteration.
* ``autodiff``: ``checkify``-based ``safe_grad``; HVPs, Hutchinson trace,
  complex-step differences; new ``implicit`` module (fixed points, Newton).
* ``transforms``: honest ``jit`` helpers (retrace counter, AOT compile info),
  RK4 ``scan`` integrator, correct remat scan, ``linear_recurrence`` via
  ``associative_scan``, per-example gradients, chunked ``vmap``.
* ``models``: parameters are array-only pytrees; NHWC convolutions; batch
  norm with explicit running state; Transformer as a ``scan`` over stacked
  layers with remat, RoPE, pre/post-LN, mask-safe softmax.
* ``training``: working optimiser states (previous NamedTuple subclasses
  had no fields), schedules, Lion, EMA, jitted step factories, gradient
  accumulation, bf16 mixed precision, checkpoints with typed keys.
* ``parallel``: ``jit`` + ``NamedSharding`` replaces ``pjit``; regex-based
  partition rules; ``shard_map`` examples; real ring all-reduce; correct
  broadcast; tests run on 8 virtual CPU devices.
* ``utils``: pytree helpers on ``jax.tree_util`` key paths; benchmarking
  that blocks on outputs and reports device/compiled-program memory.
* Notebooks updated for current JAX (``jax.tree.map``, ``jax.sharding``)
  and fixed where the code itself was wrong (inverted causal mask,
  wrong shapes, undefined names).
* Tooling: ruff/black/isort clean, CI executes notebooks, pre-commit.

0.1.0
-----

Initial release: 21 notebooks and the first version of the reference modules.
