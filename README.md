# JAX-NSL: Numerics and Systems Lab in JAX

[![CI](https://github.com/SatvikPraveen/JAX-NSL/actions/workflows/ci.yml/badge.svg)](https://github.com/SatvikPraveen/JAX-NSL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A learning resource and reference library for JAX, from array semantics and
autodiff through numerically careful linear algebra and training code to
multi-device parallelism. It has two halves that are kept honest by CI:

- **21 notebooks** (`notebooks/`) that are self-contained and executed end to
  end on every push, so they always run on current JAX.
- **`jax_nsl`** (`src/jax_nsl/`), a tested package (275 tests) whose docstrings
  explain the *reasoning* behind each function: which float32 limit it works
  around, which JAX rule it respects, what the alternative costs.

## What you will find

| Package | Highlights |
| --- | --- |
| `core` | log-softmax that keeps its digits at logits of 1000, a sigmoid with finite gradients at ±1000 (the double-`where` trick), dtype-aware finite differences, initialisers that count a conv kernel's receptive field |
| `autodiff` | `checkify`-based NaN detection, Hessian-vector products, Hutchinson trace, Gauss-Newton products, complex-step gradient checks, **implicit differentiation** of fixed points and Newton solves |
| `transforms` | retrace counter, AOT compile info (FLOPs, bytes), per-example gradients, chunked `vmap`, RK4 as a `scan`, remat scans, parallel linear recurrences with `associative_scan`, a gradient-norm-clipping function transform |
| `linalg` | preconditioned, matrix-free CG whose gradient is an **adjoint solve** (not an unrolled loop), a fully `jit`-able L-BFGS, Lanczos, power iteration |
| `models` | MLP, CNN in NHWC without transposes, batch norm with explicit running state, a Transformer whose layer stack is a single `scan` with optional `jax.checkpoint`, RoPE, pre/post-LN, mask-safe softmax |
| `training` | optimisers as `(init, update)` pairs with schedules, Lion, EMA, jitted train-step factories, gradient accumulation in `scan`, bf16 mixed precision, fp16 loss scaling, checkpoints that survive typed PRNG keys |
| `parallel` | `pmap` steps, `jit` + `NamedSharding` partition rules (FSDP-style, Megatron-style), `shard_map` matmuls, a **real ring all-reduce built from `ppermute`**; everything tested on 8 virtual CPU devices |
| `utils` | pytree helpers over `jax.tree_util` key paths, benchmarking that blocks on outputs and reports compiled-program memory |

## Install

```bash
git clone https://github.com/SatvikPraveen/JAX-NSL.git
cd JAX-NSL
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebook]"
pytest -q            # ~45 s on a laptop; parallel tests use 8 virtual CPU devices
make notebooks       # execute all 21 notebooks
```

For GPU/TPU wheels follow the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
first, then `pip install -e .`.

## A taste

```python
import jax, jax.numpy as jnp
from jax_nsl.linalg import conjugate_gradient
from jax_nsl.autodiff import fixed_point, hvp
from jax_nsl.models import create_transformer, create_causal_mask
from jax_nsl.training import adam_optimizer, cross_entropy_loss, create_train_state, make_train_step

# A solver you can differentiate through (adjoint solve, no unrolling).
a = jnp.array([[4.0, 1.0], [1.0, 3.0]])
g = jax.grad(lambda a: conjugate_gradient(a, jnp.ones(2))[0].sum())(a)

# A fixed point with an implicit gradient.
f = lambda p, x: jnp.tanh(p * x + 0.3)
dx_dp = jax.grad(lambda p: fixed_point(f, p, jnp.array(0.0)))(jnp.array(0.5))

# A transformer whose 12 layers compile as one scan under remat.
params, forward = create_transformer(d_model=128, num_heads=4, num_layers=12, vocab_size=1000, remat=True)
logits = jax.jit(forward)(params, jnp.zeros((2, 16), jnp.int32), mask=create_causal_mask(16))

# A jitted training step with clipping and metrics.
init, update = adam_optimizer(1e-3)
state = create_train_state(params, init, jax.random.PRNGKey(0))
step = make_train_step(forward, cross_entropy_loss, update, max_grad_norm=1.0)
```

Sharding on a laptop:

```python
import os; os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import jax
from jax_nsl.parallel import P, create_mesh, partition_params, create_transformer_partition_specs

mesh = create_mesh((8,), ("model",))
sharded = partition_params(params, create_transformer_partition_specs("model"), mesh)
```

## Notebooks

| | Section | Notebooks |
| --- | --- | --- |
| 01 | Fundamentals | arrays & PRNG, autodiff, custom VJP/JVP, control flow & scan |
| 02 | Linear algebra | matrix ops, iterative solvers (CG, PCG, GMRES), numerical stability |
| 03 | Neural networks | MLP, CNN, attention from scratch |
| 04 | Training | optimisers, losses, training loops with checkpointing |
| 05 | Parallelism | `pmap`, `jit` + sharding (formerly `pjit`), collectives |
| 06 | Special topics | differentiable ODEs, probabilistic gradients, research tricks |
| 07 | Capstones | physics-informed neural networks, large-scale sharded training |

The notebooks do not import `jax_nsl`; they build everything inline so each
one reads on its own. The package is where the same ideas are written
carefully and tested.

## Development

```bash
make lint          # ruff
make format        # black + isort
make test          # pytest
make test-cov      # with coverage
make notebooks     # execute every notebook with nbconvert
pre-commit install # optional hooks (ruff, black, isort, nbstripout)
```

CI runs lint, the test-suite on Python 3.11 to 3.13, and executes all
notebooks. Deprecation warnings raised from `jax_nsl` fail the tests, so API
drift in JAX shows up immediately.

## Layout

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md). Documentation sources are
in `docs/` (Sphinx, `make docs`).

## License

MIT. See [LICENSE](LICENSE).
