# File location: src/jax_nsl/__init__.py

"""
JAX-NSL: Numerics and Systems Lab in JAX.

A reference implementation library and learning resource covering JAX from
the fundamentals (arrays, PRNG, autodiff, ``jit``/``vmap``/``scan``) through
numerically careful linear algebra and training code to multi-device
parallelism (``pmap``, ``jit`` + sharding, ``shard_map``, collectives).

Subpackages
-----------
core        dtype/array helpers, PRNG sequences and initialisers, stable numerics
autodiff    derivative utilities, custom VJP/JVP rules, implicit differentiation
transforms  jit diagnostics, vmap patterns, scan/ODE/remat utilities, control flow
linalg      decompositions, norms, jittable & differentiable iterative solvers
models      MLP, CNN and Transformer building blocks in pure JAX
training    losses, optimisers, schedules, jitted train steps, checkpoints
parallel    pmap data parallelism, jit/shard_map sharding, collectives
utils       pytree helpers and benchmarking
"""

from . import autodiff, core, linalg, models, parallel, training, transforms, utils

__version__ = "0.2.0"
__author__ = "Satvik Praveen"
__email__ = "satvikpraveen707@gmail.com"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "core",
    "autodiff",
    "transforms",
    "linalg",
    "models",
    "training",
    "parallel",
    "utils",
]
