# File location: jax-nsl/src/core/__init__.py

"""
Core JAX utilities and fundamental operations.

This module provides essential building blocks for JAX computations:
- Array utilities and dtype handling
- PRNG key management
- Numerically stable operations
"""

from .arrays import *
from .prng import *
from .numerics import *

__all__ = [
    # arrays.py
    "get_dtype_info",
    "safe_cast",
    "tree_size",
    "tree_bytes",
    "tree_summary",
    "tree_map_with_path",
    "check_finite",

    # prng.py
    "PRNGSequence",
    "split_key_tree",
    "random_like",
    "make_rng_state",
    "glorot_uniform_init",
    "glorot_normal_init",
    "he_uniform_init",
    "he_normal_init",

    # numerics.py
    "safe_log",
    "safe_exp",
    "safe_sqrt",
    "logsumexp_stable",
    "stable_logsumexp",
    "softmax_stable",
    "stable_softmax",
    "log_softmax_stable",
    "clip_gradients",
    "safe_norm",
    "safe_divide",
    "stable_sigmoid",
    "numerical_gradient",
]