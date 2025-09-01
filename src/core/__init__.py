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
    
    # prng.py  
    "PRNGSequence",
    "split_key_tree",
    "random_like",
    "make_rng_state",
    
    # numerics.py
    "safe_log",
    "safe_exp", 
    "logsumexp_stable",
    "softmax_stable",
    "clip_gradients",
    "safe_norm",
    "safe_divide"
]