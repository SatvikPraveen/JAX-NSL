# File location: jax-nsl/src/transforms/__init__.py

"""
JAX transformation utilities and patterns.

This module provides utilities for working with JAX's core transformations:
JIT compilation, vectorization (vmap), scan operations, and control flow.
"""

from .jit_utils import *
from .vmap_utils import *  
from .scan_utils import *
from .control_flow import *

__all__ = [
    # jit_utils.py
    "smart_jit",
    "conditional_jit",
    "donate_argnums_jit",
    "static_argnums_jit",
    "jit_with_cache",
    "jit_with_static",
    "efficient_jit",
    "benchmark_jit",
    "warmup_jit",

    # vmap_utils.py
    "batch_apply",
    "vectorize_function",
    "batch_outer_product",
    "parallel_map",
    "batch_matrix_ops",
    "batch_gradient",
    "batch_jacobian",
    "batched_matmul",
    "batched_gradient",
    "parallel_apply",

    # scan_utils.py
    "cumulative_op",
    "cumulative_sum",
    "running_statistics",
    "sequential_apply",
    "rnn_scan",
    "ode_solve_scan",
    "solve_ode",
    "associative_scan",
    "scan_with_checkpointing",

    # control_flow.py
    "safe_cond",
    "switch_case",
    "while_loop_safe",
    "for_loop",
    "dynamic_slice_safe",
    "clip_gradient",
    "safe_divide",
    "stable_softmax",
]