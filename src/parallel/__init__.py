# File location: jax-nsl/src/parallel/__init__.py

"""
Parallelism utilities for multi-device computation.

This module provides utilities for data and model parallelism using
JAX's pmap, pjit, and collective operations.
"""

from .pmap_utils import *
from .pjit_utils import *
from .collectives import *

__all__ = [
    # pmap_utils.py
    "data_parallel_step",
    "replicate_params",
    "unreplicate_params",
    "parallel_train_step",
    "parallel_eval_step",
    "create_pmap_train_step",
    
    # pjit_utils.py
    "create_mesh",
    "partition_params",
    "create_sharded_array",
    "pjit_train_step",
    "model_parallel_forward",
    "setup_model_parallelism",
    
    # collectives.py
    "all_reduce_mean",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "barrier_sync",
    "cross_replica_mean"
]