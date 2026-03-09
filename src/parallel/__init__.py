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
    "shard_batch",
    "parallel_eval_step",
    "create_pmap_train_step",
    "parallel_train_epoch",
    "sync_gradients",
    "sync_params_across_devices",
    "device_get",
    "estimate_memory_usage",

    # pjit_utils.py
    "create_mesh",
    "partition_params",
    "create_sharded_array",
    "shard_array",
    "pjit_train_step",
    "model_parallel_forward",
    "setup_model_parallelism",
    "shard_large_layer",
    "estimate_memory_per_device",

    # collectives.py
    "all_reduce_mean",
    "all_reduce_sum",
    "all_reduce_max",
    "all_reduce_min",
    "all_gather",
    "reduce_scatter",
    "broadcast",
    "barrier_sync",
    "cross_replica_mean",
    "tree_all_reduce",
    "gradient_synchronization",
    "distributed_dot",
    "sync_batch_stats",
]