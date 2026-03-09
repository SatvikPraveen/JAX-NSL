# File location: jax-nsl/src/training/__init__.py

"""
Training utilities: losses, optimizers, and training loops.

This module provides implementations of common loss functions,
optimization algorithms, and training loop patterns for neural networks.
"""

from .losses import *
from .optimizers import *
from .train_loop import *

__all__ = [
    # losses.py
    "cross_entropy_loss",
    "mse_loss",
    "huber_loss",
    "focal_loss",
    "contrastive_loss",
    "triplet_loss",
    "kl_divergence",
    "binary_cross_entropy",
    "cosine_similarity_loss",
    "smooth_l1_loss",
    "dice_loss",

    # optimizers.py
    "sgd_optimizer",
    "momentum_optimizer",
    "adam_optimizer",
    "adamw_optimizer",
    "rmsprop_optimizer",
    "adagrad_optimizer",
    "create_learning_rate_schedule",
    "apply_optimizer",
    "clip_grads_by_global_norm",
    "get_learning_rate",

    # train_loop.py
    "TrainState",
    "create_train_state",
    "training_step",
    "evaluation_step",
    "train_epoch",
    "evaluate_model",
    "training_loop",
    "save_checkpoint",
    "load_checkpoint",
    "compute_metrics",
]