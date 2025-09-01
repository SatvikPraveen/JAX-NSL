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
    
    # optimizers.py
    "sgd_optimizer",
    "momentum_optimizer", 
    "adam_optimizer",
    "adamw_optimizer",
    "rmsprop_optimizer",
    "adagrad_optimizer",
    "create_learning_rate_schedule",
    "apply_optimizer",
    
    # train_loop.py
    "training_step",
    "evaluation_step",
    "train_epoch",
    "create_train_state",
    "save_checkpoint",
    "load_checkpoint"
]