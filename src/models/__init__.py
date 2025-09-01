# File location: jax-nsl/src/models/__init__.py

"""
Neural network models implemented in pure JAX.

This module provides implementations of common neural network architectures
without high-level frameworks, focusing on educational clarity and 
understanding of the underlying computations.
"""

from .mlp import *
from .cnn import *
from .transformer import *

__all__ = [
    # mlp.py
    "init_mlp_params",
    "mlp_forward",
    "mlp_predict",
    "create_mlp",
    "dense_layer",
    "activation_fn",
    
    # cnn.py
    "init_conv_params", 
    "conv2d_layer",
    "pooling_layer",
    "cnn_forward",
    "create_cnn",
    "batch_norm_2d",
    
    # transformer.py
    "init_attention_params",
    "multi_head_attention",
    "transformer_block",
    "positional_encoding",
    "create_transformer",
    "layer_norm"
]