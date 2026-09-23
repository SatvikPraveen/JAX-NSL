# File location: src/jax_nsl/models/__init__.py

"""
Neural network architectures in pure JAX (MLP, CNN, Transformer).
"""

from .cnn import (
    batch_norm_2d,
    cnn_forward,
    conv2d,
    conv2d_layer,
    conv_output_shape,
    create_cnn,
    depthwise_conv2d,
    global_average_pooling,
    init_batch_norm_2d,
    init_conv_params,
    pooling_layer,
    residual_block,
    separable_conv2d,
)
from .mlp import (
    activation_fn,
    batch_norm,
    count_parameters,
    create_classifier,
    create_mlp,
    create_regressor,
    dense_layer,
    get_layer_outputs,
    init_batch_norm,
    init_mlp_params,
    init_mlp_with_batch_norm,
    mlp_forward,
    mlp_predict,
    mlp_with_batch_norm,
    mlp_with_dropout,
)
from .transformer import (
    combine_masks,
    create_causal_mask,
    create_padding_mask,
    create_transformer,
    feed_forward_network,
    init_attention_params,
    init_feed_forward_params,
    init_layer_norm_params,
    init_transformer_block_params,
    layer_norm,
    multi_head_attention,
    positional_encoding,
    rms_norm,
    rotary_embedding,
    scaled_dot_product_attention,
    transformer_block,
    transformer_forward,
)

__all__ = [
    # mlp.py
    "init_mlp_params", "mlp_forward", "mlp_predict", "mlp_with_dropout", "create_mlp",
    "create_classifier", "create_regressor", "dense_layer", "activation_fn",
    "init_batch_norm", "batch_norm", "init_mlp_with_batch_norm", "mlp_with_batch_norm",
    "count_parameters", "get_layer_outputs",
    # cnn.py
    "init_conv_params", "conv2d", "conv2d_layer", "conv_output_shape", "pooling_layer",
    "global_average_pooling", "init_batch_norm_2d", "batch_norm_2d", "depthwise_conv2d",
    "separable_conv2d", "residual_block", "cnn_forward", "create_cnn",
    # transformer.py
    "init_attention_params", "init_feed_forward_params", "init_layer_norm_params",
    "init_transformer_block_params", "layer_norm", "rms_norm", "positional_encoding",
    "rotary_embedding", "create_causal_mask", "create_padding_mask", "combine_masks",
    "scaled_dot_product_attention", "multi_head_attention", "feed_forward_network",
    "transformer_block", "create_transformer", "transformer_forward",
]
