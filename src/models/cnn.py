# File location: jax-nsl/src/models/cnn.py

"""
Convolutional neural network implementation using lax.conv_general_dilated.

This module provides CNN building blocks with proper weight initialization
and efficient convolution operations.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from typing import Tuple, Dict, Any, Optional, List
from ..core.prng import glorot_uniform_init, he_normal_init


def init_conv_params(key: jax.Array,
                    kernel_shape: Tuple[int, int],
                    in_channels: int,
                    out_channels: int,
                    init_type: str = 'he') -> Dict[str, jnp.ndarray]:
    """Initialize convolutional layer parameters.
    
    Args:
        key: Random key
        kernel_shape: (height, width) of kernel
        in_channels: Input channels
        out_channels: Output channels
        init_type: Initialization scheme
        
    Returns:
        Dictionary with weights and biases
    """
    w_key, b_key = jr.split(key, 2)
    
    # Weight shape: (out_channels, in_channels, height, width)
    weight_shape = (out_channels, in_channels) + kernel_shape
    
    if init_type == 'he':
        weights = he_normal_init(w_key, weight_shape, in_axis=1)
    elif init_type == 'glorot':
        weights = glorot_uniform_init(w_key, weight_shape, in_axis=1, out_axis=0)
    else:
        weights = jr.normal(w_key, weight_shape) * 0.1
    
    biases = jnp.zeros(out_channels)
    
    return {'weights': weights, 'biases': biases}


def conv2d_layer(x: jnp.ndarray,
                params: Dict[str, jnp.ndarray],
                stride: Tuple[int, int] = (1, 1),
                padding: str = 'SAME',
                dilation: Tuple[int, int] = (1, 1),
                activation: str = 'relu') -> jnp.ndarray:
    """2D convolution layer.
    
    Args:
        x: Input with shape (batch, height, width, channels)
        params: Layer parameters
        stride: Convolution stride
        padding: Padding type ('SAME' or 'VALID')
        dilation: Dilation factors
        activation: Activation function
        
    Returns:
        Convolved output
    """
    weights = params['weights']
    biases = params['biases']
    
    # JAX conv expects (batch, in_channels, height, width)
    x_transposed = jnp.transpose(x, (0, 3, 1, 2))
    
    # Convolution
    out = lax.conv_general_dilated(
        x_transposed,
        weights,
        window_strides=stride,
        padding=padding,
        lhs_dilation=(1, 1),
        rhs_dilation=dilation,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    
    # Add bias
    out = out + biases.reshape(1, -1, 1, 1)
    
    # Transpose back to (batch, height, width, channels)
    out = jnp.transpose(out, (0, 2, 3, 1))
    
    # Apply activation
    if activation == 'relu':
        out = jax.nn.relu(out)
    elif activation == 'linear':
        pass
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    return out


def pooling_layer(x: jnp.ndarray,
                 pool_type: str = 'max',
                 window_shape: Tuple[int, int] = (2, 2),
                 stride: Tuple[int, int] = (2, 2),
                 padding: str = 'VALID') -> jnp.ndarray:
    """Pooling layer (max or average).
    
    Args:
        x: Input with shape (batch, height, width, channels)
        pool_type: 'max' or 'avg'
        window_shape: Pooling window size
        stride: Pooling stride
        padding: Padding type
        
    Returns:
        Pooled output
    """
    # Transpose to NCHW for lax operations
    x_transposed = jnp.transpose(x, (0, 3, 1, 2))
    
    if pool_type == 'max':
        out = lax.reduce_window(
            x_transposed,
            -jnp.inf,
            lax.max,
            window_dimensions=(1, 1) + window_shape,
            window_strides=(1, 1) + stride,
            padding=padding
        )
    elif pool_type == 'avg':
        out = lax.reduce_window(
            x_transposed,
            0.0,
            lax.add,
            window_dimensions=(1, 1) + window_shape,
            window_strides=(1, 1) + stride,
            padding=padding
        )
        # Divide by window size for average
        window_size = window_shape[0] * window_shape[1]
        out = out / window_size
    else:
        raise ValueError(f"Unknown pool type: {pool_type}")
    
    # Transpose back to NHWC
    return jnp.transpose(out, (0, 2, 3, 1))


def batch_norm_2d(x: jnp.ndarray,
                 params: Dict[str, jnp.ndarray],
                 training: bool = True,
                 momentum: float = 0.9,
                 epsilon: float = 1e-5) -> jnp.ndarray:
    """2D Batch normalization.
    
    Args:
        x: Input with shape (batch, height, width, channels)
        params: BatchNorm parameters (scale, offset, running_mean, running_var)
        training: Training mode flag
        momentum: Momentum for running statistics
        epsilon: Small constant for numerical stability
        
    Returns:
        Normalized output
    """
    scale = params.get('scale', jnp.ones(x.shape[-1]))
    offset = params.get('offset', jnp.zeros(x.shape[-1]))
    
    if training:
        # Compute batch statistics
        mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(x, axis=(0, 1, 2), keepdims=True)
        
        # Update running statistics (in practice, this would be stateful)
        if 'running_mean' in params:
            running_mean = momentum * params['running_mean'] + (1 - momentum) * mean
            running_var = momentum * params['running_var'] + (1 - momentum) * var
    else:
        # Use running statistics
        mean = params.get('running_mean', 0.0)
        var = params.get('running_var', 1.0)
    
    # Normalize
    x_norm = (x - mean) / jnp.sqrt(var + epsilon)
    
    # Scale and shift
    return scale * x_norm + offset


def create_cnn(conv_layers: List[Dict],
              dense_layers: List[int],
              num_classes: int,
              seed: int = 42) -> Tuple[Dict[str, Any], callable]:
    """Create CNN with specified architecture.
    
    Args:
        conv_layers: List of conv layer specs [{'channels': 32, 'kernel': (3,3), 'stride': (1,1)}, ...]
        dense_layers: Hidden dimensions for dense layers
        num_classes: Number of output classes
        seed: Random seed
        
    Returns:
        (params, forward_fn) tuple
    """
    key = jr.PRNGKey(seed)
    keys = jr.split(key, len(conv_layers) + len(dense_layers) + 1)
    
    params = {'conv_layers': [], 'dense_layers': []}
    
    # Initialize conv layers
    in_channels = 3  # Assume RGB input
    for i, layer_spec in enumerate(conv_layers):
        out_channels = layer_spec['channels']
        kernel_shape = layer_spec['kernel']
        
        layer_params = init_conv_params(
            keys[i], kernel_shape, in_channels, out_channels
        )
        params['conv_layers'].append(layer_params)
        in_channels = out_channels
    
    # Initialize dense layers (after flattening)
    # Note: in_features would need to be computed based on input size and conv operations
    in_features = 512  # Placeholder - should be computed
    dense_keys = keys[len(conv_layers):]
    
    for i, out_features in enumerate(dense_layers + [num_classes]):
        w_key, b_key = jr.split(dense_keys[i], 2)
        
        weights = glorot_uniform_init(w_key, (in_features, out_features))
        biases = jnp.zeros(out_features)
        
        params['dense_layers'].append({
            'weights': weights,
            'biases': biases
        })
        in_features = out_features
    
    def forward_fn(params, x, training=True):
        return cnn_forward(params, x, training)
    
    return params, forward_fn


def cnn_forward(params: Dict[str, Any],
               x: jnp.ndarray,
               training: bool = True) -> jnp.ndarray:
    """Forward pass through CNN.
    
    Args:
        params: Network parameters
        x: Input batch (batch, height, width, channels)
        training: Training mode flag
        
    Returns:
        Network output
    """
    # Convolutional layers
    h = x
    for i, layer_params in enumerate(params['conv_layers']):
        h = conv2d_layer(h, layer_params, activation='relu')
        h = pooling_layer(h, pool_type='max')
    
    # Flatten for dense layers
    h = h.reshape(h.shape[0], -1)
    
    # Dense layers
    for i, layer_params in enumerate(params['dense_layers']):
        weights = layer_params['weights']
        biases = layer_params['biases']
        
        h = h @ weights + biases
        
        # Apply activation (relu for hidden, linear for output)
        if i < len(params['dense_layers']) - 1:
            h = jax.nn.relu(h)
    
    return h


def depthwise_conv2d(x: jnp.ndarray,
                    kernel: jnp.ndarray,
                    stride: Tuple[int, int] = (1, 1),
                    padding: str = 'SAME') -> jnp.ndarray:
    """Depthwise convolution.
    
    Args:
        x: Input (batch, height, width, channels)
        kernel: Depthwise kernel (height, width, channels, multiplier)
        stride: Convolution stride
        padding: Padding type
        
    Returns:
        Depthwise convolved output
    """
    # Transpose to NCHW
    x_t = jnp.transpose(x, (0, 3, 1, 2))
    
    # Reshape kernel for conv_general_dilated
    # kernel: (height, width, in_channels, multiplier) -> (in_channels*multiplier, 1, height, width)
    h, w, in_ch, mult = kernel.shape
    kernel_reshaped = kernel.transpose(2, 3, 0, 1).reshape(in_ch * mult, 1, h, w)
    
    # Depthwise convolution
    out = lax.conv_general_dilated(
        x_t,
        kernel_reshaped,
        window_strides=stride,
        padding=padding,
        feature_group_count=in_ch,
        dimension_numbers=('NCHW', 'OIHW', 'NCHW')
    )
    
    return jnp.transpose(out, (0, 2, 3, 1))


def separable_conv2d(x: jnp.ndarray,
                    depthwise_kernel: jnp.ndarray,
                    pointwise_kernel: jnp.ndarray,
                    stride: Tuple[int, int] = (1, 1),
                    padding: str = 'SAME') -> jnp.ndarray:
    """Separable convolution (depthwise + pointwise).
    
    Args:
        x: Input tensor
        depthwise_kernel: Depthwise convolution kernel
        pointwise_kernel: 1x1 convolution kernel
        stride: Stride for depthwise conv
        padding: Padding type
        
    Returns:
        Separable convolution output
    """
    # Depthwise convolution
    depthwise_out = depthwise_conv2d(x, depthwise_kernel, stride, padding)
    
    # Pointwise convolution (1x1)
    pointwise_params = {'weights': pointwise_kernel, 'biases': jnp.zeros(pointwise_kernel.shape[0])}
    return conv2d_layer(depthwise_out, pointwise_params, stride=(1, 1), activation='linear')


def global_average_pooling(x: jnp.ndarray) -> jnp.ndarray:
    """Global average pooling.
    
    Args:
        x: Input (batch, height, width, channels)
        
    Returns:
        Globally averaged features (batch, channels)
    """
    return jnp.mean(x, axis=(1, 2))


def residual_block(x: jnp.ndarray,
                  params1: Dict[str, jnp.ndarray],
                  params2: Dict[str, jnp.ndarray],
                  shortcut_params: Optional[Dict[str, jnp.ndarray]] = None) -> jnp.ndarray:
    """Residual block with skip connection.
    
    Args:
        x: Input tensor
        params1: First conv layer parameters
        params2: Second conv layer parameters
        shortcut_params: Optional shortcut connection parameters
        
    Returns:
        Residual block output
    """
    # Main path
    h = conv2d_layer(x, params1, activation='relu')
    h = conv2d_layer(h, params2, activation='linear')
    
    # Skip connection
    if shortcut_params is not None:
        # Project shortcut if needed
        skip = conv2d_layer(x, shortcut_params, activation='linear')
    else:
        skip = x
    
    # Add and apply activation
    return jax.nn.relu(h + skip)