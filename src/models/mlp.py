# File location: jax-nsl/src/models/mlp.py

"""
Multi-layer perceptron implementation in pure JAX.

This module provides a clean, educational implementation of MLPs
with various activation functions and initialization schemes.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import List, Tuple, Callable, Dict, Any, Optional
from ..core.prng import glorot_uniform_init, he_normal_init


def init_mlp_params(key: jax.Array,
                   layer_sizes: List[int],
                   activation: str = 'relu',
                   output_activation: str = 'linear',
                   init_type: str = 'glorot') -> Dict[str, Any]:
    """Initialize MLP parameters.
    
    Args:
        key: Random key for initialization
        layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
        activation: Hidden layer activation function
        output_activation: Output layer activation
        init_type: Weight initialization scheme
        
    Returns:
        Dictionary of parameters
    """
    keys = jr.split(key, len(layer_sizes) - 1)
    params = {'weights': [], 'biases': []}
    
    for i, (key_i, in_size, out_size) in enumerate(zip(keys, layer_sizes[:-1], layer_sizes[1:])):
        w_key, b_key = jr.split(key_i, 2)
        
        # Initialize weights
        if init_type == 'glorot':
            W = glorot_uniform_init(w_key, (in_size, out_size))
        elif init_type == 'he':
            W = he_normal_init(w_key, (in_size, out_size))
        elif init_type == 'normal':
            W = jr.normal(w_key, (in_size, out_size)) * 0.1
        else:
            raise ValueError(f"Unknown initialization: {init_type}")
        
        # Initialize biases (typically zero)
        b = jnp.zeros(out_size)
        
        params['weights'].append(W)
        params['biases'].append(b)
    
    # Store activation functions
    params['activation'] = activation
    params['output_activation'] = output_activation
    
    return params


def activation_fn(x: jnp.ndarray, name: str) -> jnp.ndarray:
    """Apply activation function.
    
    Args:
        x: Input array
        name: Activation function name
        
    Returns:
        Activated output
    """
    if name == 'relu':
        return jax.nn.relu(x)
    elif name == 'tanh':
        return jnp.tanh(x)
    elif name == 'sigmoid':
        return jax.nn.sigmoid(x)
    elif name == 'gelu':
        return jax.nn.gelu(x)
    elif name == 'swish' or name == 'silu':
        return jax.nn.silu(x)
    elif name == 'elu':
        return jax.nn.elu(x)
    elif name == 'leaky_relu':
        return jax.nn.leaky_relu(x)
    elif name == 'linear' or name == 'identity':
        return x
    elif name == 'softmax':
        return jax.nn.softmax(x, axis=-1)
    else:
        raise ValueError(f"Unknown activation function: {name}")


def dense_layer(x: jnp.ndarray, 
               weights: jnp.ndarray, 
               bias: jnp.ndarray,
               activation: str = 'linear') -> jnp.ndarray:
    """Single dense layer computation.
    
    Args:
        x: Input features
        weights: Weight matrix
        bias: Bias vector
        activation: Activation function name
        
    Returns:
        Layer output
    """
    linear_output = x @ weights + bias
    return activation_fn(linear_output, activation)


def mlp_forward(params: Dict[str, Any], 
               x: jnp.ndarray,
               training: bool = True) -> jnp.ndarray:
    """Forward pass through MLP.
    
    Args:
        params: Model parameters
        x: Input batch with shape (batch_size, input_dim)
        training: Whether in training mode (affects dropout, etc.)
        
    Returns:
        Network output with shape (batch_size, output_dim)
    """
    weights = params['weights']
    biases = params['biases']
    activation = params['activation']
    output_activation = params['output_activation']
    
    h = x
    
    # Forward through hidden layers
    for i in range(len(weights) - 1):
        h = dense_layer(h, weights[i], biases[i], activation)
    
    # Output layer
    output = dense_layer(h, weights[-1], biases[-1], output_activation)
    
    return output


def mlp_predict(params: Dict[str, Any], x: jnp.ndarray) -> jnp.ndarray:
    """Prediction (inference mode).
    
    Args:
        params: Model parameters
        x: Input data
        
    Returns:
        Model predictions
    """
    return mlp_forward(params, x, training=False)


def create_mlp(layer_sizes: List[int],
              activation: str = 'relu',
              output_activation: str = 'linear',
              init_type: str = 'glorot',
              seed: int = 42) -> Tuple[Dict[str, Any], Callable, Callable]:
    """Create MLP with initialization and forward functions.
    
    Args:
        layer_sizes: Architecture specification
        activation: Hidden activation function
        output_activation: Output activation function
        init_type: Weight initialization
        seed: Random seed
        
    Returns:
        (params, forward_fn, predict_fn) tuple
    """
    key = jr.PRNGKey(seed)
    params = init_mlp_params(
        key, layer_sizes, activation, output_activation, init_type
    )
    
    def forward_fn(params, x, training=True):
        return mlp_forward(params, x, training)
    
    def predict_fn(params, x):
        return mlp_predict(params, x)
    
    return params, forward_fn, predict_fn


# Specialized MLP variants
def create_classifier(input_dim: int,
                     hidden_dims: List[int],
                     num_classes: int,
                     activation: str = 'relu',
                     seed: int = 42) -> Tuple[Dict[str, Any], Callable]:
    """Create classification MLP with softmax output.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: Hidden layer sizes
        num_classes: Number of output classes
        activation: Hidden activation function
        seed: Random seed
        
    Returns:
        (params, forward_fn) tuple
    """
    layer_sizes = [input_dim] + hidden_dims + [num_classes]
    params, forward_fn, _ = create_mlp(
        layer_sizes, 
        activation=activation,
        output_activation='softmax',
        seed=seed
    )
    
    return params, forward_fn


def create_regressor(input_dim: int,
                    hidden_dims: List[int],
                    output_dim: int = 1,
                    activation: str = 'relu',
                    seed: int = 42) -> Tuple[Dict[str, Any], Callable]:
    """Create regression MLP with linear output.
    
    Args:
        input_dim: Input feature dimension
        hidden_dims: Hidden layer sizes
        output_dim: Output dimension
        activation: Hidden activation function
        seed: Random seed
        
    Returns:
        (params, forward_fn) tuple
    """
    layer_sizes = [input_dim] + hidden_dims + [output_dim]
    params, forward_fn, _ = create_mlp(
        layer_sizes,
        activation=activation,
        output_activation='linear', 
        seed=seed
    )
    
    return params, forward_fn


def mlp_with_dropout(params: Dict[str, Any],
                    x: jnp.ndarray,
                    key: jax.Array,
                    dropout_rate: float = 0.1,
                    training: bool = True) -> jnp.ndarray:
    """MLP forward pass with dropout.
    
    Args:
        params: Model parameters
        x: Input batch
        key: Random key for dropout
        dropout_rate: Dropout probability
        training: Whether in training mode
        
    Returns:
        Network output with dropout applied
    """
    weights = params['weights']
    biases = params['biases']
    activation = params['activation']
    output_activation = params['output_activation']
    
    keys = jr.split(key, len(weights))
    h = x
    
    # Forward through hidden layers with dropout
    for i in range(len(weights) - 1):
        h = dense_layer(h, weights[i], biases[i], activation)
        
        if training and dropout_rate > 0:
            # Apply dropout
            keep_prob = 1.0 - dropout_rate
            mask = jr.bernoulli(keys[i], keep_prob, h.shape)
            h = jnp.where(mask, h / keep_prob, 0.0)
    
    # Output layer (no dropout)
    output = dense_layer(h, weights[-1], biases[-1], output_activation)
    
    return output


def mlp_with_batch_norm(params: Dict[str, Any],
                       x: jnp.ndarray,
                       training: bool = True) -> jnp.ndarray:
    """MLP with batch normalization (simplified version).
    
    Args:
        params: Model parameters (should include batch norm params)
        x: Input batch
        training: Whether in training mode
        
    Returns:
        Network output with batch normalization
    """
    # This is a simplified implementation
    # In practice, you'd need running statistics for inference
    
    weights = params['weights']
    biases = params['biases']
    activation = params['activation']
    output_activation = params['output_activation']
    
    h = x
    
    # Forward through hidden layers with batch norm
    for i in range(len(weights) - 1):
        # Linear transformation
        h = h @ weights[i] + biases[i]
        
        # Batch normalization
        if training:
            mean = jnp.mean(h, axis=0, keepdims=True)
            var = jnp.var(h, axis=0, keepdims=True)
        else:
            # In practice, use running statistics
            mean = 0.0
            var = 1.0
        
        h = (h - mean) / jnp.sqrt(var + 1e-5)
        
        # Activation
        h = activation_fn(h, activation)
    
    # Output layer
    output = dense_layer(h, weights[-1], biases[-1], output_activation)
    
    return output


def count_parameters(params: Dict[str, Any]) -> int:
    """Count total number of parameters in MLP.
    
    Args:
        params: Model parameters
        
    Returns:
        Total parameter count
    """
    total = 0
    for W, b in zip(params['weights'], params['biases']):
        total += W.size + b.size
    return total


def get_layer_outputs(params: Dict[str, Any],
                     x: jnp.ndarray) -> List[jnp.ndarray]:
    """Get outputs from all layers (for visualization/analysis).
    
    Args:
        params: Model parameters
        x: Input batch
        
    Returns:
        List of outputs from each layer
    """
    weights = params['weights']
    biases = params['biases']
    activation = params['activation']
    output_activation = params['output_activation']
    
    layer_outputs = [x]
    h = x
    
    # Forward through hidden layers
    for i in range(len(weights) - 1):
        h = dense_layer(h, weights[i], biases[i], activation)
        layer_outputs.append(h)
    
    # Output layer
    output = dense_layer(h, weights[-1], biases[-1], output_activation)
    layer_outputs.append(output)
    
    return layer_outputs