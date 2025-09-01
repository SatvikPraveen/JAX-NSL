# File location: jax-nsl/src/models/transformer.py

"""
Transformer attention mechanisms from scratch.

This module implements multi-head attention, positional encoding,
and transformer blocks without high-level frameworks.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Dict, Any, Optional, Tuple
from ..core.prng import glorot_uniform_init
from ..core.numerics import softmax_stable
import math


def init_attention_params(key: jax.Array,
                         d_model: int,
                         num_heads: int,
                         d_ff: Optional[int] = None) -> Dict[str, jnp.ndarray]:
    """Initialize multi-head attention parameters.
    
    Args:
        key: Random key
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension (default: 4 * d_model)
        
    Returns:
        Attention parameters dictionary
    """
    if d_ff is None:
        d_ff = 4 * d_model
    
    keys = jr.split(key, 8)
    
    d_k = d_model // num_heads
    
    params = {
        # Multi-head attention
        'W_q': glorot_uniform_init(keys[0], (d_model, d_model)),
        'W_k': glorot_uniform_init(keys[1], (d_model, d_model)),
        'W_v': glorot_uniform_init(keys[2], (d_model, d_model)),
        'W_o': glorot_uniform_init(keys[3], (d_model, d_model)),
        
        # Feed-forward network
        'W_ff1': glorot_uniform_init(keys[4], (d_model, d_ff)),
        'b_ff1': jnp.zeros(d_ff),
        'W_ff2': glorot_uniform_init(keys[5], (d_ff, d_model)),
        'b_ff2': jnp.zeros(d_model),
        
        # Layer normalization
        'ln1_scale': jnp.ones(d_model),
        'ln1_bias': jnp.zeros(d_model),
        'ln2_scale': jnp.ones(d_model),
        'ln2_bias': jnp.zeros(d_model),
    }
    
    return params


def layer_norm(x: jnp.ndarray,
              scale: jnp.ndarray,
              bias: jnp.ndarray,
              epsilon: float = 1e-6) -> jnp.ndarray:
    """Layer normalization.
    
    Args:
        x: Input tensor (..., d_model)
        scale: Scale parameters
        bias: Bias parameters
        epsilon: Small constant for numerical stability
        
    Returns:
        Layer normalized output
    """
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    normalized = (x - mean) / jnp.sqrt(var + epsilon)
    return scale * normalized + bias


def scaled_dot_product_attention(query: jnp.ndarray,
                                key: jnp.ndarray,
                                value: jnp.ndarray,
                                mask: Optional[jnp.ndarray] = None,
                                dropout_rate: float = 0.0,
                                key_rng: Optional[jax.Array] = None,
                                training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Scaled dot-product attention.
    
    Args:
        query: Query tensor (batch, seq_len, d_k)
        key: Key tensor (batch, seq_len, d_k)
        value: Value tensor (batch, seq_len, d_v)
        mask: Optional attention mask
        dropout_rate: Dropout rate for attention weights
        key_rng: Random key for dropout
        training: Training mode flag
        
    Returns:
        (attention_output, attention_weights) tuple
    """
    d_k = query.shape[-1]
    
    # Compute attention scores
    scores = jnp.matmul(query, jnp.transpose(key, (0, 2, 1))) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = jnp.where(mask, scores, -jnp.inf)
    
    # Softmax to get attention weights
    attention_weights = softmax_stable(scores, axis=-1)
    
    # Apply dropout to attention weights
    if training and dropout_rate > 0.0 and key_rng is not None:
        keep_prob = 1.0 - dropout_rate
        dropout_mask = jr.bernoulli(key_rng, keep_prob, attention_weights.shape)
        attention_weights = jnp.where(dropout_mask, attention_weights / keep_prob, 0.0)
    
    # Apply attention to values
    output = jnp.matmul(attention_weights, value)
    
    return output, attention_weights


def multi_head_attention(x: jnp.ndarray,
                        params: Dict[str, jnp.ndarray],
                        num_heads: int,
                        mask: Optional[jnp.ndarray] = None,
                        dropout_rate: float = 0.0,
                        key_rng: Optional[jax.Array] = None,
                        training: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Multi-head attention mechanism.
    
    Args:
        x: Input tensor (batch, seq_len, d_model)
        params: Attention parameters
        num_heads: Number of attention heads
        mask: Optional attention mask
        dropout_rate: Dropout rate
        key_rng: Random key for dropout
        training: Training mode flag
        
    Returns:
        (output, attention_weights) tuple
    """
    batch_size, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    
    # Linear projections
    Q = jnp.matmul(x, params['W_q'])  # (batch, seq_len, d_model)
    K = jnp.matmul(x, params['W_k'])
    V = jnp.matmul(x, params['W_v'])
    
    # Reshape for multi-head attention
    Q = Q.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Apply attention to each head
    def attention_head(qkv_head):
        q_head, k_head, v_head = qkv_head
        head_output, head_weights = scaled_dot_product_attention(
            q_head, k_head, v_head, mask, dropout_rate, key_rng, training
        )
        return head_output, head_weights
    
    # Vectorize over heads
    head_outputs, attention_weights = jax.vmap(attention_head)((Q, K, V))
    
    # Concatenate heads
    concat_output = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # Final linear projection
    output = jnp.matmul(concat_output, params['W_o'])
    
    return output, attention_weights


def feed_forward_network(x: jnp.ndarray,
                        params: Dict[str, jnp.ndarray],
                        activation: str = 'relu') -> jnp.ndarray:
    """Position-wise feed-forward network.
    
    Args:
        x: Input tensor
        params: FFN parameters (W_ff1, b_ff1, W_ff2, b_ff2)
        activation: Activation function
        
    Returns:
        FFN output
    """
    # First linear layer
    hidden = jnp.matmul(x, params['W_ff1']) + params['b_ff1']
    
    # Activation
    if activation == 'relu':
        hidden = jax.nn.relu(hidden)
    elif activation == 'gelu':
        hidden = jax.nn.gelu(hidden)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    
    # Second linear layer
    output = jnp.matmul(hidden, params['W_ff2']) + params['b_ff2']
    
    return output


def transformer_block(x: jnp.ndarray,
                     params: Dict[str, jnp.ndarray],
                     num_heads: int,
                     mask: Optional[jnp.ndarray] = None,
                     dropout_rate: float = 0.1,
                     key_rng: Optional[jax.Array] = None,
                     training: bool = True) -> jnp.ndarray:
    """Single transformer block with residual connections.
    
    Args:
        x: Input tensor (batch, seq_len, d_model)
        params: Block parameters
        num_heads: Number of attention heads
        mask: Optional attention mask
        dropout_rate: Dropout rate
        key_rng: Random key
        training: Training mode flag
        
    Returns:
        Transformer block output
    """
    # Multi-head self-attention with residual connection
    attn_output, _ = multi_head_attention(
        x, params, num_heads, mask, dropout_rate, key_rng, training
    )
    
    # Add & Norm
    x = x + attn_output
    x = layer_norm(x, params['ln1_scale'], params['ln1_bias'])
    
    # Feed-forward network with residual connection
    ff_output = feed_forward_network(x, params)
    
    # Add & Norm
    x = x + ff_output
    x = layer_norm(x, params['ln2_scale'], params['ln2_bias'])
    
    return x


def positional_encoding(seq_len: int,
                       d_model: int,
                       max_len: int = 10000) -> jnp.ndarray:
    """Generate sinusoidal positional encoding.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        max_len: Maximum sequence length for encoding
        
    Returns:
        Positional encoding matrix (seq_len, d_model)
    """
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * -(math.log(max_len) / d_model))
    
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    
    return pe


def create_transformer(d_model: int,
                      num_heads: int,
                      num_layers: int,
                      d_ff: Optional[int] = None,
                      max_seq_len: int = 1024,
                      vocab_size: Optional[int] = None,
                      seed: int = 42) -> Tuple[Dict[str, Any], callable]:
    """Create transformer model.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size (for embedding layer)
        seed: Random seed
        
    Returns:
        (params, forward_fn) tuple
    """
    key = jr.PRNGKey(seed)
    keys = jr.split(key, num_layers + 2)
    
    params = {'layers': []}
    
    # Initialize transformer layers
    for i in range(num_layers):
        layer_params = init_attention_params(keys[i], d_model, num_heads, d_ff)
        params['layers'].append(layer_params)
    
    # Optional embedding layer
    if vocab_size is not None:
        params['embedding'] = glorot_uniform_init(keys[-2], (vocab_size, d_model))
    
    # Positional encoding (fixed, not learned)
    params['pos_encoding'] = positional_encoding(max_seq_len, d_model)
    
    def forward_fn(params, x, mask=None, training=True):
        return transformer_forward(params, x, num_heads, mask, training)
    
    return params, forward_fn


def transformer_forward(params: Dict[str, Any],
                       x: jnp.ndarray,
                       num_heads: int,
                       mask: Optional[jnp.ndarray] = None,
                       training: bool = True) -> jnp.ndarray:
    """Forward pass through transformer.
    
    Args:
        params: Model parameters
        x: Input tensor (batch, seq_len) or (batch, seq_len, d_model)
        num_heads: Number of attention heads
        mask: Optional attention mask
        training: Training mode flag
        
    Returns:
        Transformer output
    """
    # Token embedding if input is token indices
    if 'embedding' in params and x.ndim == 2:
        x = params['embedding'][x]  # (batch, seq_len, d_model)
    
    seq_len = x.shape[1]
    
    # Add positional encoding
    pos_enc = params['pos_encoding'][:seq_len]
    x = x + pos_enc
    
    # Apply transformer layers
    for layer_params in params['layers']:
        x = transformer_block(x, layer_params, num_heads, mask, training=training)
    
    return x


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Create causal (lower triangular) attention mask.
    
    Args:
        seq_len: Sequence length
        
    Returns:
        Causal mask (seq_len, seq_len)
    """
    return jnp.tril(jnp.ones((seq_len, seq_len)))


def create_padding_mask(tokens: jnp.ndarray, pad_token: int = 0) -> jnp.ndarray:
    """Create padding mask for variable-length sequences.
    
    Args:
        tokens: Token tensor (batch, seq_len)
        pad_token: Padding token ID
        
    Returns:
        Padding mask (batch, seq_len)
    """
    return tokens != pad_token