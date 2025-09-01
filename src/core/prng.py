# File location: jax-nsl/src/core/prng.py

"""
PRNG key handling and common patterns.

This module provides utilities for managing JAX's pseudo-random number
generation, including key splitting, sequences, and common patterns.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import tree_util
from typing import Any, Iterator, Tuple, Union, Optional, Dict
from collections.abc import Sequence


class PRNGSequence:
    """Iterator that generates an infinite sequence of PRNG keys.
    
    Useful for situations where you need a stream of random keys
    without explicitly managing key splitting.
    
    Example:
        rng = PRNGSequence(42)
        key1 = next(rng)
        key2 = next(rng) 
        # keys are independent
    """
    
    def __init__(self, seed: Union[int, jax.Array]):
        """Initialize PRNG sequence.
        
        Args:
            seed: Initial seed or PRNGKey
        """
        if isinstance(seed, int):
            self._key = jr.PRNGKey(seed)
        else:
            self._key = seed
    
    def __iter__(self) -> Iterator[jax.Array]:
        return self
    
    def __next__(self) -> jax.Array:
        """Get next PRNG key in sequence."""
        self._key, subkey = jr.split(self._key)
        return subkey
    
    def split(self, num: int) -> jax.Array:
        """Split into multiple keys at once.
        
        Args:
            num: Number of keys to generate
            
        Returns:
            Array of PRNG keys with shape (num, 2)
        """
        self._key, *subkeys = jr.split(self._key, num + 1)
        return jnp.stack(subkeys)
    
    def fork(self, num: int) -> 'PRNGSequence':
        """Create independent PRNG sequences.
        
        Args:
            num: Number of independent sequences
            
        Returns:
            List of independent PRNGSequence objects
        """
        keys = self.split(num)
        return [PRNGSequence(key) for key in keys]


def split_key_tree(key: jax.Array, tree_structure: Any) -> Any:
    """Split a PRNG key according to pytree structure.
    
    Args:
        key: Master PRNG key
        tree_structure: PyTree defining the split structure
        
    Returns:
        PyTree with same structure containing PRNG keys
    """
    leaves = tree_util.tree_leaves(tree_structure)
    num_leaves = len(leaves)
    
    if num_leaves == 0:
        return tree_structure
    
    keys = jr.split(key, num_leaves)
    return tree_util.tree_unflatten(
        tree_util.tree_structure(tree_structure),
        keys
    )


def random_like(key: jax.Array, 
                template: jax.Array, 
                distribution: str = 'normal',
                **kwargs) -> jax.Array:
    """Generate random array with same shape/dtype as template.
    
    Args:
        key: PRNG key
        template: Template array for shape/dtype
        distribution: Distribution name ('normal', 'uniform', 'bernoulli')
        **kwargs: Distribution-specific parameters
        
    Returns:
        Random array with same shape/dtype as template
    """
    shape = template.shape
    dtype = template.dtype
    
    if distribution == 'normal':
        arr = jr.normal(key, shape, dtype=dtype)
        if 'scale' in kwargs:
            arr = arr * kwargs['scale']
        if 'loc' in kwargs:
            arr = arr + kwargs['loc']
        return arr
    
    elif distribution == 'uniform':
        minval = kwargs.get('minval', 0.0)
        maxval = kwargs.get('maxval', 1.0)
        return jr.uniform(key, shape, dtype=dtype, minval=minval, maxval=maxval)
    
    elif distribution == 'bernoulli':
        p = kwargs.get('p', 0.5)
        return jr.bernoulli(key, p, shape).astype(dtype)
    
    elif distribution == 'categorical':
        logits = kwargs.get('logits')
        if logits is None:
            raise ValueError("categorical distribution requires 'logits' parameter")
        return jr.categorical(key, logits, shape=shape).astype(dtype)
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def make_rng_state(seed: Union[int, jax.Array], 
                   names: Sequence[str]) -> Dict[str, jax.Array]:
    """Create dictionary of named PRNG keys.
    
    Args:
        seed: Master seed
        names: Names for the different RNG streams
        
    Returns:
        Dictionary mapping names to PRNG keys
    """
    if isinstance(seed, int):
        master_key = jr.PRNGKey(seed)
    else:
        master_key = seed
    
    keys = jr.split(master_key, len(names))
    return dict(zip(names, keys))


# Common initialization patterns
def glorot_uniform_init(key: jax.Array, 
                       shape: Tuple[int, ...], 
                       dtype: jnp.dtype = jnp.float32,
                       in_axis: int = -2,
                       out_axis: int = -1) -> jax.Array:
    """Glorot (Xavier) uniform initialization.
    
    Args:
        key: PRNG key
        shape: Parameter shape
        dtype: Parameter dtype
        in_axis: Input dimension axis
        out_axis: Output dimension axis
        
    Returns:
        Initialized parameter array
    """
    fan_in = shape[in_axis]
    fan_out = shape[out_axis]
    denominator = fan_in + fan_out
    variance = 2.0 / denominator
    bound = jnp.sqrt(3.0 * variance)
    return jr.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def glorot_normal_init(key: jax.Array, 
                      shape: Tuple[int, ...], 
                      dtype: jnp.dtype = jnp.float32,
                      in_axis: int = -2,
                      out_axis: int = -1) -> jax.Array:
    """Glorot (Xavier) normal initialization.
    
    Args:
        key: PRNG key
        shape: Parameter shape
        dtype: Parameter dtype
        in_axis: Input dimension axis
        out_axis: Output dimension axis
        
    Returns:
        Initialized parameter array
    """
    fan_in = shape[in_axis]
    fan_out = shape[out_axis]
    denominator = fan_in + fan_out
    variance = 2.0 / denominator
    stddev = jnp.sqrt(variance)
    return jr.normal(key, shape, dtype) * stddev


def he_uniform_init(key: jax.Array, 
                   shape: Tuple[int, ...], 
                   dtype: jnp.dtype = jnp.float32,
                   in_axis: int = -2) -> jax.Array:
    """He (Kaiming) uniform initialization for ReLU activations.
    
    Args:
        key: PRNG key
        shape: Parameter shape
        dtype: Parameter dtype
        in_axis: Input dimension axis
        
    Returns:
        Initialized parameter array
    """
    fan_in = shape[in_axis]
    variance = 2.0 / fan_in
    bound = jnp.sqrt(3.0 * variance)
    return jr.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def he_normal_init(key: jax.Array, 
                  shape: Tuple[int, ...], 
                  dtype: jnp.dtype = jnp.float32,
                  in_axis: int = -2) -> jax.Array:
    """He (Kaiming) normal initialization for ReLU activations.
    
    Args:
        key: PRNG key
        shape: Parameter shape
        dtype: Parameter dtype
        in_axis: Input dimension axis
        
    Returns:
        Initialized parameter array
    """
    fan_in = shape[in_axis]
    stddev = jnp.sqrt(2.0 / fan_in)
    return jr.normal(key, shape, dtype) * stddev


def lecun_uniform_init(key: jax.Array, 
                      shape: Tuple[int, ...], 
                      dtype: jnp.dtype = jnp.float32,
                      in_axis: int = -2) -> jax.Array:
    """LeCun uniform initialization.
    
    Args:
        key: PRNG key
        shape: Parameter shape
        dtype: Parameter dtype
        in_axis: Input dimension axis
        
    Returns:
        Initialized parameter array
    """
    fan_in = shape[in_axis]
    variance = 1.0 / fan_in
    bound = jnp.sqrt(3.0 * variance)
    return jr.uniform(key, shape, dtype, minval=-bound, maxval=bound)


def lecun_normal_init(key: jax.Array, 
                     shape: Tuple[int, ...], 
                     dtype: jnp.dtype = jnp.float32,
                     in_axis: int = -2) -> jax.Array:
    """LeCun normal initialization.
    
    Args:
        key: PRNG key
        shape: Parameter shape
        dtype: Parameter dtype
        in_axis: Input dimension axis
        
    Returns:
        Initialized parameter array
    """
    fan_in = shape[in_axis]
    stddev = jnp.sqrt(1.0 / fan_in)
    return jr.normal(key, shape, dtype) * stddev