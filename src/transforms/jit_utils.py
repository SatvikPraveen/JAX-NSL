# File location: jax-nsl/src/transforms/jit_utils.py

"""
JIT compilation utilities with static args and donation patterns.

This module provides enhanced JIT compilation utilities that handle
common patterns like static arguments, argument donation, and caching.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import Callable, Any, Optional, Union, Tuple, Set, Dict
import functools
import hashlib
import pickle


def smart_jit(fun: Callable,
              static_argnums: Optional[Union[int, Tuple[int, ...]]] = None,
              static_argnames: Optional[Union[str, Tuple[str, ...]]] = None,
              donate_argnums: Optional[Union[int, Tuple[int, ...]]] = None,
              donate_argnames: Optional[Union[str, Tuple[str, ...]]] = None,
              device: Optional[jax.Device] = None,
              backend: Optional[str] = None,
              inline: bool = False) -> Callable:
    """Smart JIT wrapper with enhanced argument handling.
    
    Args:
        fun: Function to JIT compile
        static_argnums: Argument positions to treat as static
        static_argnames: Argument names to treat as static  
        donate_argnums: Argument positions to donate buffers
        donate_argnames: Argument names to donate buffers
        device: Target device for computation
        backend: Target backend
        inline: Whether to inline the function
        
    Returns:
        JIT-compiled function
    """
    return jit(
        fun,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        device=device,
        backend=backend,
        inline=inline
    )


def conditional_jit(condition: bool = True) -> Callable:
    """Conditionally apply JIT compilation.
    
    Useful for debugging where you want to disable JIT temporarily.
    
    Args:
        condition: Whether to apply JIT compilation
        
    Returns:
        JIT decorator if condition is True, identity otherwise
    """
    def decorator(fun: Callable) -> Callable:
        if condition:
            return jit(fun)
        else:
            return fun
    return decorator


def donate_argnums_jit(argnums: Union[int, Tuple[int, ...]]) -> Callable:
    """JIT decorator with argument donation for memory efficiency.
    
    Args:
        argnums: Argument positions to donate
        
    Returns:
        JIT decorator with donation
    """
    def decorator(fun: Callable) -> Callable:
        return jit(fun, donate_argnums=argnums)
    return decorator


def static_argnums_jit(argnums: Union[int, Tuple[int, ...]]) -> Callable:
    """JIT decorator with static arguments.
    
    Args:
        argnums: Argument positions to treat as static
        
    Returns:
        JIT decorator with static args
    """
    def decorator(fun: Callable) -> Callable:
        return jit(fun, static_argnums=argnums)
    return decorator


class JITCache:
    """Cache for JIT-compiled functions to avoid recompilation."""
    
    def __init__(self, max_size: int = 128):
        self.cache: Dict[str, Callable] = {}
        self.max_size = max_size
        self.access_order = []
    
    def _get_key(self, fun: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function and arguments."""
        # This is a simplified key generation - in practice you'd want
        # more sophisticated handling of argument types and values
        key_data = {
            'fun_name': fun.__name__,
            'args_types': [type(arg).__name__ for arg in args],
            'kwargs_keys': list(kwargs.keys()) if kwargs else []
        }
        return hashlib.md5(str(key_data).encode()).hexdigest()
    
    def get_or_compile(self, fun: Callable, *args, **kwargs) -> Callable:
        """Get cached compiled function or compile and cache."""
        key = self._get_key(fun, args, kwargs)
        
        if key in self.cache:
            # Move to end for LRU tracking
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        
        # Compile and cache
        compiled_fun = jit(fun)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = compiled_fun
        self.access_order.append(key)
        
        return compiled_fun


# Global JIT cache instance
_global_jit_cache = JITCache()


def jit_with_cache(fun: Callable = None, *, use_global_cache: bool = True) -> Callable:
    """JIT compile with caching to avoid recompilation overhead.
    
    Args:
        fun: Function to compile (if used as decorator without args)
        use_global_cache: Whether to use global cache instance
        
    Returns:
        Cached JIT-compiled function
    """
    def decorator(f: Callable) -> Callable:
        cache = _global_jit_cache if use_global_cache else JITCache()
        
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            compiled_f = cache.get_or_compile(f, *args, **kwargs)
            return compiled_f(*args, **kwargs)
        
        return wrapper
    
    if fun is None:
        # Called as @jit_with_cache()
        return decorator
    else:
        # Called as @jit_with_cache
        return decorator(fun)


def profile_jit_compilation(fun: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Profile JIT compilation time vs execution time.
    
    Args:
        fun: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary with timing information
    """
    import time
    
    # Time compilation
    start_compile = time.time()
    compiled_fun = jit(fun)
    
    # First call triggers compilation
    result = compiled_fun(*args, **kwargs)
    result.block_until_ready()  # Ensure computation completes
    end_compile = time.time()
    
    compile_time = end_compile - start_compile
    
    # Time subsequent execution
    start_exec = time.time()
    result = compiled_fun(*args, **kwargs)
    result.block_until_ready()
    end_exec = time.time()
    
    exec_time = end_exec - start_exec
    
    # Compare with non-JIT version
    start_no_jit = time.time()
    result_no_jit = fun(*args, **kwargs)
    if hasattr(result_no_jit, 'block_until_ready'):
        result_no_jit.block_until_ready()
    end_no_jit = time.time()
    
    no_jit_time = end_no_jit - start_no_jit
    
    return {
        'compile_time': compile_time,
        'jit_exec_time': exec_time,
        'no_jit_time': no_jit_time,
        'speedup': no_jit_time / exec_time if exec_time > 0 else float('inf'),
        'compile_overhead': compile_time / exec_time if exec_time > 0 else float('inf')
    }


def adaptive_jit(min_speedup: float = 2.0, 
                max_compile_overhead: float = 10.0) -> Callable:
    """Adaptive JIT that only compiles if speedup exceeds threshold.
    
    Args:
        min_speedup: Minimum speedup required to justify JIT
        max_compile_overhead: Maximum acceptable compilation overhead
        
    Returns:
        Adaptive JIT decorator
    """
    def decorator(fun: Callable) -> Callable:
        compiled_fun = None
        should_jit = None
        
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            nonlocal compiled_fun, should_jit
            
            if should_jit is None:
                # Profile on first call
                profile = profile_jit_compilation(fun, *args, **kwargs)
                should_jit = (profile['speedup'] >= min_speedup and 
                            profile['compile_overhead'] <= max_compile_overhead)
                
                if should_jit:
                    compiled_fun = jit(fun)
                    print(f"JIT enabled for {fun.__name__}: "
                          f"{profile['speedup']:.1f}x speedup")
                else:
                    print(f"JIT disabled for {fun.__name__}: "
                          f"speedup={profile['speedup']:.1f}, "
                          f"overhead={profile['compile_overhead']:.1f}")
            
            if should_jit and compiled_fun is not None:
                return compiled_fun(*args, **kwargs)
            else:
                return fun(*args, **kwargs)
        
        return wrapper
    
    return decorator


def warmup_jit(fun: Callable, example_inputs: Tuple[Any, ...], 
              num_warmup: int = 3) -> Callable:
    """Warmup JIT-compiled function with example inputs.
    
    Args:
        fun: Function to JIT and warmup
        example_inputs: Example inputs for warmup
        num_warmup: Number of warmup iterations
        
    Returns:
        Warmed-up JIT function
    """
    compiled_fun = jit(fun)
    
    # Warmup iterations
    for _ in range(num_warmup):
        result = compiled_fun(*example_inputs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
    
    return compiled_fun