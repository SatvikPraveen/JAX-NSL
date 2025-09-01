# File location: jax-nsl/src/utils/benchmarking.py

"""
Performance benchmarking and profiling utilities.

This module provides tools for measuring execution time, memory usage,
and comparing different implementations.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Dict, Any, List, Optional, Tuple
import time
import functools
import gc
import tracemalloc


def warmup_function(fn: Callable, 
                   *args, 
                   num_warmup: int = 3,
                   **kwargs) -> None:
    """Warmup function by running it multiple times.
    
    Args:
        fn: Function to warmup
        *args: Function arguments
        num_warmup: Number of warmup iterations
        **kwargs: Function keyword arguments
    """
    for _ in range(num_warmup):
        result = fn(*args, **kwargs)
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()


def benchmark_function(fn: Callable,
                      *args,
                      num_runs: int = 10,
                      num_warmup: int = 3,
                      return_all: bool = False,
                      **kwargs) -> Dict[str, float]:
    """Benchmark function execution time.
    
    Args:
        fn: Function to benchmark
        *args: Function arguments
        num_runs: Number of benchmark runs
        num_warmup: Number of warmup runs
        return_all: Whether to return all timing measurements
        **kwargs: Function keyword arguments
        
    Returns:
        Timing statistics dictionary
    """
    # Warmup
    warmup_function(fn, *args, num_warmup=num_warmup, **kwargs)
    
    # Benchmark runs
    times = []
    for _ in range(num_runs):
        start_time = time.perf_counter()
        result = fn(*args, **kwargs)
        
        # Ensure computation is complete
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        elif isinstance(result, (tuple, list)):
            for r in result:
                if hasattr(r, 'block_until_ready'):
                    r.block_until_ready()
        
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    # Compute statistics
    times_array = jnp.array(times)
    stats = {
        'mean_time': float(jnp.mean(times_array)),
        'std_time': float(jnp.std(times_array)),
        'min_time': float(jnp.min(times_array)),
        'max_time': float(jnp.max(times_array)),
        'median_time': float(jnp.median(times_array)),
        'num_runs': num_runs
    }
    
    if return_all:
        stats['all_times'] = times
    
    return stats


def time_jit_compilation(fn: Callable,
                        *args,
                        **kwargs) -> Dict[str, float]:
    """Measure JIT compilation time separately from execution time.
    
    Args:
        fn: Function to JIT compile and measure
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Compilation and execution timing statistics
    """
    # Time compilation (first run)
    start_compile = time.perf_counter()
    jitted_fn = jax.jit(fn)
    result = jitted_fn(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    end_compile = time.perf_counter()
    
    compile_time = end_compile - start_compile
    
    # Time subsequent execution
    start_exec = time.perf_counter()
    result = jitted_fn(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    end_exec = time.perf_counter()
    
    exec_time = end_exec - start_exec
    
    return {
        'compile_time': compile_time,
        'execution_time': exec_time,
        'total_time': compile_time + exec_time,
        'compile_overhead': compile_time / exec_time if exec_time > 0 else float('inf')
    }


def measure_throughput(fn: Callable,
                      batch_sizes: List[int],
                      *args,
                      num_runs: int = 5,
                      **kwargs) -> Dict[int, Dict[str, float]]:
    """Measure throughput at different batch sizes.
    
    Args:
        fn: Function to measure throughput
        batch_sizes: List of batch sizes to test
        *args: Additional function arguments
        num_runs: Number of runs per batch size
        **kwargs: Function keyword arguments
        
    Returns:
        Dictionary mapping batch sizes to throughput metrics
    """
    results = {}
    
    for batch_size in batch_sizes:
        # Create batch data (assuming first arg is batch data)
        if args:
            batch_args = (jnp.ones((batch_size,) + args[0].shape[1:]),) + args[1:]
        else:
            batch_args = args
        
        # Benchmark this batch size
        timing_stats = benchmark_function(fn, *batch_args, num_runs=num_runs, **kwargs)
        
        # Compute throughput metrics
        mean_time = timing_stats['mean_time']
        throughput = batch_size / mean_time if mean_time > 0 else 0
        
        results[batch_size] = {
            'throughput_samples_per_sec': throughput,
            'latency_per_sample_ms': (mean_time * 1000) / batch_size,
            'total_time_sec': mean_time,
            'std_time_sec': timing_stats['std_time']
        }
    
    return results


def profile_memory_usage(fn: Callable,
                        *args,
                        **kwargs) -> Dict[str, Any]:
    """Profile memory usage of function execution.
    
    Args:
        fn: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Memory usage statistics
    """
    # Start memory tracing
    tracemalloc.start()
    
    # Get initial memory snapshot
    gc.collect()
    initial_snapshot = tracemalloc.take_snapshot()
    
    # Execute function
    result = fn(*args, **kwargs)
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    
    # Get final memory snapshot
    gc.collect()
    final_snapshot = tracemalloc.take_snapshot()
    
    # Stop tracing
    tracemalloc.stop()
    
    # Compute memory usage
    top_stats = final_snapshot.compare_to(initial_snapshot, 'lineno')
    
    total_memory_mb = sum(stat.size for stat in top_stats) / (1024 * 1024)
    peak_memory_mb = final_snapshot.get_traced_memory()[1] / (1024 * 1024)
    
    return {
        'total_memory_mb': total_memory_mb,
        'peak_memory_mb': peak_memory_mb,
        'top_allocations': [(stat.traceback.format()[-1], stat.size / (1024 * 1024)) 
                           for stat in top_stats[:5]]
    }


def compare_implementations(implementations: Dict[str, Callable],
                           *args,
                           num_runs: int = 10,
                           **kwargs) -> Dict[str, Dict[str, float]]:
    """Compare multiple implementations of the same function.
    
    Args:
        implementations: Dictionary mapping names to functions
        *args: Function arguments
        num_runs: Number of benchmark runs
        **kwargs: Function keyword arguments
        
    Returns:
        Comparison results for each implementation
    """
    results = {}
    
    for name, fn in implementations.items():
        try:
            timing_stats = benchmark_function(fn, *args, num_runs=num_runs, **kwargs)
            results[name] = timing_stats
        except Exception as e:
            results[name] = {'error': str(e)}
    
    # Add speedup comparisons
    if len([r for r in results.values() if 'mean_time' in r]) > 1:
        baseline_time = min(r['mean_time'] for r in results.values() if 'mean_time' in r)
        
        for name, stats in results.items():
            if 'mean_time' in stats:
                stats['speedup'] = baseline_time / stats['mean_time']
                stats['relative_performance'] = stats['mean_time'] / baseline_time
    
    return results


def benchmark_gradient_computation(fn: Callable,
                                  *args,
                                  num_runs: int = 5,
                                  **kwargs) -> Dict[str, float]:
    """Benchmark gradient computation performance.
    
    Args:
        fn: Function to compute gradients for
        *args: Function arguments
        num_runs: Number of benchmark runs
        **kwargs: Function keyword arguments
        
    Returns:
        Gradient computation timing statistics
    """
    grad_fn = jax.grad(fn)
    return benchmark_function(grad_fn, *args, num_runs=num_runs, **kwargs)


def benchmark_vmap_scaling(fn: Callable,
                          single_input: Any,
                          batch_sizes: List[int],
                          num_runs: int = 3) -> Dict[int, Dict[str, float]]:
    """Benchmark vmap scaling with different batch sizes.
    
    Args:
        fn: Function to vectorize
        single_input: Single input example
        batch_sizes: Batch sizes to test
        num_runs: Number of runs per batch size
        
    Returns:
        Scaling results for each batch size
    """
    vmapped_fn = jax.vmap(fn)
    results = {}
    
    for batch_size in batch_sizes:
        # Create batched input
        if isinstance(single_input, jnp.ndarray):
            batch_input = jnp.broadcast_to(single_input, (batch_size,) + single_input.shape)
        else:
            batch_input = jax.tree_util.tree_map(
                lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), 
                single_input
            )
        
        # Benchmark
        timing_stats = benchmark_function(vmapped_fn, batch_input, num_runs=num_runs)
        
        # Compute scaling metrics
        time_per_sample = timing_stats['mean_time'] / batch_size
        
        results[batch_size] = {
            'total_time': timing_stats['mean_time'],
            'time_per_sample': time_per_sample,
            'samples_per_second': 1.0 / time_per_sample,
            'std_time': timing_stats['std_time']
        }
    
    return results


def create_performance_report(benchmark_results: Dict[str, Dict[str, float]],
                             title: str = "Performance Report") -> str:
    """Create formatted performance report.
    
    Args:
        benchmark_results: Results from benchmark functions
        title: Report title
        
    Returns:
        Formatted report string
    """
    report = [f"\n{title}", "=" * len(title), ""]
    
    for name, results in benchmark_results.items():
        report.append(f"{name}:")
        report.append("-" * (len(name) + 1))
        
        if 'error' in results:
            report.append(f"  ERROR: {results['error']}")
        else:
            if 'mean_time' in results:
                report.append(f"  Mean time: {results['mean_time']:.6f} sec")
                report.append(f"  Std time:  {results['std_time']:.6f} sec")
                report.append(f"  Min time:  {results['min_time']:.6f} sec")
                report.append(f"  Max time:  {results['max_time']:.6f} sec")
            
            if 'speedup' in results:
                report.append(f"  Speedup:   {results['speedup']:.2f}x")
            
            if 'throughput_samples_per_sec' in results:
                report.append(f"  Throughput: {results['throughput_samples_per_sec']:.1f} samples/sec")
        
        report.append("")
    
    return "\n".join(report)


def auto_benchmark(fn: Callable,
                  input_shapes: List[Tuple[int, ...]],
                  dtypes: List[jnp.dtype] = None,
                  compile_modes: List[bool] = None) -> Dict[str, Any]:
    """Automatically benchmark function with different configurations.
    
    Args:
        fn: Function to benchmark
        input_shapes: List of input shapes to test
        dtypes: List of dtypes to test (default: [float32])
        compile_modes: List of compilation modes (default: [False, True])
        
    Returns:
        Comprehensive benchmark results
    """
    if dtypes is None:
        dtypes = [jnp.float32]
    if compile_modes is None:
        compile_modes = [False, True]
    
    results = {}
    
    for shape in input_shapes:
        for dtype in dtypes:
            for compile_mode in compile_modes:
                config_name = f"shape_{shape}_dtype_{dtype.name}_jit_{compile_mode}"
                
                # Create test input
                key = jax.random.PRNGKey(42)
                test_input = jax.random.normal(key, shape, dtype=dtype)
                
                # Choose function version
                test_fn = jax.jit(fn) if compile_mode else fn
                
                try:
                    timing_stats = benchmark_function(test_fn, test_input)
                    results[config_name] = timing_stats
                    results[config_name]['config'] = {
                        'shape': shape,
                        'dtype': dtype.name,
                        'jit': compile_mode
                    }
                except Exception as e:
                    results[config_name] = {'error': str(e)}
    
    return results


class PerformanceProfiler:
    """Context manager for profiling function performance."""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        print(f"{self.name}: {duration:.6f} seconds")
    
    @property
    def duration(self):
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None