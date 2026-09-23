# File location: src/jax_nsl/utils/benchmarking.py

"""
Benchmarking JAX code correctly.

Three things make naive timing of JAX wrong:

1. **Asynchronous dispatch** - a call returns before the device has finished;
   you must ``block_until_ready()`` on the *outputs* before stopping the clock.
2. **Compilation** - the first call of a jitted function includes tracing and
   XLA compilation; warm up first (or time it separately).
3. **Host memory is not device memory** - ``tracemalloc`` measures Python
   allocations, which say nothing about accelerator usage.  Ask the device
   (``memory_stats``) or the compiled executable (``memory_analysis``).
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def _block(x: Any) -> Any:
    """Block on every array in a pytree; returns the tree."""
    return jax.block_until_ready(x)


def warmup_function(fn: Callable, *args, num_warmup: int = 3, **kwargs) -> None:
    """Call ``fn`` ``num_warmup`` times (compiles and populates caches)."""
    for _ in range(num_warmup):
        _block(fn(*args, **kwargs))


def benchmark_function(fn: Callable, *args, num_runs: int = 10, num_warmup: int = 3,
                       return_all: bool = False, **kwargs) -> Dict[str, Any]:
    """Wall-clock statistics (seconds) for ``fn(*args, **kwargs)`` after warm-up."""
    warmup_function(fn, *args, num_warmup=num_warmup, **kwargs)
    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _block(fn(*args, **kwargs))
        times.append(time.perf_counter() - t0)
    stats: Dict[str, Any] = {
        "mean_time": statistics.fmean(times),
        "std_time": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "min_time": min(times),
        "max_time": max(times),
        "median_time": statistics.median(times),
        "num_runs": num_runs,
    }
    if return_all:
        stats["all_times"] = times
    return stats


def time_jit_compilation(fn: Callable, *args, **kwargs) -> Dict[str, float]:
    """Compile time (first call minus steady state) vs. execution time of ``jit(fn)``."""
    jitted = jax.jit(fn)
    t0 = time.perf_counter()
    _block(jitted(*args, **kwargs))
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    _block(jitted(*args, **kwargs))
    exec_time = time.perf_counter() - t0
    compile_time = max(first - exec_time, 0.0)
    return {"compile_time": compile_time, "execution_time": exec_time, "total_time": first,
            "compile_overhead": compile_time / exec_time if exec_time > 0 else float("inf")}


def measure_throughput(fn: Callable, batch_sizes: Sequence[int], example_input: Array,
                       *args, num_runs: int = 5, **kwargs) -> Dict[int, Dict[str, float]]:
    """Samples/second at several batch sizes (``example_input`` provides the per-sample shape)."""
    results = {}
    for b in batch_sizes:
        x = jnp.ones((b,) + example_input.shape[1:], example_input.dtype)
        stats = benchmark_function(fn, x, *args, num_runs=num_runs, **kwargs)
        mean = stats["mean_time"]
        results[b] = {"throughput_samples_per_sec": b / mean if mean > 0 else float("inf"),
                      "latency_per_sample_ms": 1000.0 * mean / b, "total_time_sec": mean,
                      "std_time_sec": stats["std_time"]}
    return results


# ---------------------------------------------------------------------------
# Memory and FLOPs
# ---------------------------------------------------------------------------

def device_memory_stats(device: Optional[jax.Device] = None) -> Dict[str, float]:
    """Live/peak bytes on a device (MB) when the backend reports them (GPU/TPU; CPU gives {})."""
    device = device or jax.devices()[0]
    stats = device.memory_stats() or {}
    mb = 1024 * 1024
    return {k: v / mb for k, v in stats.items() if k in ("bytes_in_use", "peak_bytes_in_use",
                                                         "bytes_limit", "bytes_reserved")}


def live_array_bytes() -> int:
    """Bytes held by all live ``jax.Array`` objects (a backend-independent proxy)."""
    return int(sum(a.nbytes for a in jax.live_arrays()))


def profile_memory_usage(fn: Callable, *args, **kwargs) -> Dict[str, Any]:
    """Static memory analysis of ``jit(fn)`` plus live-array deltas around one call.

    ``temp_bytes`` is XLA's estimate of scratch memory for the executable,
    ``argument_bytes``/``output_bytes`` the I/O footprint; these come from
    the compiled program, not from measurement, so they are exact for the
    given shapes.
    """
    compiled = jax.jit(fn).lower(*args, **kwargs).compile()
    out: Dict[str, Any] = {}
    try:
        mem = compiled.memory_analysis()
        out.update(temp_bytes=getattr(mem, "temp_size_in_bytes", None),
                   argument_bytes=getattr(mem, "argument_size_in_bytes", None),
                   output_bytes=getattr(mem, "output_size_in_bytes", None))
    except Exception:
        pass
    before = live_array_bytes()
    result = _block(compiled(*args, **kwargs))
    out["live_array_delta_bytes"] = live_array_bytes() - before
    out["result_bytes"] = int(sum(a.nbytes for a in jax.tree_util.tree_leaves(result)))
    out.update({f"device_{k}": v for k, v in device_memory_stats().items()})
    return out


def count_flops(fn: Callable, *args, **kwargs) -> Optional[float]:
    """XLA's FLOP estimate for ``jit(fn)`` at these shapes (``None`` if unavailable)."""
    compiled = jax.jit(fn).lower(*args, **kwargs).compile()
    cost = compiled.cost_analysis() or {}
    if isinstance(cost, list):
        cost = cost[0] if cost else {}
    return cost.get("flops")


# ---------------------------------------------------------------------------
# Comparisons and reports
# ---------------------------------------------------------------------------

def compare_implementations(implementations: Dict[str, Callable], *args, num_runs: int = 10,
                            **kwargs) -> Dict[str, Dict[str, Any]]:
    """Benchmark several functions on the same inputs; adds ``speedup`` relative to the fastest."""
    results: Dict[str, Dict[str, Any]] = {}
    for name, fn in implementations.items():
        try:
            results[name] = benchmark_function(fn, *args, num_runs=num_runs, **kwargs)
        except Exception as e:  # report, don't abort the whole comparison
            results[name] = {"error": str(e)}
    timed = [r["mean_time"] for r in results.values() if "mean_time" in r]
    if len(timed) > 1:
        best = min(timed)
        for r in results.values():
            if "mean_time" in r:
                r["speedup"] = best / r["mean_time"]
                r["relative_performance"] = r["mean_time"] / best
    return results


def benchmark_gradient_computation(fn: Callable, *args, num_runs: int = 5, **kwargs) -> Dict[str, Any]:
    """Time ``jit(grad(fn))``."""
    return benchmark_function(jax.jit(jax.grad(fn)), *args, num_runs=num_runs, **kwargs)


def benchmark_vmap_scaling(fn: Callable, single_input: Any, batch_sizes: Sequence[int],
                           num_runs: int = 3) -> Dict[int, Dict[str, float]]:
    """Time ``jit(vmap(fn))`` at several batch sizes to see how close to linear it scales."""
    vf = jax.jit(jax.vmap(fn))
    results = {}
    for b in batch_sizes:
        batch = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (b,) + jnp.shape(x)), single_input)
        stats = benchmark_function(vf, batch, num_runs=num_runs)
        per = stats["mean_time"] / b
        results[b] = {"total_time": stats["mean_time"], "time_per_sample": per,
                      "samples_per_second": 1.0 / per if per > 0 else float("inf"),
                      "std_time": stats["std_time"]}
    return results


def auto_benchmark(fn: Callable, input_shapes: Sequence[Tuple[int, ...]],
                   dtypes: Optional[Sequence[Any]] = None,
                   compile_modes: Sequence[bool] = (False, True)) -> Dict[str, Dict[str, Any]]:
    """Grid over shapes x dtypes x {eager, jit}."""
    dtypes = list(dtypes) if dtypes else [jnp.float32]
    results = {}
    for shape in input_shapes:
        for dtype in dtypes:
            x = jax.random.normal(jax.random.PRNGKey(42), shape, dtype)
            for use_jit in compile_modes:
                name = f"shape_{shape}_dtype_{jnp.dtype(dtype).name}_jit_{use_jit}"
                test_fn = jax.jit(fn) if use_jit else fn
                try:
                    results[name] = benchmark_function(test_fn, x)
                    results[name]["config"] = {"shape": shape, "dtype": jnp.dtype(dtype).name, "jit": use_jit}
                except Exception as e:
                    results[name] = {"error": str(e)}
    return results


def create_performance_report(benchmark_results: Dict[str, Dict[str, Any]],
                              title: str = "Performance Report") -> str:
    """Plain-text table of benchmark results."""
    lines = [title, "=" * len(title), ""]
    for name, r in benchmark_results.items():
        lines.append(f"{name}:")
        if "error" in r:
            lines.append(f"  ERROR: {r['error']}")
        else:
            if "mean_time" in r:
                lines.append(f"  mean {r['mean_time'] * 1e3:9.3f} ms   std {r['std_time'] * 1e3:8.3f} ms   "
                             f"min {r['min_time'] * 1e3:8.3f} ms   max {r['max_time'] * 1e3:8.3f} ms")
            if "speedup" in r:
                lines.append(f"  speedup vs best: {r['speedup']:.2f}x")
            if "throughput_samples_per_sec" in r:
                lines.append(f"  throughput: {r['throughput_samples_per_sec']:.1f} samples/s")
        lines.append("")
    return "\n".join(lines)


class PerformanceProfiler:
    """``with PerformanceProfiler('step') as p: ...`` then read ``p.duration`` (seconds).

    Blocks on the arrays passed to :meth:`track` (or nothing) before stopping
    the clock, so async dispatch does not hide the real cost.
    """

    def __init__(self, name: str = "operation", verbose: bool = False):
        self.name = name
        self.verbose = verbose
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._tracked: List[Any] = []

    def track(self, value: Any) -> Any:
        """Register outputs to block on at exit; returns them unchanged."""
        self._tracked.append(value)
        return value

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        for v in self._tracked:
            _block(v)
        self.end_time = time.perf_counter()
        if self.verbose:
            print(f"{self.name}: {self.duration:.6f} s")

    @property
    def duration(self) -> Optional[float]:
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time
