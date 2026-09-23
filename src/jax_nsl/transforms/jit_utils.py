# File location: src/jax_nsl/transforms/jit_utils.py

"""
JIT compilation: static/donated arguments, retrace diagnostics, AOT compile.

``jax.jit`` already caches compiled executables keyed on the argument
*signature* (shapes, dtypes, pytree structure, static values), so a
hand-rolled cache buys nothing.  What people actually need is to know *when*
a function is being retraced (:func:`count_compilations`), how expensive the
compiled program is (:func:`compile_info`) and to separate compile time from
run time when benchmarking (:func:`profile_jit_compilation`).
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any

import jax
from jax import jit

ArgNums = int | tuple[int, ...] | None
ArgNames = str | tuple[str, ...] | None


def smart_jit(
    fun: Callable,
    static_argnums: ArgNums = None,
    static_argnames: ArgNames = None,
    donate_argnums: ArgNums = None,
    donate_argnames: ArgNames = None,
    inline: bool = False,
) -> Callable:
    """``jax.jit`` with the commonly used keyword arguments spelled out.

    * ``static_*``: hashable Python values baked into the trace (a new value
      means a recompile).
    * ``donate_*``: let XLA reuse the input buffers for the outputs - essential
      for ``params = step(params, ...)`` loops with large models.
    """
    return jit(
        fun,
        static_argnums=static_argnums,
        static_argnames=static_argnames,
        donate_argnums=donate_argnums,
        donate_argnames=donate_argnames,
        inline=inline,
    )


efficient_jit = smart_jit


def jit_with_static(fun: Callable, static_argnums: int | tuple[int, ...] = ()) -> Callable:
    """``jit(fun, static_argnums=...)``."""
    return jit(fun, static_argnums=static_argnums)


def conditional_jit(condition: bool = True) -> Callable:
    """Decorator that applies ``jit`` only when ``condition`` is true (handy for debugging)."""

    def decorator(fun: Callable) -> Callable:
        return jit(fun) if condition else fun

    return decorator


def donate_argnums_jit(argnums: int | tuple[int, ...]) -> Callable:
    """Decorator form of ``jit(..., donate_argnums=argnums)``."""
    return lambda fun: jit(fun, donate_argnums=argnums)


def static_argnums_jit(argnums: int | tuple[int, ...]) -> Callable:
    """Decorator form of ``jit(..., static_argnums=argnums)``."""
    return lambda fun: jit(fun, static_argnums=argnums)


# ---------------------------------------------------------------------------
# Retrace diagnostics
# ---------------------------------------------------------------------------


def count_compilations(fun: Callable, **jit_kwargs) -> Callable:
    """``jit(fun)`` that counts how many times it has been *traced*.

    Python code inside a jitted function runs once per trace, so a counter
    incremented there counts recompilations.  Read it via
    ``wrapped.compilations``.  Use this to find accidental retraces caused
    by changing shapes, dtypes or static arguments.
    """
    state = {"n": 0}

    def counting(*args, **kwargs):
        state["n"] += 1
        return fun(*args, **kwargs)

    compiled = jit(counting, **jit_kwargs)

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        return compiled(*args, **kwargs)

    wrapped.compilations = property(lambda self: state["n"])  # type: ignore[attr-defined]
    wrapped.compilation_count = lambda: state["n"]  # type: ignore[attr-defined]
    return wrapped


def aot_compile(fun: Callable, *example_args, static_argnums: ArgNums = None, **example_kwargs):
    """Ahead-of-time compile ``fun`` for the given example arguments.

    Returns a ``jax.stages.Compiled`` object: call it like the function, or
    inspect it with :func:`compile_info`.  Useful to pay compilation cost up
    front (e.g. before a timed benchmark) and to check FLOP/memory estimates
    without running anything.
    """
    return jit(fun, static_argnums=static_argnums).lower(*example_args, **example_kwargs).compile()


def compile_info(compiled) -> dict[str, Any]:
    """FLOPs, bytes accessed and memory analysis of a compiled executable."""
    cost = compiled.cost_analysis() or {}
    if isinstance(cost, list):  # older JAX returned a list per device
        cost = cost[0] if cost else {}
    info: dict[str, Any] = {
        "flops": cost.get("flops"),
        "bytes_accessed": cost.get("bytes accessed"),
        "transcendentals": cost.get("transcendentals"),
    }
    try:
        mem = compiled.memory_analysis()
        info.update(
            temp_bytes=getattr(mem, "temp_size_in_bytes", None),
            argument_bytes=getattr(mem, "argument_size_in_bytes", None),
            output_bytes=getattr(mem, "output_size_in_bytes", None),
            generated_code_bytes=getattr(mem, "generated_code_size_in_bytes", None),
        )
    except Exception:  # memory analysis is backend-dependent
        pass
    return info


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _block(x: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def profile_jit_compilation(fun: Callable, *args, **kwargs) -> dict[str, float]:
    """Separate compile time from steady-state run time (and time the un-jitted call)."""
    compiled = jit(fun)

    t0 = time.perf_counter()
    _block(compiled(*args, **kwargs))
    compile_plus_run = time.perf_counter() - t0

    t0 = time.perf_counter()
    _block(compiled(*args, **kwargs))
    run = time.perf_counter() - t0

    t0 = time.perf_counter()
    _block(fun(*args, **kwargs))
    eager = time.perf_counter() - t0

    return {
        "compile_time": max(compile_plus_run - run, 0.0),
        "jit_exec_time": run,
        "no_jit_time": eager,
        "speedup": eager / run if run > 0 else float("inf"),
    }


def warmup_jit(fun: Callable, example_inputs: tuple[Any, ...], num_warmup: int = 1) -> Callable:
    """Compile ``fun`` and run it ``num_warmup`` times so later calls are steady state."""
    compiled = jit(fun)
    for _ in range(num_warmup):
        _block(compiled(*example_inputs))
    return compiled


def benchmark_jit(
    fun: Callable, *args, warmup_runs: int = 3, benchmark_runs: int = 10
) -> tuple[float, float]:
    """``(mean_warmup_time, mean_run_time)`` in seconds for ``jit(fun)(*args)``."""
    compiled = jit(fun)
    t0 = time.perf_counter()
    for _ in range(warmup_runs):
        _block(compiled(*args))
    warmup_time = (time.perf_counter() - t0) / max(warmup_runs, 1)

    times = []
    for _ in range(benchmark_runs):
        t1 = time.perf_counter()
        _block(compiled(*args))
        times.append(time.perf_counter() - t1)
    return float(warmup_time), float(sum(times) / len(times))
