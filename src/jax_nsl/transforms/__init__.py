# File location: src/jax_nsl/transforms/__init__.py

"""
JAX transformations: jit, vmap, scan and structured control flow.
"""

from .control_flow import (
    binary_search,
    bounded_while_loop,
    clip_gradient,
    clip_gradient_norm,
    conditional_update,
    dynamic_slice_safe,
    for_loop,
    gather_nd,
    iterative_solver,
    safe_cond,
    safe_divide,
    scatter_add_nd,
    select_n,
    stable_softmax,
    switch_case,
    while_loop_safe,
)
from .jit_utils import (
    aot_compile,
    benchmark_jit,
    compile_info,
    conditional_jit,
    count_compilations,
    donate_argnums_jit,
    efficient_jit,
    jit_with_static,
    profile_jit_compilation,
    smart_jit,
    static_argnums_jit,
    warmup_jit,
)
from .scan_utils import (
    associative_scan,
    bidirectional_rnn_scan,
    cumulative_op,
    cumulative_sum,
    dynamic_rnn,
    linear_recurrence,
    ode_solve_scan,
    parallel_cumsum,
    rnn_scan,
    running_statistics,
    scan_layers,
    scan_with_checkpointing,
    sequential_apply,
    solve_ode,
    stack_params,
    windowed_scan,
)
from .vmap_utils import (
    batch_apply,
    batch_apply_along_axis,
    batch_gradient,
    batch_jacobian,
    batch_matrix_ops,
    batch_outer_product,
    batch_solve,
    batched_gradient,
    batched_matmul,
    chunked_vmap,
    clip_per_example_gradients,
    loop_batch_apply,
    nested_vmap,
    parallel_apply,
    parallel_map,
    per_example_gradients,
    selective_vmap,
    vectorize_function,
    vmap_with_signature,
)

__all__ = [
    # jit_utils.py
    "smart_jit", "efficient_jit", "jit_with_static", "conditional_jit", "donate_argnums_jit",
    "static_argnums_jit", "count_compilations", "aot_compile", "compile_info",
    "profile_jit_compilation", "warmup_jit", "benchmark_jit",
    # vmap_utils.py
    "batch_apply", "vectorize_function", "loop_batch_apply", "parallel_apply",
    "batch_outer_product", "batched_matmul", "batch_solve", "batch_matrix_ops",
    "batch_apply_along_axis", "nested_vmap", "selective_vmap", "vmap_with_signature",
    "batch_gradient", "batch_jacobian", "batched_gradient", "per_example_gradients",
    "clip_per_example_gradients", "chunked_vmap", "parallel_map",
    # scan_utils.py
    "cumulative_op", "cumulative_sum", "parallel_cumsum", "associative_scan",
    "linear_recurrence", "running_statistics", "sequential_apply", "scan_layers",
    "stack_params", "rnn_scan", "bidirectional_rnn_scan", "dynamic_rnn", "windowed_scan",
    "ode_solve_scan", "solve_ode", "scan_with_checkpointing",
    # control_flow.py
    "safe_cond", "switch_case", "while_loop_safe", "bounded_while_loop", "for_loop",
    "dynamic_slice_safe", "conditional_update", "binary_search", "iterative_solver",
    "select_n", "gather_nd", "scatter_add_nd", "clip_gradient", "clip_gradient_norm",
    "safe_divide", "stable_softmax",
]
