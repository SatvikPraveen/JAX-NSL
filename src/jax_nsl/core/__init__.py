# File location: src/jax_nsl/core/__init__.py

"""
Core JAX utilities: dtype/array helpers, PRNG management, stable numerics.
"""

from .arrays import (
    check_finite,
    create_mesh_grid,
    get_dtype_info,
    pad_to_shape,
    safe_cast,
    sliding_window,
    tree_bytes,
    tree_map_with_path,
    tree_size,
    tree_summary,
)
from .numerics import (
    clip_gradients,
    default_fd_step,
    gumbel_softmax,
    log_softmax_stable,
    logsumexp_stable,
    numerical_gradient,
    safe_divide,
    safe_exp,
    safe_log,
    safe_norm,
    safe_sqrt,
    smooth_max,
    smooth_min,
    softmax_stable,
    stable_logsumexp,
    stable_sigmoid,
    stable_softmax,
    stable_tanh,
)
from .prng import (
    PRNGSequence,
    as_key,
    compute_fans,
    glorot_normal_init,
    glorot_uniform_init,
    he_normal_init,
    he_uniform_init,
    lecun_normal_init,
    lecun_uniform_init,
    make_rng_state,
    orthogonal_init,
    random_like,
    split_key_tree,
)

__all__ = [
    # arrays.py
    "get_dtype_info", "safe_cast", "check_finite", "tree_size", "tree_bytes",
    "tree_summary", "tree_map_with_path", "create_mesh_grid", "sliding_window",
    "pad_to_shape",
    # prng.py
    "PRNGSequence", "as_key", "split_key_tree", "random_like", "make_rng_state",
    "compute_fans", "glorot_uniform_init", "glorot_normal_init", "he_uniform_init",
    "he_normal_init", "lecun_uniform_init", "lecun_normal_init", "orthogonal_init",
    # numerics.py
    "safe_log", "safe_exp", "safe_sqrt", "safe_divide", "stable_sigmoid", "stable_tanh",
    "logsumexp_stable", "stable_logsumexp", "softmax_stable", "stable_softmax",
    "log_softmax_stable", "smooth_max", "smooth_min", "gumbel_softmax",
    "safe_norm", "clip_gradients", "default_fd_step", "numerical_gradient",
]
