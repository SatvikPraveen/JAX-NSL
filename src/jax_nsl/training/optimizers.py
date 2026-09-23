# File location: src/jax_nsl/training/optimizers.py

"""
Optimisers as pure functions over pytrees, with learning-rate schedules and
parameter EMA.

Every optimiser is a factory returning ``(init, update)``:

    init(params) -> state            # state carries params and moments
    update(state, grads) -> state    # a pure function, safe to jit

The state types are ``NamedTuple``s so they are pytrees (they flow through
``jit``/``scan``/``device_put``) and each field is explicit.  A learning
rate may be a float or a schedule ``step -> lr``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import tree_util

Array = jax.Array
Schedule = Callable[[Array], Array]
LearningRate = float | Schedule


class SGDState(NamedTuple):
    step: Array
    params: Any


class MomentumState(NamedTuple):
    step: Array
    params: Any
    momentum: Any


class AdamState(NamedTuple):
    step: Array
    params: Any
    mu: Any  # first moment
    nu: Any  # second moment


class RMSPropState(NamedTuple):
    step: Array
    params: Any
    velocity: Any


class AdaGradState(NamedTuple):
    step: Array
    params: Any
    sum_of_squares: Any


OptimizerState = SGDState | MomentumState | AdamState | RMSPropState | AdaGradState


def get_learning_rate(learning_rate: LearningRate, step: int | Array) -> Array:
    """Evaluate a float or schedule at ``step``."""
    if callable(learning_rate):
        return jnp.asarray(learning_rate(step), jnp.float32)
    return jnp.asarray(learning_rate, jnp.float32)


def _add_weight_decay(grads: Any, params: Any, weight_decay: float) -> Any:
    if weight_decay == 0.0:
        return grads
    return tree_util.tree_map(lambda g, p: g + weight_decay * p, grads, params)


# ---------------------------------------------------------------------------
# Optimisers
# ---------------------------------------------------------------------------


def sgd_optimizer(
    learning_rate: LearningRate, weight_decay: float = 0.0
) -> tuple[Callable, Callable]:
    """Plain SGD with coupled (L2) weight decay."""

    def init(params):
        return SGDState(step=jnp.zeros((), jnp.int32), params=params)

    def update(state, grads):
        step = state.step + 1
        lr = get_learning_rate(learning_rate, state.step)
        grads = _add_weight_decay(grads, state.params, weight_decay)
        params = tree_util.tree_map(lambda p, g: p - lr * g, state.params, grads)
        return SGDState(step=step, params=params)

    return init, update


def momentum_optimizer(
    learning_rate: LearningRate,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    nesterov: bool = False,
) -> tuple[Callable, Callable]:
    """SGD with (Nesterov) momentum ``v <- mu v + g``, ``p <- p - lr * (g + mu v | v)``."""

    def init(params):
        return MomentumState(
            step=jnp.zeros((), jnp.int32),
            params=params,
            momentum=tree_util.tree_map(jnp.zeros_like, params),
        )

    def update(state, grads):
        lr = get_learning_rate(learning_rate, state.step)
        grads = _add_weight_decay(grads, state.params, weight_decay)
        v = tree_util.tree_map(lambda m, g: momentum * m + g, state.momentum, grads)
        if nesterov:
            direction = tree_util.tree_map(lambda g, m: g + momentum * m, grads, v)
        else:
            direction = v
        params = tree_util.tree_map(lambda p, d: p - lr * d, state.params, direction)
        return MomentumState(step=state.step + 1, params=params, momentum=v)

    return init, update


def _adam_moments(state: AdamState, grads: Any, beta1: float, beta2: float):
    step = state.step + 1
    mu = tree_util.tree_map(lambda m, g: beta1 * m + (1 - beta1) * g, state.mu, grads)
    nu = tree_util.tree_map(lambda v, g: beta2 * v + (1 - beta2) * jnp.square(g), state.nu, grads)
    # Bias correction: E[mu] = (1 - beta1^t) g at the start, so divide it out.
    c1 = 1.0 - beta1 ** step.astype(jnp.float32)
    c2 = 1.0 - beta2 ** step.astype(jnp.float32)
    mu_hat = tree_util.tree_map(lambda m: m / c1, mu)
    nu_hat = tree_util.tree_map(lambda v: v / c2, nu)
    return step, mu, nu, mu_hat, nu_hat


def adam_optimizer(
    learning_rate: LearningRate = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[Callable, Callable]:
    """Adam (Kingma & Ba) with optional *coupled* L2 weight decay (added to the gradient)."""

    def init(params):
        zeros = tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(step=jnp.zeros((), jnp.int32), params=params, mu=zeros, nu=zeros)

    def update(state, grads):
        lr = get_learning_rate(learning_rate, state.step)
        grads = _add_weight_decay(grads, state.params, weight_decay)
        step, mu, nu, mu_hat, nu_hat = _adam_moments(state, grads, beta1, beta2)
        params = tree_util.tree_map(
            lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps), state.params, mu_hat, nu_hat
        )
        return AdamState(step=step, params=params, mu=mu, nu=nu)

    return init, update


def adamw_optimizer(
    learning_rate: LearningRate = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[Callable, Callable]:
    """AdamW: *decoupled* weight decay ``p <- p - lr * (adam_step + wd * p)``.

    Unlike L2-in-the-gradient, decoupled decay is not rescaled by the
    adaptive denominator, so it acts uniformly on every parameter.
    """

    def init(params):
        zeros = tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(step=jnp.zeros((), jnp.int32), params=params, mu=zeros, nu=zeros)

    def update(state, grads):
        lr = get_learning_rate(learning_rate, state.step)
        step, mu, nu, mu_hat, nu_hat = _adam_moments(state, grads, beta1, beta2)
        params = tree_util.tree_map(
            lambda p, m, v: p - lr * (m / (jnp.sqrt(v) + eps) + weight_decay * p),
            state.params,
            mu_hat,
            nu_hat,
        )
        return AdamState(step=step, params=params, mu=mu, nu=nu)

    return init, update


def rmsprop_optimizer(
    learning_rate: LearningRate = 1e-2,
    decay: float = 0.9,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> tuple[Callable, Callable]:
    """RMSProp: divide by a running RMS of the gradient."""

    def init(params):
        return RMSPropState(
            step=jnp.zeros((), jnp.int32),
            params=params,
            velocity=tree_util.tree_map(jnp.zeros_like, params),
        )

    def update(state, grads):
        lr = get_learning_rate(learning_rate, state.step)
        grads = _add_weight_decay(grads, state.params, weight_decay)
        v = tree_util.tree_map(
            lambda v, g: decay * v + (1 - decay) * jnp.square(g), state.velocity, grads
        )
        params = tree_util.tree_map(
            lambda p, g, v: p - lr * g / (jnp.sqrt(v) + eps), state.params, grads, v
        )
        return RMSPropState(step=state.step + 1, params=params, velocity=v)

    return init, update


def adagrad_optimizer(
    learning_rate: LearningRate = 1e-2, eps: float = 1e-8, weight_decay: float = 0.0
) -> tuple[Callable, Callable]:
    """AdaGrad: per-parameter step ``lr / sqrt(sum g^2)`` (decays monotonically)."""

    def init(params):
        return AdaGradState(
            step=jnp.zeros((), jnp.int32),
            params=params,
            sum_of_squares=tree_util.tree_map(jnp.zeros_like, params),
        )

    def update(state, grads):
        lr = get_learning_rate(learning_rate, state.step)
        grads = _add_weight_decay(grads, state.params, weight_decay)
        s = tree_util.tree_map(lambda s, g: s + jnp.square(g), state.sum_of_squares, grads)
        params = tree_util.tree_map(
            lambda p, g, s: p - lr * g / (jnp.sqrt(s) + eps), state.params, grads, s
        )
        return AdaGradState(step=state.step + 1, params=params, sum_of_squares=s)

    return init, update


def lion_optimizer(
    learning_rate: LearningRate = 1e-4,
    beta1: float = 0.9,
    beta2: float = 0.99,
    weight_decay: float = 0.0,
) -> tuple[Callable, Callable]:
    """Lion (Chen et al. 2023): sign of an interpolated momentum, no second moment.

    Uses about half the optimiser memory of Adam; typical learning rates are
    3-10x smaller because every update has unit magnitude per element.
    """

    def init(params):
        return MomentumState(
            step=jnp.zeros((), jnp.int32),
            params=params,
            momentum=tree_util.tree_map(jnp.zeros_like, params),
        )

    def update(state, grads):
        lr = get_learning_rate(learning_rate, state.step)
        direction = tree_util.tree_map(
            lambda m, g: jnp.sign(beta1 * m + (1 - beta1) * g), state.momentum, grads
        )
        params = tree_util.tree_map(
            lambda p, d: p - lr * (d + weight_decay * p), state.params, direction
        )
        m = tree_util.tree_map(lambda m, g: beta2 * m + (1 - beta2) * g, state.momentum, grads)
        return MomentumState(step=state.step + 1, params=params, momentum=m)

    return init, update


def apply_optimizer(optimizer_state: Any, grads: Any, update_fn: Callable) -> Any:
    """``update_fn(optimizer_state, grads)`` (kept for API symmetry)."""
    return update_fn(optimizer_state, grads)


# ---------------------------------------------------------------------------
# Gradient processing and EMA
# ---------------------------------------------------------------------------


def clip_grads_by_global_norm(grads: Any, max_norm: float) -> Any:
    """Rescale the whole gradient pytree so its global L2 norm is at most ``max_norm``."""
    norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in tree_util.tree_leaves(grads)))
    factor = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return tree_util.tree_map(lambda g: g * factor, grads)


def ema_update(ema_params: Any, params: Any, decay: float = 0.999) -> Any:
    """``ema <- decay * ema + (1 - decay) * params`` leaf-wise."""
    return tree_util.tree_map(lambda e, p: decay * e + (1.0 - decay) * p, ema_params, params)


def ema_update_debiased(ema_params: Any, params: Any, step: Array, decay: float = 0.999) -> Any:
    """EMA with the warm-up-corrected decay ``min(decay, (1 + t) / (10 + t))``.

    Early in training the plain EMA is dominated by the (random) initial
    parameters; ramping the decay up avoids that bias.
    """
    d = jnp.minimum(decay, (1.0 + step) / (10.0 + step))
    return tree_util.tree_map(lambda e, p: d * e + (1.0 - d) * p, ema_params, params)


# ---------------------------------------------------------------------------
# Learning-rate schedules
# ---------------------------------------------------------------------------


def create_learning_rate_schedule(schedule_type: str, base_lr: float, **kwargs) -> Schedule:
    """Build a schedule ``step -> lr`` (all functions of a traced step, so jit-safe).

    Types and their keyword arguments:

    * ``'constant'``
    * ``'linear'``: ``total_steps``, ``final_lr``
    * ``'cosine'``: ``total_steps``, ``final_lr``
    * ``'exponential'``: ``decay_rate``, ``decay_steps``
    * ``'step'``: ``step_size``, ``gamma``
    * ``'warmup_cosine'``: ``warmup_steps``, ``total_steps``, ``final_lr`` -
      linear warm-up then cosine decay, the default for transformer training.
    """
    total = kwargs.get("total_steps", 1000)
    final_lr = kwargs.get("final_lr", 0.0)

    if schedule_type == "constant":
        return lambda step: jnp.asarray(base_lr, jnp.float32)
    if schedule_type == "linear":
        return lambda step: base_lr + (final_lr - base_lr) * jnp.minimum(step / total, 1.0)
    if schedule_type == "cosine":
        return lambda step: final_lr + 0.5 * (base_lr - final_lr) * (
            1.0 + jnp.cos(math.pi * jnp.minimum(step / total, 1.0))
        )
    if schedule_type == "exponential":
        rate, steps = kwargs.get("decay_rate", 0.96), kwargs.get("decay_steps", 100)
        return lambda step: base_lr * rate ** (step / steps)
    if schedule_type == "step":
        size, gamma = kwargs.get("step_size", 100), kwargs.get("gamma", 0.1)
        return lambda step: base_lr * gamma ** jnp.floor(step / size)
    if schedule_type == "warmup_cosine":
        warmup = kwargs.get("warmup_steps", 100)

        def schedule(step):
            step = jnp.asarray(step, jnp.float32)
            warm = base_lr * step / jnp.maximum(warmup, 1)
            progress = jnp.clip((step - warmup) / jnp.maximum(total - warmup, 1), 0.0, 1.0)
            cos = final_lr + 0.5 * (base_lr - final_lr) * (1.0 + jnp.cos(math.pi * progress))
            return jnp.where(step < warmup, warm, cos)

        return schedule
    raise ValueError(f"Unknown schedule type: {schedule_type}")
