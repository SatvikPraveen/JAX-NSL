# File location: jax-nsl/src/training/optimizers.py

"""
Optimizers: SGD, Adam(W), schedules.

This module implements common optimization algorithms with
learning rate schedules and parameter update rules.
"""

import jax
import jax.numpy as jnp
from jax import tree_util
from typing import NamedTuple, Callable, Any, Optional, Dict, Union
import math


class OptimizerState(NamedTuple):
    """Base optimizer state."""
    step: int
    params: Any
    
    
class SGDState(OptimizerState):
    """SGD optimizer state."""
    pass


class MomentumState(OptimizerState):
    """SGD with momentum optimizer state."""
    momentum: Any


class AdamState(OptimizerState):
    """Adam optimizer state."""
    momentum: Any
    velocity: Any


class RMSPropState(OptimizerState):
    """RMSProp optimizer state."""
    velocity: Any


class AdaGradState(OptimizerState):
    """AdaGrad optimizer state."""
    sum_of_squares: Any


def sgd_optimizer(learning_rate: float,
                 weight_decay: float = 0.0) -> Callable:
    """SGD optimizer with optional weight decay.
    
    Args:
        learning_rate: Learning rate
        weight_decay: L2 regularization strength
        
    Returns:
        Optimizer function
    """
    def init(params):
        return SGDState(step=0, params=params)
    
    def update(state, grads):
        step = state.step + 1
        
        if weight_decay > 0:
            # Add weight decay to gradients
            grads = tree_util.tree_map(
                lambda g, p: g + weight_decay * p,
                grads, state.params
            )
        
        # Parameter update
        new_params = tree_util.tree_map(
            lambda p, g: p - learning_rate * g,
            state.params, grads
        )
        
        return SGDState(step=step, params=new_params)
    
    return init, update


def momentum_optimizer(learning_rate: float,
                      momentum: float = 0.9,
                      weight_decay: float = 0.0,
                      nesterov: bool = False) -> Callable:
    """SGD with momentum optimizer.
    
    Args:
        learning_rate: Learning rate
        momentum: Momentum coefficient
        weight_decay: Weight decay strength
        nesterov: Whether to use Nesterov momentum
        
    Returns:
        Optimizer function
    """
    def init(params):
        momentum_state = tree_util.tree_map(jnp.zeros_like, params)
        return MomentumState(step=0, params=params, momentum=momentum_state)
    
    def update(state, grads):
        step = state.step + 1
        
        if weight_decay > 0:
            grads = tree_util.tree_map(
                lambda g, p: g + weight_decay * p,
                grads, state.params
            )
        
        # Update momentum
        new_momentum = tree_util.tree_map(
            lambda m, g: momentum * m + g,
            state.momentum, grads
        )
        
        if nesterov:
            # Nesterov momentum
            effective_grad = tree_util.tree_map(
                lambda g, m: g + momentum * m,
                grads, new_momentum
            )
        else:
            # Standard momentum
            effective_grad = new_momentum
        
        # Parameter update
        new_params = tree_util.tree_map(
            lambda p, g: p - learning_rate * g,
            state.params, effective_grad
        )
        
        return MomentumState(step=step, params=new_params, momentum=new_momentum)
    
    return init, update


def adam_optimizer(learning_rate: float = 0.001,
                  beta1: float = 0.9,
                  beta2: float = 0.999,
                  eps: float = 1e-8,
                  weight_decay: float = 0.0) -> Callable:
    """Adam optimizer.
    
    Args:
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        eps: Small constant for numerical stability
        weight_decay: Weight decay strength
        
    Returns:
        Optimizer function
    """
    def init(params):
        momentum = tree_util.tree_map(jnp.zeros_like, params)
        velocity = tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(step=0, params=params, momentum=momentum, velocity=velocity)
    
    def update(state, grads):
        step = state.step + 1
        
        if weight_decay > 0:
            grads = tree_util.tree_map(
                lambda g, p: g + weight_decay * p,
                grads, state.params
            )
        
        # Update biased first and second moments
        new_momentum = tree_util.tree_map(
            lambda m, g: beta1 * m + (1 - beta1) * g,
            state.momentum, grads
        )
        
        new_velocity = tree_util.tree_map(
            lambda v, g: beta2 * v + (1 - beta2) * g * g,
            state.velocity, grads
        )
        
        # Bias correction
        momentum_corrected = tree_util.tree_map(
            lambda m: m / (1 - beta1 ** step),
            new_momentum
        )
        
        velocity_corrected = tree_util.tree_map(
            lambda v: v / (1 - beta2 ** step),
            new_velocity
        )
        
        # Parameter update
        new_params = tree_util.tree_map(
            lambda p, m, v: p - learning_rate * m / (jnp.sqrt(v) + eps),
            state.params, momentum_corrected, velocity_corrected
        )
        
        return AdamState(
            step=step,
            params=new_params,
            momentum=new_momentum,
            velocity=new_velocity
        )
    
    return init, update


def adamw_optimizer(learning_rate: float = 0.001,
                   beta1: float = 0.9,
                   beta2: float = 0.999,
                   eps: float = 1e-8,
                   weight_decay: float = 0.01) -> Callable:
    """AdamW optimizer with decoupled weight decay.
    
    Args:
        learning_rate: Learning rate
        beta1: Exponential decay rate for first moment
        beta2: Exponential decay rate for second moment
        eps: Small constant for numerical stability
        weight_decay: Weight decay strength
        
    Returns:
        Optimizer function
    """
    def init(params):
        momentum = tree_util.tree_map(jnp.zeros_like, params)
        velocity = tree_util.tree_map(jnp.zeros_like, params)
        return AdamState(step=0, params=params, momentum=momentum, velocity=velocity)
    
    def update(state, grads):
        step = state.step + 1
        
        # Update biased first and second moments (without weight decay)
        new_momentum = tree_util.tree_map(
            lambda m, g: beta1 * m + (1 - beta1) * g,
            state.momentum, grads
        )
        
        new_velocity = tree_util.tree_map(
            lambda v, g: beta2 * v + (1 - beta2) * g * g,
            state.velocity, grads
        )
        
        # Bias correction
        momentum_corrected = tree_util.tree_map(
            lambda m: m / (1 - beta1 ** step),
            new_momentum
        )
        
        velocity_corrected = tree_util.tree_map(
            lambda v: v / (1 - beta2 ** step),
            new_velocity
        )
        
        # Parameter update with decoupled weight decay
        new_params = tree_util.tree_map(
            lambda p, m, v: p - learning_rate * (m / (jnp.sqrt(v) + eps) + weight_decay * p),
            state.params, momentum_corrected, velocity_corrected
        )
        
        return AdamState(
            step=step,
            params=new_params,
            momentum=new_momentum,
            velocity=new_velocity
        )
    
    return init, update


def rmsprop_optimizer(learning_rate: float = 0.01,
                     decay: float = 0.9,
                     eps: float = 1e-8,
                     weight_decay: float = 0.0) -> Callable:
    """RMSProp optimizer.
    
    Args:
        learning_rate: Learning rate
        decay: Decay rate for moving average of squared gradients
        eps: Small constant for numerical stability
        weight_decay: Weight decay strength
        
    Returns:
        Optimizer function
    """
    def init(params):
        velocity = tree_util.tree_map(jnp.zeros_like, params)
        return RMSPropState(step=0, params=params, velocity=velocity)
    
    def update(state, grads):
        step = state.step + 1
        
        if weight_decay > 0:
            grads = tree_util.tree_map(
                lambda g, p: g + weight_decay * p,
                grads, state.params
            )
        
        # Update velocity (moving average of squared gradients)
        new_velocity = tree_util.tree_map(
            lambda v, g: decay * v + (1 - decay) * g * g,
            state.velocity, grads
        )
        
        # Parameter update
        new_params = tree_util.tree_map(
            lambda p, g, v: p - learning_rate * g / (jnp.sqrt(v) + eps),
            state.params, grads, new_velocity
        )
        
        return RMSPropState(step=step, params=new_params, velocity=new_velocity)
    
    return init, update


def adagrad_optimizer(learning_rate: float = 0.01,
                     eps: float = 1e-8,
                     weight_decay: float = 0.0) -> Callable:
    """AdaGrad optimizer.
    
    Args:
        learning_rate: Learning rate
        eps: Small constant for numerical stability
        weight_decay: Weight decay strength
        
    Returns:
        Optimizer function
    """
    def init(params):
        sum_of_squares = tree_util.tree_map(jnp.zeros_like, params)
        return AdaGradState(step=0, params=params, sum_of_squares=sum_of_squares)
    
    def update(state, grads):
        step = state.step + 1
        
        if weight_decay > 0:
            grads = tree_util.tree_map(
                lambda g, p: g + weight_decay * p,
                grads, state.params
            )
        
        # Update sum of squared gradients
        new_sum_of_squares = tree_util.tree_map(
            lambda s, g: s + g * g,
            state.sum_of_squares, grads
        )
        
        # Parameter update
        new_params = tree_util.tree_map(
            lambda p, g, s: p - learning_rate * g / (jnp.sqrt(s) + eps),
            state.params, grads, new_sum_of_squares
        )
        
        return AdaGradState(
            step=step,
            params=new_params,
            sum_of_squares=new_sum_of_squares
        )
    
    return init, update


def create_learning_rate_schedule(schedule_type: str,
                                 base_lr: float,
                                 **kwargs) -> Callable:
    """Create learning rate schedule.
    
    Args:
        schedule_type: Type of schedule ('constant', 'linear', 'cosine', 'exponential', 'step')
        base_lr: Base learning rate
        **kwargs: Schedule-specific parameters
        
    Returns:
        Learning rate schedule function
    """
    if schedule_type == 'constant':
        return lambda step: base_lr
    
    elif schedule_type == 'linear':
        total_steps = kwargs.get('total_steps', 1000)
        final_lr = kwargs.get('final_lr', 0.0)
        
        def linear_schedule(step):
            progress = jnp.minimum(step / total_steps, 1.0)
            return base_lr * (1 - progress) + final_lr * progress
        
        return linear_schedule
    
    elif schedule_type == 'cosine':
        total_steps = kwargs.get('total_steps', 1000)
        final_lr = kwargs.get('final_lr', 0.0)
        
        def cosine_schedule(step):
            progress = jnp.minimum(step / total_steps, 1.0)
            return final_lr + (base_lr - final_lr) * 0.5 * (1 + jnp.cos(math.pi * progress))
        
        return cosine_schedule
    
    elif schedule_type == 'exponential':
        decay_rate = kwargs.get('decay_rate', 0.96)
        decay_steps = kwargs.get('decay_steps', 100)
        
        def exponential_schedule(step):
            return base_lr * (decay_rate ** (step / decay_steps))
        
        return exponential_schedule
    
    elif schedule_type == 'step':
        step_size = kwargs.get('step_size', 100)
        gamma = kwargs.get('gamma', 0.1)
        
        def step_schedule(step):
            return base_lr * (gamma ** (step // step_size))
        
        return step_schedule
    
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def apply_optimizer(optimizer_state: Any,
                   grads: Any,
                   update_fn: Callable) -> Any:
    """Apply optimizer update.
    
    Args:
        optimizer_state: Current optimizer state
        grads: Gradients
        update_fn: Optimizer update function
        
    Returns:
        Updated optimizer state
    """
    return update_fn(optimizer_state, grads)


def clip_grads_by_global_norm(grads: Any, max_norm: float) -> Any:
    """Clip gradients by global norm.
    
    Args:
        grads: Gradient tree
        max_norm: Maximum gradient norm
        
    Returns:
        Clipped gradients
    """
    # Compute global norm
    global_norm = jnp.sqrt(sum(
        jnp.sum(g ** 2) for g in tree_util.tree_leaves(grads)
    ))
    
    # Clip if necessary
    clip_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-8))
    
    return tree_util.tree_map(lambda g: g * clip_factor, grads)


def get_learning_rate(optimizer_state: Any) -> float:
    """Extract current learning rate from optimizer state.
    
    Args:
        optimizer_state: Optimizer state
        
    Returns:
        Current learning rate
    """
    # This is a placeholder - actual implementation depends on optimizer type
    return 0.001  # Default fallback