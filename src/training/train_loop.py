# File location: jax-nsl/src/training/train_loop.py

"""
Training loop utilities and state management.

This module provides JIT-friendly training loops, state management,
and checkpointing utilities for neural network training.
"""

import jax
import jax.numpy as jnp
from jax import jit
from typing import NamedTuple, Callable, Dict, Any, Tuple, Optional
import pickle
import os
from .optimizers import OptimizerState
from .losses import cross_entropy_loss


class TrainState(NamedTuple):
    """Training state container."""
    step: int
    params: Any
    optimizer_state: Any
    rng: jax.Array
    metrics: Dict[str, float]


def create_train_state(params: Any,
                      optimizer_init: Callable,
                      rng: jax.Array,
                      initial_metrics: Optional[Dict[str, float]] = None) -> TrainState:
    """Create initial training state.
    
    Args:
        params: Model parameters
        optimizer_init: Optimizer initialization function
        rng: Random number generator key
        initial_metrics: Initial metrics dictionary
        
    Returns:
        Initial training state
    """
    optimizer_state = optimizer_init(params)
    metrics = initial_metrics if initial_metrics is not None else {}
    
    return TrainState(
        step=0,
        params=params,
        optimizer_state=optimizer_state,
        rng=rng,
        metrics=metrics
    )


@jit
def training_step(state: TrainState,
                 batch: Dict[str, jnp.ndarray],
                 forward_fn: Callable,
                 loss_fn: Callable,
                 optimizer_update: Callable) -> Tuple[TrainState, Dict[str, float]]:
    """Single training step.
    
    Args:
        state: Current training state
        batch: Batch of training data
        forward_fn: Model forward function
        loss_fn: Loss function
        optimizer_update: Optimizer update function
        
    Returns:
        (updated_state, metrics) tuple
    """
    def loss_and_metrics(params):
        predictions = forward_fn(params, batch['inputs'], training=True)
        loss = loss_fn(predictions, batch['labels'])
        
        # Compute additional metrics
        accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch['labels'])
        
        return loss, {'loss': loss, 'accuracy': accuracy}
    
    # Compute gradients
    (loss_value, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(state.params)
    
    # Update parameters
    new_optimizer_state = optimizer_update(state.optimizer_state, grads)
    
    # Update state
    new_state = TrainState(
        step=state.step + 1,
        params=new_optimizer_state.params,
        optimizer_state=new_optimizer_state,
        rng=state.rng,
        metrics=metrics
    )
    
    return new_state, metrics


@jit
def evaluation_step(state: TrainState,
                   batch: Dict[str, jnp.ndarray],
                   forward_fn: Callable,
                   loss_fn: Callable) -> Dict[str, float]:
    """Single evaluation step.
    
    Args:
        state: Current training state
        batch: Batch of validation data
        forward_fn: Model forward function
        loss_fn: Loss function
        
    Returns:
        Evaluation metrics
    """
    predictions = forward_fn(state.params, batch['inputs'], training=False)
    loss = loss_fn(predictions, batch['labels'])
    accuracy = jnp.mean(jnp.argmax(predictions, axis=-1) == batch['labels'])
    
    return {'val_loss': loss, 'val_accuracy': accuracy}


def train_epoch(state: TrainState,
               train_loader: Any,
               forward_fn: Callable,
               loss_fn: Callable,
               optimizer_update: Callable,
               num_batches: Optional[int] = None) -> Tuple[TrainState, Dict[str, float]]:
    """Train for one epoch.
    
    Args:
        state: Training state
        train_loader: Training data loader
        forward_fn: Model forward function
        loss_fn: Loss function
        optimizer_update: Optimizer update function
        num_batches: Optional limit on number of batches
        
    Returns:
        (updated_state, epoch_metrics) tuple
    """
    epoch_metrics = []
    
    for i, batch in enumerate(train_loader):
        if num_batches is not None and i >= num_batches:
            break
            
        state, batch_metrics = training_step(
            state, batch, forward_fn, loss_fn, optimizer_update
        )
        epoch_metrics.append(batch_metrics)
    
    # Average metrics across batches
    avg_metrics = {}
    if epoch_metrics:
        for key in epoch_metrics[0].keys():
            avg_metrics[key] = jnp.mean(jnp.array([m[key] for m in epoch_metrics]))
    
    return state, avg_metrics


def evaluate_model(state: TrainState,
                  val_loader: Any,
                  forward_fn: Callable,
                  loss_fn: Callable,
                  num_batches: Optional[int] = None) -> Dict[str, float]:
    """Evaluate model on validation set.
    
    Args:
        state: Training state
        val_loader: Validation data loader
        forward_fn: Model forward function
        loss_fn: Loss function
        num_batches: Optional limit on number of batches
        
    Returns:
        Validation metrics
    """
    val_metrics = []
    
    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break
            
        batch_metrics = evaluation_step(state, batch, forward_fn, loss_fn)
        val_metrics.append(batch_metrics)
    
    # Average metrics across batches
    avg_metrics = {}
    if val_metrics:
        for key in val_metrics[0].keys():
            avg_metrics[key] = jnp.mean(jnp.array([m[key] for m in val_metrics]))
    
    return avg_metrics


def training_loop(initial_state: TrainState,
                 train_loader: Any,
                 val_loader: Any,
                 forward_fn: Callable,
                 loss_fn: Callable,
                 optimizer_update: Callable,
                 num_epochs: int,
                 eval_every: int = 1,
                 save_every: int = 10,
                 checkpoint_dir: Optional[str] = None) -> TrainState:
    """Complete training loop.
    
    Args:
        initial_state: Initial training state
        train_loader: Training data loader
        val_loader: Validation data loader
        forward_fn: Model forward function
        loss_fn: Loss function
        optimizer_update: Optimizer update function
        num_epochs: Number of training epochs
        eval_every: Evaluate every N epochs
        save_every: Save checkpoint every N epochs
        checkpoint_dir: Directory to save checkpoints
        
    Returns:
        Final training state
    """
    state = initial_state
    
    for epoch in range(num_epochs):
        # Training
        state, train_metrics = train_epoch(
            state, train_loader, forward_fn, loss_fn, optimizer_update
        )
        
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_metrics['loss']:.4f}, "
              f"Train Acc: {train_metrics['accuracy']:.4f}")
        
        # Evaluation
        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate_model(state, val_loader, forward_fn, loss_fn)
            print(f"Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['val_accuracy']:.4f}")
        
        # Checkpointing
        if checkpoint_dir and (epoch + 1) % save_every == 0:
            save_checkpoint(state, checkpoint_dir, epoch + 1)
            print(f"Saved checkpoint at epoch {epoch + 1}")
    
    return state


def save_checkpoint(state: TrainState,
                   checkpoint_dir: str,
                   epoch: int) -> None:
    """Save training checkpoint.
    
    Args:
        state: Training state to save
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch number
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pkl')
    
    # Convert JAX arrays to regular numpy for serialization
    serializable_state = {
        'step': int(state.step),
        'params': jax.tree_util.tree_map(lambda x: x.__array__(), state.params),
        'optimizer_state': jax.tree_util.tree_map(lambda x: x.__array__() if hasattr(x, '__array__') else x, state.optimizer_state),
        'rng': state.rng.__array__(),
        'metrics': state.metrics,
        'epoch': epoch
    }
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(serializable_state, f)


def load_checkpoint(checkpoint_path: str,
                   optimizer_init: Callable) -> TrainState:
    """Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        optimizer_init: Optimizer initialization function
        
    Returns:
        Loaded training state
    """
    with open(checkpoint_path, 'rb') as f:
        checkpoint_data = pickle.load(f)
    
    # Convert back to JAX arrays
    params = jax.tree_util.tree_map(jnp.array, checkpoint_data['params'])
    rng = jnp.array(checkpoint_data['rng'])
    
    # Recreate optimizer state
    optimizer_state = optimizer_init(params)
    optimizer_state = jax.tree_util.tree_map(
        lambda new, saved: jnp.array(saved) if hasattr(saved, '__array__') else saved,
        optimizer_state, checkpoint_data['optimizer_state']
    )
    
    return TrainState(
        step=checkpoint_data['step'],
        params=params,
        optimizer_state=optimizer_state,
        rng=rng,
        metrics=checkpoint_data['metrics']
    )


def create_simple_train_loop(model_fn: Callable,
                           loss_fn: Callable,
                           optimizer: Tuple[Callable, Callable],
                           learning_rate: float = 0.001) -> Callable:
    """Create a simple training loop function.
    
    Args:
        model_fn: Model function
        loss_fn: Loss function
        optimizer: (init, update) optimizer tuple
        learning_rate: Learning rate
        
    Returns:
        Training loop function
    """
    opt_init, opt_update = optimizer
    
    def train_loop(params, train_data, num_epochs=10):
        opt_state = opt_init(params)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in train_data:
                def loss_fn_wrapper(p):
                    predictions = model_fn(p, batch['inputs'])
                    return loss_fn(predictions, batch['labels'])
                
                loss_val, grads = jax.value_and_grad(loss_fn_wrapper)(params)
                opt_state = opt_update(opt_state, grads)
                params = opt_state.params
                epoch_loss += loss_val
            
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_data):.4f}")
        
        return params
    
    return train_loop


def compute_metrics(predictions: jnp.ndarray,
                   labels: jnp.ndarray,
                   task_type: str = 'classification') -> Dict[str, float]:
    """Compute common evaluation metrics.
    
    Args:
        predictions: Model predictions
        labels: True labels
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary of computed metrics
    """
    if task_type == 'classification':
        pred_classes = jnp.argmax(predictions, axis=-1)
        accuracy = jnp.mean(pred_classes == labels)
        
        # Top-5 accuracy if applicable
        if predictions.shape[-1] >= 5:
            top5_pred = jnp.argsort(predictions, axis=-1)[:, -5:]
            top5_acc = jnp.mean(jnp.any(top5_pred == labels[:, None], axis=1))
        else:
            top5_acc = accuracy
        
        return {
            'accuracy': float(accuracy),
            'top5_accuracy': float(top5_acc)
        }
    
    elif task_type == 'regression':
        mse = jnp.mean((predictions - labels) ** 2)
        mae = jnp.mean(jnp.abs(predictions - labels))
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(jnp.sqrt(mse))
        }
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")