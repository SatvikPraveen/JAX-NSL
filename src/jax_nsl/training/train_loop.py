# File location: src/jax_nsl/training/train_loop.py

"""
Training loops: jitted step factories, gradient accumulation, mixed precision,
and checkpointing.

The central idea is that a training *step* is a pure function
``(state, batch) -> (state, metrics)`` closed over the model, loss and
optimiser.  Closing over the callables (rather than passing them as
arguments) is what makes the step ``jit``-able: functions are not valid JAX
types, so a ``@jit``-decorated function that *takes* ``forward_fn`` as an
argument can never be called.
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Callable, Dict, Iterable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .optimizers import clip_grads_by_global_norm

Array = jax.Array
Batch = Dict[str, Array]


class TrainState(NamedTuple):
    """Everything needed to resume training. ``params`` mirrors ``optimizer_state.params``."""
    step: Array
    params: Any
    optimizer_state: Any
    rng: Array
    metrics: Dict[str, Any]


def create_train_state(params: Any, optimizer_init: Callable, rng: Array,
                       initial_metrics: Optional[Dict[str, Any]] = None) -> TrainState:
    """Initialise a :class:`TrainState`."""
    return TrainState(step=jnp.zeros((), jnp.int32), params=params,
                      optimizer_state=optimizer_init(params), rng=rng,
                      metrics=initial_metrics or {})


def _default_metrics(predictions: Array, batch: Batch, loss: Array) -> Dict[str, Array]:
    metrics = {"loss": loss}
    if predictions.ndim >= 2 and batch["labels"].ndim == predictions.ndim - 1:
        metrics["accuracy"] = jnp.mean(jnp.argmax(predictions, axis=-1) == batch["labels"])
    return metrics


# ---------------------------------------------------------------------------
# Step factories
# ---------------------------------------------------------------------------

def make_train_step(forward_fn: Callable, loss_fn: Callable, optimizer_update: Callable,
                    metrics_fn: Callable = _default_metrics, max_grad_norm: Optional[float] = None,
                    use_rng: bool = False, jit: bool = True) -> Callable:
    """Build ``step(state, batch) -> (state, metrics)``.

    Args:
        forward_fn: ``forward_fn(params, inputs, training=True[, key=...])``.
        loss_fn: ``loss_fn(predictions, labels) -> scalar``.
        optimizer_update: ``update(opt_state, grads) -> opt_state`` from an optimiser factory.
        metrics_fn: ``(predictions, batch, loss) -> dict`` of scalars.
        max_grad_norm: Optional global-norm clipping before the update.
        use_rng: Split ``state.rng`` each step and pass ``key=`` to ``forward_fn``
            (for dropout).
        jit: Compile the step.
    """
    def step(state: TrainState, batch: Batch) -> Tuple[TrainState, Dict[str, Array]]:
        rng, sub = jax.random.split(state.rng)

        def loss_and_metrics(params):
            kwargs = {"key": sub} if use_rng else {}
            preds = forward_fn(params, batch["inputs"], training=True, **kwargs)
            loss = loss_fn(preds, batch["labels"])
            return loss, metrics_fn(preds, batch, loss)

        (_, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(state.params)
        if max_grad_norm is not None:
            grads = clip_grads_by_global_norm(grads, max_grad_norm)
        opt_state = optimizer_update(state.optimizer_state, grads)
        new_state = TrainState(step=state.step + 1, params=opt_state.params,
                               optimizer_state=opt_state, rng=rng, metrics=metrics)
        return new_state, metrics

    return jax.jit(step) if jit else step


def make_eval_step(forward_fn: Callable, loss_fn: Callable, metrics_fn: Callable = _default_metrics,
                   jit: bool = True) -> Callable:
    """Build ``eval_step(params, batch) -> metrics`` (prefixed with ``val_``)."""
    def step(params: Any, batch: Batch) -> Dict[str, Array]:
        preds = forward_fn(params, batch["inputs"], training=False)
        loss = loss_fn(preds, batch["labels"])
        return {f"val_{k}": v for k, v in metrics_fn(preds, batch, loss).items()}

    return jax.jit(step) if jit else step


def training_step(state: TrainState, batch: Batch, forward_fn: Callable, loss_fn: Callable,
                  optimizer_update: Callable) -> Tuple[TrainState, Dict[str, Array]]:
    """Un-jitted single step for experimentation; prefer :func:`make_train_step` in loops."""
    return make_train_step(forward_fn, loss_fn, optimizer_update, jit=False)(state, batch)


def evaluation_step(state: TrainState, batch: Batch, forward_fn: Callable, loss_fn: Callable
                    ) -> Dict[str, Array]:
    """Un-jitted evaluation step."""
    return make_eval_step(forward_fn, loss_fn, jit=False)(state.params, batch)


# ---------------------------------------------------------------------------
# Gradient accumulation
# ---------------------------------------------------------------------------

def accumulate_gradients(loss_fn: Callable[[Any, Batch], Array], params: Any, microbatches: Batch
                         ) -> Tuple[Array, Any]:
    """Mean loss and gradient over microbatches stacked on a leading axis.

    ``lax.scan`` processes one microbatch at a time, so peak memory is that
    of a single microbatch while the *result* equals the gradient of the
    full batch (for a mean-reduced loss).  Use this to train with an
    effective batch size larger than fits in memory.

    Args:
        loss_fn: ``loss_fn(params, batch) -> scalar`` for one microbatch.
        params: Parameters.
        microbatches: Pytree whose leaves have shape ``(n_micro, micro_batch, ...)``.
    """
    n = jax.tree_util.tree_leaves(microbatches)[0].shape[0]
    grad_fn = jax.value_and_grad(loss_fn)

    def body(carry, mb):
        loss_acc, grad_acc = carry
        loss, g = grad_fn(params, mb)
        return (loss_acc + loss / n, jax.tree_util.tree_map(lambda a, b: a + b / n, grad_acc, g)), None

    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    (loss, grads), _ = lax.scan(body, (jnp.zeros(()), zeros), microbatches)
    return loss, grads


def split_into_microbatches(batch: Batch, num_microbatches: int) -> Batch:
    """Reshape ``(B, ...)`` leaves into ``(n, B // n, ...)``."""
    def split(x):
        b = x.shape[0]
        if b % num_microbatches:
            raise ValueError(f"batch size {b} not divisible by {num_microbatches}")
        return x.reshape((num_microbatches, b // num_microbatches) + x.shape[1:])

    return jax.tree_util.tree_map(split, batch)


def make_accumulating_train_step(forward_fn: Callable, loss_fn: Callable, optimizer_update: Callable,
                                 num_microbatches: int, max_grad_norm: Optional[float] = None) -> Callable:
    """Like :func:`make_train_step` but accumulates over ``num_microbatches`` slices of each batch."""
    def micro_loss(params, mb):
        return loss_fn(forward_fn(params, mb["inputs"], training=True), mb["labels"])

    @jax.jit
    def step(state: TrainState, batch: Batch):
        loss, grads = accumulate_gradients(micro_loss, state.params,
                                           split_into_microbatches(batch, num_microbatches))
        if max_grad_norm is not None:
            grads = clip_grads_by_global_norm(grads, max_grad_norm)
        opt_state = optimizer_update(state.optimizer_state, grads)
        metrics = {"loss": loss}
        return TrainState(state.step + 1, opt_state.params, opt_state, state.rng, metrics), metrics

    return step


# ---------------------------------------------------------------------------
# Mixed precision
# ---------------------------------------------------------------------------

def cast_floating(tree: Any, dtype: Any) -> Any:
    """Cast every floating-point leaf of a pytree to ``dtype`` (ints/bools untouched)."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, tree)


def with_mixed_precision(forward_fn: Callable, compute_dtype: Any = jnp.bfloat16,
                         output_dtype: Any = jnp.float32) -> Callable:
    """Run ``forward_fn`` with params and inputs cast to ``compute_dtype``.

    Parameters stay in float32 in the optimiser (the "master copy"); only the
    forward/backward arithmetic happens in the low-precision type.  Outputs
    are upcast so the loss and its softmax are computed in float32.  bf16 has
    float32's exponent range, so - unlike fp16 - no loss scaling is needed.
    """
    def wrapped(params, inputs, *args, **kwargs):
        out = forward_fn(cast_floating(params, compute_dtype), cast_floating(inputs, compute_dtype),
                         *args, **kwargs)
        return cast_floating(out, output_dtype)

    return wrapped


def scaled_loss_and_grad(loss_fn: Callable, loss_scale: float = 1024.0) -> Callable:
    """``value_and_grad`` with static loss scaling for fp16 (gradients are unscaled on return).

    Multiplying the loss by ``S`` shifts small gradients above fp16's
    underflow threshold (~6e-8); dividing the resulting gradient by ``S``
    recovers the true value.  bf16 does not need this.
    """
    vg = jax.value_and_grad(lambda *a, **k: loss_fn(*a, **k) * loss_scale)

    def wrapped(*args, **kwargs):
        loss, grads = vg(*args, **kwargs)
        return loss / loss_scale, jax.tree_util.tree_map(lambda g: g / loss_scale, grads)

    return wrapped


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------

def _average(metric_list):
    if not metric_list:
        return {}
    return {k: float(np.mean([float(m[k]) for m in metric_list])) for k in metric_list[0]}


def train_epoch(state: TrainState, train_loader: Iterable[Batch], train_step: Callable,
                num_batches: Optional[int] = None) -> Tuple[TrainState, Dict[str, float]]:
    """Run ``train_step`` over a loader; returns the state and epoch-averaged metrics."""
    collected = []
    for i, batch in enumerate(train_loader):
        if num_batches is not None and i >= num_batches:
            break
        state, metrics = train_step(state, batch)
        collected.append(metrics)
    return state, _average(collected)


def evaluate_model(params: Any, val_loader: Iterable[Batch], eval_step: Callable,
                   num_batches: Optional[int] = None) -> Dict[str, float]:
    """Average ``eval_step`` metrics over a loader."""
    collected = []
    for i, batch in enumerate(val_loader):
        if num_batches is not None and i >= num_batches:
            break
        collected.append(eval_step(params, batch))
    return _average(collected)


def training_loop(initial_state: TrainState, train_loader: Callable[[], Iterable[Batch]],
                  val_loader: Optional[Callable[[], Iterable[Batch]]], train_step: Callable,
                  eval_step: Optional[Callable], num_epochs: int, eval_every: int = 1,
                  save_every: int = 0, checkpoint_dir: Optional[str] = None,
                  log_fn: Optional[Callable[[int, Dict[str, float]], None]] = print) -> TrainState:
    """Epoch loop with optional evaluation, checkpointing and logging.

    ``train_loader``/``val_loader`` are *callables returning iterables* so a
    fresh (reshuffled) iterator is created each epoch.
    """
    state = initial_state
    for epoch in range(1, num_epochs + 1):
        state, train_metrics = train_epoch(state, train_loader(), train_step)
        log = {f"train_{k}": v for k, v in train_metrics.items()}
        if val_loader is not None and eval_step is not None and epoch % eval_every == 0:
            log.update(evaluate_model(state.params, val_loader(), eval_step))
        if log_fn is not None:
            log_fn(epoch, log)
        if checkpoint_dir and save_every and epoch % save_every == 0:
            save_checkpoint(state, checkpoint_dir, epoch)
    return state


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _to_host(tree: Any) -> Any:
    def convert(x):
        if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
            return {"__prng_key__": np.asarray(jax.random.key_data(x)),
                    "impl": str(jax.random.key_impl(x))}
        return np.asarray(x) if isinstance(x, (jax.Array, np.ndarray)) else x

    return jax.tree_util.tree_map(convert, tree)


def _to_device(tree: Any) -> Any:
    def convert(x):
        if isinstance(x, dict) and "__prng_key__" in x:
            return jax.random.wrap_key_data(jnp.asarray(x["__prng_key__"]), impl=x["impl"])
        return jnp.asarray(x) if isinstance(x, np.ndarray) else x

    is_key = lambda x: isinstance(x, dict) and "__prng_key__" in x  # noqa: E731
    return jax.tree_util.tree_map(convert, tree, is_leaf=is_key)


def save_checkpoint(state: TrainState, checkpoint_dir: str, epoch: int) -> str:
    """Pickle the state with every array moved to host memory; returns the path."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pkl")
    with open(path, "wb") as f:
        pickle.dump({"state": _to_host(state), "epoch": epoch}, f)
    return path


def load_checkpoint(checkpoint_path: str) -> TrainState:
    """Load a checkpoint written by :func:`save_checkpoint`."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    return TrainState(*_to_device(tuple(data["state"])))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(predictions: Array, labels: Array, task_type: str = "classification"
                    ) -> Dict[str, float]:
    """Accuracy/top-5 for classification, MSE/MAE/RMSE for regression."""
    if task_type == "classification":
        pred = jnp.argmax(predictions, axis=-1)
        accuracy = jnp.mean(pred == labels)
        k = min(5, predictions.shape[-1])
        topk = jnp.argsort(predictions, axis=-1)[:, -k:]
        topk_acc = jnp.mean(jnp.any(topk == labels[:, None], axis=1))
        return {"accuracy": float(accuracy), "top5_accuracy": float(topk_acc)}
    if task_type == "regression":
        mse = jnp.mean((predictions - labels) ** 2)
        return {"mse": float(mse), "mae": float(jnp.mean(jnp.abs(predictions - labels))),
                "rmse": float(jnp.sqrt(mse))}
    raise ValueError(f"Unknown task type: {task_type}")
