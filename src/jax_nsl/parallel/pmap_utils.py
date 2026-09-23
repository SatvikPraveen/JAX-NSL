# File location: src/jax_nsl/parallel/pmap_utils.py

"""
Data parallelism with ``pmap`` (SPMD over a leading device axis).

``pmap`` maps a function over the first axis of its inputs, one slice per
device, and gives the body a named axis for collectives.  The recipe:

1. *Replicate* parameters: add a leading axis of size ``num_devices``.
2. *Shard* the batch: reshape ``(B, ...)`` to ``(num_devices, B / num_devices, ...)``.
3. Inside the pmapped step, ``lax.pmean(grads, 'batch')`` averages gradients
   so every replica applies the same update and stays in sync.

``pmap`` is the older API; :mod:`jax_nsl.parallel.pjit_utils` shows the
``jit`` + sharding approach that generalises to model parallelism.  Both
are worth knowing because a lot of existing code uses ``pmap``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, pmap
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

Array = jax.Array
Batch = dict[str, Array]


def replicate_params(params: Any, num_devices: int | None = None) -> Any:
    """Broadcast every leaf to ``(num_devices, ...)`` with slice ``i`` living on device ``i``.

    This is the layout ``pmap`` expects.  It is expressed with a one-axis
    mesh and ``NamedSharding(P('devices'))``, the drop-in replacement for the
    removed ``jax.device_put_replicated``.
    """
    if num_devices is None:
        num_devices = jax.local_device_count()
    devices = np.array(jax.local_devices()[:num_devices])
    sharding = NamedSharding(Mesh(devices, ("devices",)), P("devices"))
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(jnp.broadcast_to(x, (num_devices,) + jnp.shape(x)), sharding),
        params,
    )


def unreplicate_params(replicated: Any) -> Any:
    """Take replica 0 of every leaf."""
    return jax.tree_util.tree_map(lambda x: x[0], replicated)


def shard_batch(batch: Any, num_devices: int | None = None) -> Any:
    """Reshape leaves ``(B, ...) -> (num_devices, B // num_devices, ...)``."""
    if num_devices is None:
        num_devices = jax.local_device_count()

    def shard(x):
        if x.shape[0] % num_devices:
            raise ValueError(f"Batch size {x.shape[0]} not divisible by {num_devices} devices")
        return x.reshape((num_devices, x.shape[0] // num_devices) + x.shape[1:])

    return jax.tree_util.tree_map(shard, batch)


def sync_gradients(gradients: Any, axis_name: str = "batch") -> Any:
    """Average a gradient pytree across replicas (call inside ``pmap``)."""
    return lax.pmean(gradients, axis_name=axis_name)


def sync_params_across_devices(params: Any) -> Any:
    """Force replicas back into agreement by averaging them (e.g. after float drift)."""
    return pmap(lambda p: lax.pmean(p, "devices"), axis_name="devices")(params)


def device_get(replicated: Any) -> Any:
    """Copy replica 0 of every leaf to host memory."""
    return jax.device_get(unreplicate_params(replicated))


# ---------------------------------------------------------------------------
# Training steps
# ---------------------------------------------------------------------------


def data_parallel_step(
    loss_fn: Callable[[Any, Batch], Array], params: Any, batch: Batch, lr: float = 0.01
) -> tuple[Any, Array]:
    """One synchronous SGD step with gradients averaged across devices.

    ``loss_fn`` and ``lr`` are static (a Python callable / float cannot be a
    pmapped argument); ``params`` must be replicated and ``batch`` sharded.
    Returns ``(new_params, per_device_loss)`` where the new params are again
    replicated (identical on every device).
    """

    @functools.partial(pmap, axis_name="batch", static_broadcasted_argnums=(0, 3))
    def step(loss_fn, params, batch, lr):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch)
        grads = lax.pmean(grads, "batch")
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
        return params, loss

    return step(loss_fn, params, batch, lr)


def create_pmap_train_step(
    forward_fn: Callable, loss_fn: Callable, optimizer_update: Callable, axis_name: str = "batch"
) -> Callable:
    """Pmapped ``step(opt_state, batch) -> (opt_state, metrics)`` with gradient all-reduce."""

    @functools.partial(pmap, axis_name=axis_name)
    def step(opt_state, batch):
        def loss_and_metrics(params):
            preds = forward_fn(params, batch["inputs"], training=True)
            loss = loss_fn(preds, batch["labels"])
            acc = jnp.mean(jnp.argmax(preds, axis=-1) == batch["labels"])
            return loss, {"loss": loss, "accuracy": acc}

        (_, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(opt_state.params)
        grads = lax.pmean(grads, axis_name)
        metrics = lax.pmean(metrics, axis_name)
        return optimizer_update(opt_state, grads), metrics

    return step


def parallel_eval_step(
    forward_fn: Callable, loss_fn: Callable, axis_name: str = "batch"
) -> Callable:
    """Pmapped ``eval(params, batch) -> metrics`` averaged over devices."""

    @functools.partial(pmap, axis_name=axis_name)
    def step(params, batch):
        preds = forward_fn(params, batch["inputs"], training=False)
        loss = loss_fn(preds, batch["labels"])
        acc = jnp.mean(jnp.argmax(preds, axis=-1) == batch["labels"])
        return lax.pmean({"loss": loss, "accuracy": acc}, axis_name)

    return step


def parallel_train_epoch(
    opt_state: Any,
    train_loader: Iterable[Batch],
    train_step: Callable,
    num_devices: int | None = None,
) -> tuple[Any, dict[str, float]]:
    """Replicate ``opt_state``, run a pmapped step over each batch, unreplicate at the end."""
    if num_devices is None:
        num_devices = jax.local_device_count()
    state = replicate_params(opt_state, num_devices)
    collected = []
    for batch in train_loader:
        state, metrics = train_step(state, shard_batch(batch, num_devices))
        collected.append(unreplicate_params(metrics))
    avg = (
        {k: float(jnp.mean(jnp.stack([m[k] for m in collected]))) for k in collected[0]}
        if collected
        else {}
    )
    return unreplicate_params(state), avg


def create_parallel_inference_fn(forward_fn: Callable) -> Callable:
    """Batched inference across devices with automatic padding of ragged batches."""
    infer = pmap(lambda params, x: forward_fn(params, x, training=False), axis_name="batch")

    def inference_fn(params: Any, inputs: Array, num_devices: int | None = None) -> Array:
        if num_devices is None:
            num_devices = jax.local_device_count()
        n = inputs.shape[0]
        pad = (-n) % num_devices
        if pad:
            inputs = jnp.concatenate([inputs, jnp.zeros((pad,) + inputs.shape[1:], inputs.dtype)])
        sharded = inputs.reshape((num_devices, -1) + inputs.shape[1:])
        out = infer(replicate_params(params, num_devices), sharded)
        return out.reshape((-1,) + out.shape[2:])[:n]

    return inference_fn


def estimate_memory_usage(
    params: Any, batch_size: int, num_devices: int | None = None
) -> dict[str, float]:
    """Rough memory budget (MB) for replicated data-parallel training with an Adam-like optimiser."""
    if num_devices is None:
        num_devices = jax.local_device_count()
    param_mb = sum(p.nbytes for p in jax.tree_util.tree_leaves(params)) / (1024 * 1024)
    replicated = param_mb * num_devices
    return {
        "parameters_mb": param_mb,
        "replicated_parameters_mb": replicated,
        "gradients_mb": replicated,
        "optimizer_state_mb": 2 * replicated,
        "total_estimated_mb": 4 * replicated,
        "per_device_batch_size": batch_size // num_devices,
        "num_devices": num_devices,
    }
