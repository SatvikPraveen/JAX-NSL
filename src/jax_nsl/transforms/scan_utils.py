# File location: src/jax_nsl/transforms/scan_utils.py

"""
``lax.scan`` patterns: cumulative ops, RNNs, ODE integration, remat, and
parallel prefix scans.

``scan`` is the workhorse for anything sequential: it compiles the body once
(unlike a Python loop, which unrolls) and is reverse-mode differentiable
(unlike ``while_loop``).  When the recurrence is *associative* (cumulative
sums, linear recurrences) ``lax.associative_scan`` evaluates it in
``O(log n)`` parallel depth instead of ``O(n)``.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import lax

Array = jax.Array


# ---------------------------------------------------------------------------
# Cumulative operations
# ---------------------------------------------------------------------------

def cumulative_op(op: Callable[[Array, Array], Array], xs: Array, init: Optional[Array] = None,
                  axis: int = 0, reverse: bool = False) -> Array:
    """Inclusive cumulative ``op`` along ``axis`` using ``scan``.

    With ``init=None`` the first output equals the first element, so the
    result has the same shape as ``xs`` (like ``jnp.cumsum``).
    """
    xs = jnp.moveaxis(xs, axis, 0)

    def step(carry, x):
        out = op(carry, x)
        return out, out

    if init is None:
        _, rest = lax.scan(step, xs[0], xs[1:], reverse=reverse)
        results = jnp.concatenate([xs[:1], rest], axis=0)
    else:
        _, results = lax.scan(step, init, xs, reverse=reverse)
    return jnp.moveaxis(results, 0, axis)


def cumulative_sum(xs: Array, axis: int = 0) -> Array:
    """``cumsum`` via :func:`cumulative_op` (see :func:`parallel_cumsum` for the log-depth version)."""
    return cumulative_op(jnp.add, xs, axis=axis)


def parallel_cumsum(xs: Array, axis: int = 0) -> Array:
    """``cumsum`` via ``associative_scan`` - log-depth parallel prefix sum."""
    return lax.associative_scan(jnp.add, xs, axis=axis)


def associative_scan(op: Callable, elems: Any, reverse: bool = False, axis: int = 0) -> Any:
    """Thin wrapper over :func:`jax.lax.associative_scan` (``op`` must be associative)."""
    return lax.associative_scan(op, elems, reverse=reverse, axis=axis)


def linear_recurrence(a: Array, b: Array) -> Array:
    """Solve ``x_t = a_t * x_{t-1} + b_t`` (with ``x_{-1} = 0``) in parallel.

    The pair ``(a, b)`` composes associatively::

        (a2, b2) o (a1, b1) = (a2 * a1, a2 * b1 + b2)

    so the whole recurrence is one ``associative_scan`` - the trick behind
    parallel training of linear RNNs / state-space models (S4, S5, Mamba).
    Broadcasting over trailing dimensions is supported.
    """
    def combine(left, right):
        a_l, b_l = left
        a_r, b_r = right
        return a_r * a_l, a_r * b_l + b_r

    _, x = lax.associative_scan(combine, (a, b))
    return x


def running_statistics(xs: Array, axis: int = 0) -> Tuple[Array, Array]:
    """Running mean and (unbiased) variance via Welford's online algorithm.

    Welford's update avoids the catastrophic cancellation of the naive
    ``E[x^2] - E[x]^2`` formula.  The variance at ``t = 0`` is defined as 0.
    """
    xs = jnp.moveaxis(xs, axis, 0)

    def step(state, x):
        count, mean, m2 = state
        count = count + 1
        delta = x - mean
        mean = mean + delta / count
        m2 = m2 + delta * (x - mean)
        var = m2 / jnp.maximum(count - 1, 1)
        return (count, mean, m2), (mean, var)

    init = (jnp.asarray(0.0, xs.dtype), jnp.zeros_like(xs[0]), jnp.zeros_like(xs[0]))
    _, (means, variances) = lax.scan(step, init, xs)
    return jnp.moveaxis(means, 0, axis), jnp.moveaxis(variances, 0, axis)


# ---------------------------------------------------------------------------
# Sequential application
# ---------------------------------------------------------------------------

def sequential_apply(funs: Sequence[Callable], init_state: Any,
                     inputs: Optional[Sequence[Any]] = None) -> Tuple[Any, List[Any]]:
    """Apply *different* functions in sequence: ``state, out = fun_i(state[, x_i])``.

    Heterogeneous Python callables cannot be scanned (they are not arrays),
    so this is a plain loop that unrolls under ``jit``.  For *identical*
    functions with different parameters use :func:`scan_layers`.
    """
    if inputs is None:
        inputs = [None] * len(funs)
    state, outputs = init_state, []
    for fun, x in zip(funs, inputs):
        state, out = fun(state) if x is None else fun(state, x)
        outputs.append(out)
    return state, outputs


def scan_layers(layer_fn: Callable[[Any, Any], Any], stacked_params: Any, x: Any,
                remat: bool = False) -> Any:
    """Apply one layer function with ``L`` stacked parameter sets: ``x = layer(p_i, x)``.

    Stacking per-layer parameters along a leading axis and scanning over them
    compiles the layer *once* instead of ``L`` times - this is how large
    transformer stacks are written in JAX.  With ``remat=True`` each layer's
    activations are recomputed in the backward pass (``jax.checkpoint``),
    trading compute for ``O(1)``-in-depth activation memory.
    """
    body = layer_fn
    if remat:
        body = jax.checkpoint(layer_fn)

    def step(carry, params):
        return body(params, carry), None

    out, _ = lax.scan(step, x, stacked_params)
    return out


def stack_params(param_list: Sequence[Any]) -> Any:
    """Stack a list of identically structured pytrees along a new leading axis."""
    return jax.tree_util.tree_map(lambda *leaves: jnp.stack(leaves), *param_list)


def rnn_scan(rnn_cell: Callable[[Any, Any], Tuple[Any, Any]], init_state: Any, inputs: Array,
             reverse: bool = False, unroll: int = 1) -> Tuple[Any, Array]:
    """Run ``state, out = rnn_cell(state, x_t)`` over ``inputs`` (time-major).

    Args:
        rnn_cell: ``(state, x) -> (new_state, output)``.
        init_state: Initial hidden state.
        inputs: Array of shape ``(seq_len, ...)``.
        reverse: Process the sequence backwards (outputs stay time-aligned).
        unroll: Unroll factor for ``scan`` (trades compile time for speed).
    """
    return lax.scan(rnn_cell, init_state, inputs, reverse=reverse, unroll=unroll)


def bidirectional_rnn_scan(cell_fwd: Callable, cell_bwd: Callable, init_fwd: Any, init_bwd: Any,
                           inputs: Array) -> Tuple[Tuple[Any, Any], Array]:
    """Forward and backward scans with outputs concatenated on the last axis."""
    sf, of = lax.scan(cell_fwd, init_fwd, inputs)
    sb, ob = lax.scan(cell_bwd, init_bwd, inputs, reverse=True)
    return (sf, sb), jnp.concatenate([of, ob], axis=-1)


def dynamic_rnn(cell: Callable, inputs: Array, sequence_lengths: Array, init_state: Any,
                time_major: bool = True) -> Tuple[Any, Array]:
    """RNN over padded batches: state and outputs freeze once ``t >= length``.

    Args:
        cell: ``(state, x_t) -> (new_state, out_t)`` operating on a whole batch.
        inputs: ``(time, batch, ...)`` if ``time_major`` else ``(batch, time, ...)``.
        sequence_lengths: ``(batch,)`` valid lengths.
        init_state: Batched initial state (leading axis = batch).
        time_major: Layout of ``inputs``/outputs.
    """
    if not time_major:
        inputs = jnp.swapaxes(inputs, 0, 1)

    def broadcast_mask(mask, like):
        return mask.reshape(mask.shape + (1,) * (like.ndim - 1))

    def step(state, xs):
        t, x = xs
        active = t < sequence_lengths
        new_state, out = cell(state, x)
        new_state = jax.tree_util.tree_map(
            lambda n, o: jnp.where(broadcast_mask(active, n), n, o), new_state, state)
        out = jnp.where(broadcast_mask(active, out), out, jnp.zeros_like(out))
        return new_state, out

    final_state, outputs = lax.scan(step, init_state, (jnp.arange(inputs.shape[0]), inputs))
    if not time_major:
        outputs = jnp.swapaxes(outputs, 0, 1)
    return final_state, outputs


def windowed_scan(fun: Callable, inputs: Array, window_size: int, stride: int = 1,
                  init: Optional[Any] = None) -> Tuple[Any, Array]:
    """Apply ``fun`` to sliding windows of ``inputs`` (leading axis) with ``scan``."""
    num_windows = (inputs.shape[0] - window_size) // stride + 1
    idx = jnp.arange(window_size)[None, :] + jnp.arange(num_windows)[:, None] * stride
    windows = inputs[idx]
    if init is None:
        _, outputs = lax.scan(lambda _, w: (None, fun(w)), None, windows)
        return None, outputs
    return lax.scan(fun, init, windows)


# ---------------------------------------------------------------------------
# ODE integration
# ---------------------------------------------------------------------------

def _euler(f, y, t, dt):
    return y + dt * f(y, t)


def _midpoint(f, y, t, dt):
    k1 = f(y, t)
    return y + dt * f(y + 0.5 * dt * k1, t + 0.5 * dt)


def _rk4(f, y, t, dt):
    k1 = f(y, t)
    k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = f(y + dt * k3, t + dt)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


_STEPPERS = {"euler": _euler, "midpoint": _midpoint, "rk4": _rk4}


def ode_solve_scan(ode_fn: Callable[[Any, Array], Any], y0: Any, t: Array,
                   method: str = "rk4") -> Any:
    """Fixed-grid explicit integration of ``dy/dt = ode_fn(y, t)`` on the time points ``t``.

    Returns the trajectory including ``y0`` (leading axis = time).  Because
    the whole solve is a ``scan``, gradients w.r.t. ``y0``, ``t`` and any
    parameters closed over by ``ode_fn`` come from ordinary reverse mode
    (discretise-then-optimise).

    Args:
        ode_fn: ``(y, t) -> dy/dt``; ``y`` may be any pytree.
        y0: Initial state.
        t: 1-D array of (possibly non-uniform) time points.
        method: ``'euler'`` (order 1), ``'midpoint'`` (order 2) or ``'rk4'`` (order 4).
    """
    stepper = _STEPPERS[method]

    def step(y, ts):
        t_curr, t_next = ts
        y_next = stepper(ode_fn, y, t_curr, t_next - t_curr)
        return y_next, y_next

    _, trajectory = lax.scan(step, y0, (t[:-1], t[1:]))
    return jax.tree_util.tree_map(lambda a, tr: jnp.concatenate([a[None], tr], axis=0), y0, trajectory)


solve_ode = ode_solve_scan


# ---------------------------------------------------------------------------
# Rematerialisation
# ---------------------------------------------------------------------------

def scan_with_checkpointing(fun: Callable, init: Any, xs: Array, checkpoint_every: int = 1,
                            policy: Optional[Callable] = None) -> Tuple[Any, Array]:
    """``scan`` that only stores one carry per ``checkpoint_every`` steps for the backward pass.

    The sequence is split into chunks; each chunk is scanned inside
    ``jax.checkpoint`` so its intermediate activations are recomputed during
    backprop instead of saved.  Memory for the backward pass drops from
    ``O(n)`` to ``O(n / k + k)``; compute rises by roughly one extra forward.
    A remainder shorter than ``checkpoint_every`` is scanned without remat so
    no padding ever enters the recurrence.

    Args:
        fun: Scan body ``(carry, x) -> (carry, y)``.
        init: Initial carry.
        xs: Sequence (leading axis).
        checkpoint_every: Chunk length.
        policy: Optional ``jax.checkpoint_policies`` entry (e.g.
            ``jax.checkpoint_policies.dots_saveable``) to fine-tune what is saved.
    """
    if checkpoint_every <= 1:
        return lax.scan(fun, init, xs)

    n = xs.shape[0]
    num_full = n // checkpoint_every
    remainder = n - num_full * checkpoint_every

    def chunk_scan(carry, chunk):
        return lax.scan(fun, carry, chunk)

    remat_chunk = jax.checkpoint(chunk_scan, policy=policy) if policy else jax.checkpoint(chunk_scan)

    carry = init
    outputs = []
    if num_full > 0:
        main = xs[: num_full * checkpoint_every].reshape((num_full, checkpoint_every) + xs.shape[1:])
        carry, ys = lax.scan(remat_chunk, carry, main)
        outputs.append(ys.reshape((num_full * checkpoint_every,) + ys.shape[2:]))
    if remainder > 0:
        carry, ys = lax.scan(fun, carry, xs[num_full * checkpoint_every:])
        outputs.append(ys)
    return carry, jnp.concatenate(outputs, axis=0)
