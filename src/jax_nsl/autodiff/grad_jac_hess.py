# File location: src/jax_nsl/autodiff/grad_jac_hess.py

"""
Gradients, Jacobians, Hessians and the products that avoid forming them.

Choosing a mode
---------------
* ``jax.grad`` / ``jacrev``: reverse mode, cost ~ (#outputs) backward passes.
  Best for scalar losses (one pass).
* ``jacfwd``: forward mode, cost ~ (#inputs) JVPs.  Best for tall Jacobians.
* ``hvp`` (forward-over-reverse): one JVP of a gradient, no Hessian ever
  materialised - the workhorse of second-order and curvature methods.

Checking gradients
------------------
:func:`finite_diff_grad` implements central differences and the *complex-step*
method, which has no subtractive cancellation and is accurate to machine
precision for analytic functions.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import grad, hessian, jacfwd, jacobian, jacrev
from jax.experimental import checkify

Array = jax.Array


# ---------------------------------------------------------------------------
# Checked derivatives
# ---------------------------------------------------------------------------

def checked_grad(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0,
                 has_aux: bool = False) -> Callable:
    """``grad`` that also reports non-finite results through ``checkify``.

    Returns a function ``(*args) -> (err, grads)``.  Call ``err.throw()`` to
    raise a ``ValueError`` if any gradient leaf is ``nan``/``inf``.  Unlike a
    Python ``try/except``, this works inside ``jit``: the error is carried as
    data and thrown on the host afterwards.
    """
    grad_fn = grad(fun, argnums=argnums, has_aux=has_aux)

    def with_check(*args, **kwargs):
        out = grad_fn(*args, **kwargs)
        grads = out[0] if has_aux else out
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(g))
                                    for g in jax.tree_util.tree_leaves(grads)]))
        checkify.check(finite, "non-finite gradient")
        return out

    return checkify.checkify(with_check)


def safe_grad(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0, has_aux: bool = False,
              on_nonfinite: str = "raise") -> Callable:
    """Gradient with explicit handling of non-finite values.

    Args:
        fun: Scalar function to differentiate.
        argnums: As in :func:`jax.grad`.
        has_aux: As in :func:`jax.grad`.
        on_nonfinite: ``'raise'`` (throw a ``ValueError`` - call the result
            outside ``jit``, or use :func:`checked_grad` inside), ``'zero'``
            (replace non-finite leaves with zeros, i.e. skip the update - a
            common mixed-precision recovery strategy) or ``'nan'`` (return
            as-is).
    """
    if on_nonfinite == "raise":
        checked = checked_grad(fun, argnums=argnums, has_aux=has_aux)

        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            err, out = checked(*args, **kwargs)
            err.throw()
            return out

        return wrapped

    grad_fn = grad(fun, argnums=argnums, has_aux=has_aux)
    if on_nonfinite == "nan":
        return grad_fn
    if on_nonfinite != "zero":
        raise ValueError(f"Unknown on_nonfinite={on_nonfinite!r}")

    @functools.wraps(fun)
    def zeroed(*args, **kwargs):
        out = grad_fn(*args, **kwargs)
        grads, aux = (out if has_aux else (out, None))
        grads = jax.tree_util.tree_map(lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
        return (grads, aux) if has_aux else grads

    return zeroed


# Kept for API compatibility: these are now plain aliases of the JAX functions.
safe_jacobian = jacobian
safe_hessian = hessian


# ---------------------------------------------------------------------------
# Value + derivative in one call
# ---------------------------------------------------------------------------

def grad_and_value(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0,
                   has_aux: bool = False) -> Callable:
    """``(grad, value)`` (or ``(grad, value, aux)``) from a single forward/backward pass."""
    vg = jax.value_and_grad(fun, argnums=argnums, has_aux=has_aux)

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        value, g = vg(*args, **kwargs)
        if has_aux:
            value, aux = value
            return g, value, aux
        return g, value

    return wrapped


def jacobian_and_value(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """``(jacobian, value)``; the value comes from the JVP/VJP primal, not a second call."""
    def wrapped(*args, **kwargs):
        value = fun(*args, **kwargs)
        return jacobian(fun, argnums=argnums)(*args, **kwargs), value

    return wrapped


def hessian_and_value(fun: Callable, argnums: Union[int, Tuple[int, ...]] = 0) -> Callable:
    """``(hessian, value)`` for a scalar function."""
    def wrapped(*args, **kwargs):
        return hessian(fun, argnums=argnums)(*args, **kwargs), fun(*args, **kwargs)

    return wrapped


# ---------------------------------------------------------------------------
# Jacobians: picking the mode
# ---------------------------------------------------------------------------

def auto_jacobian(fun: Callable, argnums: int = 0) -> Callable:
    """Pick ``jacfwd`` or ``jacrev`` from the input/output sizes at call time.

    Forward mode costs one JVP per *input* element, reverse mode one VJP per
    *output* element, so wide functions (few inputs, many outputs) prefer
    forward mode and tall ones reverse mode.
    """
    def wrapped(*args, **kwargs):
        out_shape = jax.eval_shape(fun, *args, **kwargs)
        n_out = sum(int(jnp.prod(jnp.array(o.shape))) for o in jax.tree_util.tree_leaves(out_shape))
        n_in = sum(int(jnp.prod(jnp.array(a.shape)))
                   for a in jax.tree_util.tree_leaves(args[argnums]))
        mode = jacfwd if n_in <= n_out else jacrev
        return mode(fun, argnums=argnums)(*args, **kwargs)

    return wrapped


def batch_jacobian(fun: Callable, argnums: int = 0) -> Callable:
    """``vmap(jacobian(fun))`` - per-example Jacobians for a batch."""
    return jax.vmap(jacobian(fun, argnums=argnums))


def batch_hessian(fun: Callable, argnums: int = 0) -> Callable:
    """``vmap(hessian(fun))`` - per-example Hessians for a batch."""
    return jax.vmap(hessian(fun, argnums=argnums))


# ---------------------------------------------------------------------------
# Products with the Hessian
# ---------------------------------------------------------------------------

def directional_derivative(fun: Callable, x: Any, v: Any) -> Any:
    """``J(x) v`` via a single JVP (forward mode)."""
    _, tangent = jax.jvp(fun, (x,), (v,))
    return tangent


def hvp(fun: Callable[[Any], Array], x: Any, v: Any) -> Any:
    """Hessian-vector product ``H(x) v`` by forward-over-reverse.

    ``grad(fun)`` is a reverse pass; applying ``jvp`` to it adds one forward
    pass.  Memory is that of a single gradient - no ``n x n`` Hessian.
    """
    return jax.jvp(grad(fun), (x,), (v,))[1]


def hvp_reverse_over_reverse(fun: Callable[[Any], Array], x: Any, v: Any) -> Any:
    """Same product as :func:`hvp` computed as ``grad(x -> <grad f(x), v>)``.

    Slower and more memory-hungry than forward-over-reverse; included so the
    two can be compared.
    """
    return grad(lambda y: jax.tree_util.tree_reduce(
        jnp.add, jax.tree_util.tree_map(lambda g, t: jnp.sum(g * t), grad(fun)(y), v)))(x)


def gauss_newton_vp(model: Callable[[Any], Array], x: Any, v: Any) -> Any:
    """Gauss-Newton matrix-vector product ``J^T J v`` for a residual model.

    For a least-squares loss ``0.5 ||r(x)||^2`` this is the positive
    semi-definite part of the Hessian; it is what Levenberg-Marquardt and
    natural-gradient methods use.
    """
    _, jvp_out = jax.jvp(model, (x,), (v,))
    _, vjp_fn = jax.vjp(model, x)
    return vjp_fn(jvp_out)[0]


def hessian_diagonal(fun: Callable[[Array], Array], x: Array) -> Array:
    """Exact Hessian diagonal via ``n`` HVPs with basis vectors (``x`` must be 1-D)."""
    n = x.shape[0]
    basis = jnp.eye(n, dtype=x.dtype)
    return jax.vmap(lambda e: jnp.dot(e, hvp(fun, x, e)))(basis)


def hessian_trace_hutchinson(fun: Callable[[Array], Array], x: Array, key: Array,
                             num_samples: int = 32) -> Array:
    """Hutchinson estimator ``E[v^T H v]`` with Rademacher ``v``; unbiased for ``tr(H)``.

    Each sample costs one HVP, so this scales to parameter counts where the
    Hessian itself could never be stored.
    """
    vs = jax.random.rademacher(key, (num_samples,) + x.shape, dtype=x.dtype)
    quad = jax.vmap(lambda v: jnp.sum(v * hvp(fun, x, v)))(vs)
    return jnp.mean(quad)


# ---------------------------------------------------------------------------
# Finite differences and gradient checking
# ---------------------------------------------------------------------------

def finite_diff_grad(fun: Callable[[Array], Array], x: Array, eps: Optional[float] = None,
                     method: str = "central") -> Array:
    """Numerical gradient of a scalar function.

    Args:
        fun: Scalar-valued function.
        x: Point (any shape).
        eps: Step; defaults to a dtype-aware value (see
            :func:`jax_nsl.core.numerics.default_fd_step`).  Ignored by the
            complex-step method, which uses ``1e-20``.
        method: ``'forward'``, ``'backward'``, ``'central'`` or ``'complex'``.
            The complex-step formula ``Im f(x + i h) / h`` has no
            subtraction, so ``h`` can be tiny and the result is exact to
            round-off - but ``fun`` must be analytic (no ``abs``, ``max``,
            comparisons) and support complex inputs.
    """
    from jax_nsl.core.numerics import default_fd_step

    flat = x.ravel()
    n = flat.size

    if method == "complex":
        h = 1e-20
        cdtype = jnp.result_type(flat.dtype, jnp.complex64)

        def partial(i):
            ei = jnp.zeros(n, dtype=cdtype).at[i].set(1j * h)
            return jnp.imag(fun((flat.astype(cdtype) + ei).reshape(x.shape))) / h

        return jax.vmap(partial)(jnp.arange(n)).reshape(x.shape).astype(x.dtype)

    if eps is None:
        eps = default_fd_step(x, order=2 if method == "central" else 1)
    f0 = fun(x)

    def partial(i):
        ei = jnp.zeros(n, dtype=flat.dtype).at[i].set(eps).reshape(x.shape)
        if method == "forward":
            return (fun(x + ei) - f0) / eps
        if method == "backward":
            return (f0 - fun(x - ei)) / eps
        if method == "central":
            return (fun(x + ei) - fun(x - ei)) / (2 * eps)
        raise ValueError(f"Unknown method: {method}")

    return jax.vmap(partial)(jnp.arange(n)).reshape(x.shape)


def gradient_checker(fun: Callable[[Array], Array], x: Array, eps: Optional[float] = None,
                     rtol: float = 1e-2, atol: float = 1e-4, method: str = "central"
                     ) -> Tuple[bool, float]:
    """Compare ``jax.grad`` with finite differences; returns ``(ok, max_abs_error)``.

    Default tolerances are appropriate for float32 central differences.  Use
    ``method='complex'`` with ``rtol=1e-5`` for analytic functions.
    """
    g_ad = grad(fun)(x)
    g_fd = finite_diff_grad(fun, x, eps=eps, method=method)
    max_error = float(jnp.max(jnp.abs(g_ad - g_fd)))
    return bool(jnp.allclose(g_ad, g_fd, rtol=rtol, atol=atol)), max_error


def gradient_check_report(fun: Callable[[Array], Array], x: Array) -> Dict[str, float]:
    """Max abs error of forward/central/complex-step estimates against ``jax.grad``."""
    g_ad = grad(fun)(x)
    out = {}
    for method in ("forward", "central", "complex"):
        try:
            g = finite_diff_grad(fun, x, method=method)
            out[method] = float(jnp.max(jnp.abs(g - g_ad)))
        except Exception as e:  # complex step needs an analytic fun
            out[method] = float("nan")
            out[f"{method}_error"] = str(e)[:80]
    return out


# ---------------------------------------------------------------------------
# Convenience wrappers (compute and return the value directly)
# ---------------------------------------------------------------------------

def compute_gradient(fun: Callable, x: Any, argnums: int = 0) -> Any:
    """``grad(fun, argnums)(x)``."""
    return grad(fun, argnums=argnums)(x)


def compute_jacobian(fun: Callable, x: Any, argnums: int = 0) -> Any:
    """``jacobian(fun, argnums)(x)``."""
    return jacobian(fun, argnums=argnums)(x)


def compute_hessian(fun: Callable, x: Any, argnums: int = 0) -> Any:
    """``hessian(fun, argnums)(x)``."""
    return hessian(fun, argnums=argnums)(x)
