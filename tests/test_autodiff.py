# tests/test_autodiff.py
"""Tests for jax_nsl.autodiff: derivatives, custom rules, implicit differentiation."""

import jax
import jax.numpy as jnp
import pytest
from jax import grad, hessian, jvp, random

from jax_nsl.autodiff.custom_jvp import (
    custom_sqrt_jvp,
    gaussian_activation_jvp,
    learnable_activation_jvp,
    smooth_abs_jvp,
    soft_sign_jvp,
)
from jax_nsl.autodiff.custom_vjp import (
    clip_gradient_vjp,
    custom_sqrt_vjp,
    gated_linear_unit_vjp,
    gradient_reversal,
    smooth_abs_vjp,
    ste_round,
    straight_through_estimator,
    swish_vjp,
)
from jax_nsl.autodiff.grad_jac_hess import (
    auto_jacobian,
    checked_grad,
    compute_gradient,
    compute_hessian,
    compute_jacobian,
    finite_diff_grad,
    gauss_newton_vp,
    grad_and_value,
    gradient_checker,
    hessian_diagonal,
    hessian_trace_hutchinson,
    hvp,
    hvp_reverse_over_reverse,
    safe_grad,
)
from jax_nsl.autodiff.implicit import fixed_point, fixed_point_unrolled, implicit_newton_solve


class TestGradJacHess:
    def test_gradient_simple(self):
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(compute_gradient(lambda x: jnp.sum(x**2), x), 2.0 * x)

    def test_jacobian_vector_function(self):
        def f(x):
            return jnp.array([x[0] ** 2 + x[1], x[0] - x[1] ** 2])

        x = jnp.array([2.0, 3.0])
        expected = jnp.array([[2 * x[0], 1.0], [1.0, -2 * x[1]]])
        assert jnp.allclose(compute_jacobian(f, x), expected)
        assert jnp.allclose(auto_jacobian(f)(x), expected)

    def test_hessian_scalar_function(self):
        def f(x):
            return x[0] ** 3 + x[1] ** 2 + x[0] * x[1]

        x = jnp.array([1.0, 2.0])
        assert jnp.allclose(compute_hessian(f, x), jnp.array([[6 * x[0], 1.0], [1.0, 2.0]]))

    def test_grad_and_value_single_pass(self):
        calls = []

        def f(x):
            calls.append(1)
            return jnp.sum(x**2)

        g, v = grad_and_value(f)(jnp.array([1.0, 2.0]))
        assert jnp.allclose(g, jnp.array([2.0, 4.0])) and jnp.allclose(v, 5.0)
        assert len(calls) == 1

    def test_grad_and_value_with_aux(self):
        def f(x):
            return jnp.sum(x**2), {"n": x.shape[0]}

        g, v, aux = grad_and_value(f, has_aux=True)(jnp.ones(3))
        assert aux["n"] == 3 and jnp.allclose(v, 3.0)


class TestSafeGrad:
    def test_checked_grad_flags_nan(self):
        f = lambda x: jnp.sum(jnp.sqrt(x))  # noqa: E731  grad is inf at 0
        err, g = checked_grad(f)(jnp.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="non-finite"):
            err.throw()
        err, g = checked_grad(f)(jnp.array([1.0, 4.0]))
        err.throw()  # no error
        assert jnp.allclose(g, jnp.array([0.5, 0.25]))

    def test_safe_grad_raise_and_zero(self):
        f = lambda x: jnp.sum(jnp.sqrt(x))  # noqa: E731
        with pytest.raises(ValueError):
            safe_grad(f)(jnp.array([0.0, 1.0]))
        g = safe_grad(f, on_nonfinite="zero")(jnp.array([0.0, 1.0]))
        assert jnp.array_equal(g, jnp.array([0.0, 0.5]))

    def test_checked_grad_inside_jit(self):
        f = lambda x: jnp.sum(jnp.log(x))  # noqa: E731
        err, g = jax.jit(checked_grad(f))(jnp.array([1.0, 2.0]))
        err.throw()
        assert jnp.allclose(g, jnp.array([1.0, 0.5]))


class TestHessianProducts:
    def _quadratic(self):
        a = jnp.array([[3.0, 1.0, 0.0], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0]])
        return (lambda x: 0.5 * x @ a @ x), a

    def test_hvp_matches_dense_hessian(self):
        f, a = self._quadratic()
        x, v = jnp.ones(3), jnp.array([1.0, -1.0, 2.0])
        assert jnp.allclose(hvp(f, x, v), a @ v, atol=1e-6)
        assert jnp.allclose(hvp_reverse_over_reverse(f, x, v), a @ v, atol=1e-6)

    def test_hvp_on_pytree(self):
        f = lambda p: jnp.sum(p["w"] ** 2) * p["b"] ** 2  # noqa: E731
        p = {"w": jnp.array([1.0, 2.0]), "b": jnp.array(3.0)}
        v = {"w": jnp.array([1.0, 0.0]), "b": jnp.array(0.0)}
        out = hvp(f, p, v)
        assert jnp.allclose(out["w"], jnp.array([18.0, 0.0]))

    def test_hessian_diagonal_and_trace(self):
        f, a = self._quadratic()
        x = jnp.ones(3)
        assert jnp.allclose(hessian_diagonal(f, x), jnp.diag(a), atol=1e-6)
        est = hessian_trace_hutchinson(f, x, random.PRNGKey(0), num_samples=256)
        assert jnp.allclose(est, jnp.trace(a), rtol=0.15)

    def test_gauss_newton_vp(self):
        j = jnp.array([[1.0, 2.0], [0.0, 1.0], [3.0, -1.0]])
        model = lambda x: j @ x  # noqa: E731
        v = jnp.array([1.0, 1.0])
        assert jnp.allclose(gauss_newton_vp(model, jnp.zeros(2), v), j.T @ (j @ v))


class TestFiniteDifferences:
    def test_gradient_checker_default(self):
        ok, err = gradient_checker(lambda x: jnp.sum(jnp.sin(x) * x), jnp.array([0.5, 1.0, 2.0]))
        assert ok

    def test_complex_step_is_machine_precision(self):
        f = lambda x: jnp.sum(jnp.exp(x) * x**2)  # noqa: E731
        x = jnp.array([0.5, 1.0, 2.0])
        g_cs = finite_diff_grad(f, x, method="complex")
        assert jnp.allclose(g_cs, grad(f)(x), rtol=1e-6)
        g_fd = finite_diff_grad(f, x, method="central")
        assert jnp.max(jnp.abs(g_cs - grad(f)(x))) <= jnp.max(jnp.abs(g_fd - grad(f)(x)))


class TestCustomVJP:
    def test_custom_sqrt_vjp(self):
        x = jnp.array([4.0, 9.0, 16.0])
        assert jnp.allclose(custom_sqrt_vjp(x), jnp.sqrt(x))
        g = grad(lambda x: jnp.sum(custom_sqrt_vjp(x)))(x)
        assert jnp.allclose(g, 0.5 / jnp.sqrt(x))
        assert jnp.isfinite(grad(lambda x: jnp.sum(custom_sqrt_vjp(x)))(jnp.array([0.0]))).all()

    def test_smooth_abs_vjp(self):
        x = jnp.array([-1e-3, 0.0, 1e-3])
        g = grad(lambda x: jnp.sum(smooth_abs_vjp(x, eps=1e-2)))(x)
        assert jnp.all(jnp.isfinite(g))
        assert g[0] < 0 < g[2]

    def test_clip_gradient_vjp(self):
        x = jnp.array([1.0, 2.0, 3.0])
        g = grad(lambda x: jnp.sum(10.0 * clip_gradient_vjp(x, -0.5, 0.5)))(x)
        assert jnp.allclose(g, 0.5)

    def test_straight_through_estimator(self):
        x = jnp.array([0.2, 0.7])
        assert jnp.array_equal(straight_through_estimator(x), jnp.array([0.0, 1.0]))
        assert jnp.allclose(grad(lambda x: jnp.sum(straight_through_estimator(x)))(x), 1.0)

    def test_ste_round(self):
        x = jnp.array([0.4, 1.6])
        assert jnp.array_equal(ste_round(x), jnp.array([0.0, 2.0]))
        assert jnp.allclose(grad(lambda x: jnp.sum(3.0 * ste_round(x)))(x), 3.0)

    def test_gradient_reversal(self):
        x = jnp.array([1.0, 2.0])
        assert jnp.array_equal(gradient_reversal(x), x)
        g = grad(lambda x: jnp.sum(x**2 + gradient_reversal(x**2, 2.0)))(x)
        assert jnp.allclose(g, 2 * x - 2.0 * 2 * x)

    def test_swish_and_glu_match_autodiff(self):
        x = random.normal(random.PRNGKey(0), (6,))
        ref_swish = grad(lambda x: jnp.sum(x * jax.nn.sigmoid(x)))(x)
        assert jnp.allclose(grad(lambda x: jnp.sum(swish_vjp(x)))(x), ref_swish, atol=1e-6)
        ref_glu = grad(lambda x: jnp.sum(jax.nn.glu(x)))(x)
        assert jnp.allclose(
            grad(lambda x: jnp.sum(gated_linear_unit_vjp(x)))(x), ref_glu, atol=1e-6
        )


class TestCustomJVP:
    def test_custom_sqrt_jvp(self):
        x, v = jnp.array([4.0, 9.0]), jnp.array([1.0, 1.0])
        primals, tangents = jvp(custom_sqrt_jvp, (x,), (v,))
        assert jnp.allclose(primals, jnp.sqrt(x))
        assert jnp.allclose(tangents, 0.5 / jnp.sqrt(x) * v)

    def test_smooth_abs_jvp(self):
        x, v = jnp.array([-1.0, 0.0, 1.0]), jnp.ones(3)
        primals, tangents = jvp(lambda x: smooth_abs_jvp(x, eps=1e-2), (x,), (v,))
        assert jnp.all(jnp.isfinite(primals)) and jnp.all(jnp.isfinite(tangents))

    @pytest.mark.parametrize(
        "fn,ref",
        [
            (soft_sign_jvp, lambda x: x / (1.0 + jnp.abs(x))),
            (gaussian_activation_jvp, lambda x: jnp.exp(-(x**2) / 2)),
        ],
    )
    def test_jvp_rules_match_autodiff(self, fn, ref):
        x = jnp.array([-2.0, -0.5, 0.5, 2.0])
        v = jnp.ones(4)
        _, t_custom = jvp(fn, (x,), (v,))
        _, t_ref = jvp(ref, (x,), (v,))
        assert jnp.allclose(t_custom, t_ref, atol=1e-6)

    def test_learnable_activation_polynomial(self):
        coeffs = jnp.array([1.0, 0.0, 2.0])  # 1 + 2x^2
        x = jnp.array([0.5, 2.0])
        y, dy = jvp(lambda x: learnable_activation_jvp(x, coeffs), (x,), (jnp.ones(2),))
        assert jnp.allclose(y, 1 + 2 * x**2)
        assert jnp.allclose(dy, 4 * x)


class TestImplicitDifferentiation:
    def test_fixed_point_matches_closed_form(self):
        # x = tanh(a x + b): differentiate x*(a, b) implicitly and compare to
        # the implicit function theorem evaluated with the dense Jacobian.
        def f(params, x):
            a, b = params
            return jnp.tanh(a * x + b)

        params = (jnp.array(0.5), jnp.array(0.3))
        x_star = fixed_point(f, params, jnp.array(0.0), tolerance=1e-7)
        assert jnp.allclose(x_star, f(params, x_star), atol=1e-5)

        g_a, g_b = grad(lambda p: fixed_point(f, p, jnp.array(0.0), tolerance=1e-7))(params)
        dfdx = grad(lambda x: f(params, x))(x_star)
        dfda, dfdb = grad(f, argnums=0)(params, x_star)
        assert jnp.allclose(g_a, dfda / (1 - dfdx), atol=1e-4)
        assert jnp.allclose(g_b, dfdb / (1 - dfdx), atol=1e-4)

    def test_fixed_point_vs_unrolled(self):
        def f(params, x):
            return 0.5 * jnp.cos(x) + params

        p = jnp.array(0.2)
        g_implicit = grad(lambda p: fixed_point(f, p, jnp.array(0.0), tolerance=1e-8))(p)
        g_unrolled = grad(lambda p: fixed_point_unrolled(f, p, jnp.array(0.0), 100))(p)
        assert jnp.allclose(g_implicit, g_unrolled, atol=1e-4)

    def test_fixed_point_pytree_state_and_jit(self):
        def f(params, x):
            return {"u": 0.3 * jnp.sin(x["v"]) + params, "v": 0.3 * jnp.cos(x["u"])}

        x0 = {"u": jnp.array(0.0), "v": jnp.array(0.0)}
        sol = jax.jit(lambda p: fixed_point(f, p, x0, tolerance=1e-7))(jnp.array(0.1))
        again = f(jnp.array(0.1), sol)
        assert jnp.allclose(sol["u"], again["u"], atol=1e-5)
        g = grad(lambda p: fixed_point(f, p, x0, tolerance=1e-7)["u"])(jnp.array(0.1))
        assert jnp.isfinite(g)

    def test_implicit_newton_solve_gradient(self):
        # Root of x^2 - p = 0 is sqrt(p); d/dp = 1 / (2 sqrt(p)).
        residual = lambda p, x: x**2 - p  # noqa: E731
        p = jnp.array([4.0])
        root = implicit_newton_solve(residual, p, jnp.array([1.0]))
        assert jnp.allclose(root, 2.0, atol=1e-5)
        g = grad(lambda p: implicit_newton_solve(residual, p, jnp.array([1.0])).sum())(p)
        assert jnp.allclose(g, 0.25, atol=1e-4)


class TestMixedAndHigherOrder:
    def test_grad_of_jvp(self):
        f = lambda x: jnp.sum(x**3)  # noqa: E731
        x, v = jnp.array([1.0, 2.0]), jnp.array([1.0, 0.0])
        g = grad(lambda x: jvp(f, (x,), (v,))[1])(x)
        assert jnp.allclose(g, 6 * x * v)

    def test_jvp_of_grad(self):
        f = lambda x: jnp.sum(x**4)  # noqa: E731
        x, v = jnp.array([1.0, 2.0]), jnp.array([0.5, 0.5])
        assert jnp.allclose(jvp(grad(f), (x,), (v,))[1], 12 * x**2 * v)

    def test_third_derivative(self):
        assert jnp.allclose(grad(grad(grad(lambda x: x**4)))(2.0), 48.0)

    def test_mixed_partials(self):
        f = lambda x, y: x**2 * y**3  # noqa: E731
        assert jnp.allclose(grad(grad(f, argnums=0), argnums=1)(2.0, 3.0), 6.0 * 2.0 * 9.0)

    def test_hessian_conditioning(self):
        f = lambda x: 1e6 * x[0] ** 2 + 1e-6 * x[1] ** 2  # noqa: E731
        h = hessian(f)(jnp.array([1.0, 1.0]))
        assert jnp.allclose(jnp.diag(h), jnp.array([2e6, 2e-6]))


if __name__ == "__main__":
    pytest.main([__file__])
