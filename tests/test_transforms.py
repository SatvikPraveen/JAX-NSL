# tests/test_transforms.py
"""Tests for jax_nsl.transforms: jit, vmap, scan and control-flow utilities."""

import jax
import jax.numpy as jnp
import pytest
from jax import grad, jit, random, vmap

from jax_nsl.transforms.control_flow import (
    binary_search,
    clip_gradient,
    clip_gradient_norm,
    conditional_update,
    gather_nd,
    iterative_solver,
    safe_cond,
    safe_divide,
    scatter_add_nd,
    stable_softmax,
    switch_case,
    while_loop_safe,
)
from jax_nsl.transforms.jit_utils import (
    aot_compile,
    benchmark_jit,
    compile_info,
    count_compilations,
    efficient_jit,
    jit_with_static,
    profile_jit_compilation,
)
from jax_nsl.transforms.scan_utils import (
    cumulative_sum,
    dynamic_rnn,
    linear_recurrence,
    parallel_cumsum,
    rnn_scan,
    running_statistics,
    scan_layers,
    scan_with_checkpointing,
    solve_ode,
    stack_params,
    windowed_scan,
)
from jax_nsl.transforms.vmap_utils import (
    batched_gradient,
    batched_matmul,
    chunked_vmap,
    clip_per_example_gradients,
    loop_batch_apply,
    parallel_apply,
    per_example_gradients,
    vmap_with_signature,
)


class TestJitUtils:
    def test_jit_with_static(self):
        f = jit_with_static(lambda x, n: x**n, static_argnums=(1,))
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(f(x, 3), x**3)

    def test_efficient_jit(self):
        f = efficient_jit(lambda x: x * 2 + 1)
        x = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(f(x), x * 2 + 1)

    def test_benchmark_jit(self):
        x = random.normal(random.PRNGKey(0), (64, 64))
        warm, run = benchmark_jit(jnp.dot, x, x, warmup_runs=2, benchmark_runs=3)
        assert warm > 0 and run > 0

    def test_count_compilations_detects_retrace(self):
        f = count_compilations(lambda x: x + 1)
        f(jnp.ones(3)); f(jnp.ones(3))
        assert f.compilation_count() == 1
        f(jnp.ones(4))  # new shape -> retrace
        assert f.compilation_count() == 2
        f(jnp.ones(4, jnp.int32))  # new dtype -> retrace
        assert f.compilation_count() == 3

    def test_aot_compile_and_info(self):
        compiled = aot_compile(lambda a, b: a @ b, jnp.ones((8, 8)), jnp.ones((8, 8)))
        assert jnp.allclose(compiled(jnp.ones((8, 8)), jnp.ones((8, 8))), 8.0)
        info = compile_info(compiled)
        assert info["flops"] is None or info["flops"] >= 2 * 8 * 8 * 8 - 64

    def test_profile_jit_compilation(self):
        stats = profile_jit_compilation(lambda x: jnp.sin(x).sum(), jnp.ones(100))
        assert set(stats) >= {"compile_time", "jit_exec_time", "no_jit_time", "speedup"}


class TestVmapUtils:
    def test_batched_matmul(self):
        k1, k2 = random.split(random.PRNGKey(0))
        a = random.normal(k1, (4, 3, 3))
        b = random.normal(k2, (4, 3, 3))
        expected = jnp.stack([a[i] @ b[i] for i in range(4)])
        assert jnp.allclose(batched_matmul(a, b), expected, atol=1e-5)

    def test_batched_gradient(self):
        quadratic = lambda x: jnp.sum(x**2)  # noqa: E731
        x = random.normal(random.PRNGKey(42), (5, 3))
        assert jnp.allclose(batched_gradient(quadratic, x), 2 * x)

    def test_parallel_apply(self):
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])
        offsets = jnp.array([0.1, 0.2, 0.3, 0.4])
        out = parallel_apply(lambda x, o: x**2 + o, xs, offsets)
        assert jnp.allclose(out, xs**2 + offsets)
        assert jnp.allclose(loop_batch_apply(lambda x: x**2, xs), xs**2)

    def test_per_example_gradients_mean_equals_batch_gradient(self):
        params = {"w": jnp.array([1.0, -1.0]), "b": jnp.array(0.5)}
        xs = random.normal(random.PRNGKey(0), (8, 2))
        ys = random.normal(random.PRNGKey(1), (8,))

        def loss(p, x, y):
            return (p["w"] @ x + p["b"] - y) ** 2

        pe = per_example_gradients(loss, params, xs, ys)
        assert pe["w"].shape == (8, 2) and pe["b"].shape == (8,)
        batch_grad = grad(lambda p: jnp.mean(vmap(loss, in_axes=(None, 0, 0))(p, xs, ys)))(params)
        assert jnp.allclose(jnp.mean(pe["w"], axis=0), batch_grad["w"], atol=1e-5)

    def test_clip_per_example_gradients(self):
        pe = {"w": jnp.array([[3.0, 4.0], [0.3, 0.4]]), "b": jnp.array([0.0, 0.0])}
        clipped = clip_per_example_gradients(pe, max_norm=1.0)
        # example 0 has norm 5 -> scaled by 0.2; example 1 has norm 0.5 -> unchanged
        assert jnp.allclose(clipped["w"], (jnp.array([0.6, 0.8]) + jnp.array([0.3, 0.4])) / 2)

    def test_chunked_vmap_matches_vmap_with_remainder(self):
        xs = random.normal(random.PRNGKey(0), (10, 3))
        f = lambda x: jnp.sum(x**2)  # noqa: E731
        assert jnp.allclose(chunked_vmap(f, xs, chunk_size=4), vmap(f)(xs))

    def test_vmap_with_signature(self):
        @vmap_with_signature("(m,n),(n)->(m)")
        def matvec(a, x):
            return a @ x

        a = random.normal(random.PRNGKey(0), (5, 3, 2))
        x = random.normal(random.PRNGKey(1), (5, 2))
        assert jnp.allclose(matvec(a, x), jnp.einsum("bmn,bn->bm", a, x), atol=1e-5)


class TestScanUtils:
    def test_cumulative_sums(self):
        xs = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert jnp.allclose(cumulative_sum(xs), jnp.cumsum(xs))
        assert jnp.allclose(parallel_cumsum(xs), jnp.cumsum(xs))
        m = random.normal(random.PRNGKey(0), (3, 4))
        assert jnp.allclose(cumulative_sum(m, axis=1), jnp.cumsum(m, axis=1), atol=1e-6)

    def test_linear_recurrence_matches_sequential(self):
        a = random.uniform(random.PRNGKey(0), (16, 2), minval=0.5, maxval=0.99)
        b = random.normal(random.PRNGKey(1), (16, 2))
        x_par = linear_recurrence(a, b)

        def step(x, ab):
            x = ab[0] * x + ab[1]
            return x, x

        _, x_seq = jax.lax.scan(step, jnp.zeros(2), (a, b))
        assert jnp.allclose(x_par, x_seq, atol=1e-5)

    def test_running_statistics_welford(self):
        xs = random.normal(random.PRNGKey(0), (50,)) * 3 + 100.0
        means, variances = running_statistics(xs)
        assert jnp.allclose(means[-1], jnp.mean(xs), atol=1e-4)
        assert jnp.allclose(variances[-1], jnp.var(xs, ddof=1), rtol=1e-3)
        assert variances[0] == 0.0

    def test_rnn_scan(self):
        k1, k2, k3 = random.split(random.PRNGKey(0), 3)
        w_h = random.normal(k1, (4, 4))
        w_x = random.normal(k2, (4, 3))
        inputs = random.normal(k3, (5, 3))

        def cell(h, x):
            h_new = jnp.tanh(w_h @ h + w_x @ x)
            return h_new, h_new

        final_h, all_h = rnn_scan(cell, jnp.zeros(4), inputs)
        assert final_h.shape == (4,) and all_h.shape == (5, 4)
        assert jnp.allclose(all_h[0], jnp.tanh(w_x @ inputs[0]))
        _, rev = rnn_scan(cell, jnp.zeros(4), inputs, reverse=True)
        assert jnp.allclose(rev[-1], jnp.tanh(w_x @ inputs[-1]))

    def test_dynamic_rnn_freezes_after_length(self):
        def cell(h, x):
            return h + x, h + x

        inputs = jnp.ones((4, 2, 1))  # time=4, batch=2
        lengths = jnp.array([2, 4])
        final, outs = dynamic_rnn(cell, inputs, lengths, jnp.zeros((2, 1)))
        assert jnp.allclose(final[:, 0], jnp.array([2.0, 4.0]))
        assert jnp.allclose(outs[:, 0, 0], jnp.array([1.0, 2.0, 0.0, 0.0]))

    def test_scan_layers_equals_loop(self):
        layers = [{"w": random.normal(random.PRNGKey(i), (3, 3))} for i in range(4)]
        layer = lambda p, x: jnp.tanh(p["w"] @ x)  # noqa: E731
        x = jnp.ones(3)
        expected = x
        for p in layers:
            expected = layer(p, expected)
        stacked = stack_params(layers)
        assert stacked["w"].shape == (4, 3, 3)
        assert jnp.allclose(scan_layers(layer, stacked, x), expected, atol=1e-6)
        assert jnp.allclose(scan_layers(layer, stacked, x, remat=True), expected, atol=1e-6)

    @pytest.mark.parametrize("method,tol", [("euler", 6e-2), ("midpoint", 2e-3), ("rk4", 1e-5)])
    def test_solve_ode_orders(self, method, tol):
        t = jnp.linspace(0.0, 1.0, 11)
        sol = solve_ode(lambda y, t: -y, jnp.array([2.0]), t, method=method)
        assert sol.shape == (11, 1)
        assert jnp.allclose(sol[:, 0], 2.0 * jnp.exp(-t), rtol=tol)

    def test_solve_ode_is_differentiable_wrt_parameters(self):
        t = jnp.linspace(0.0, 1.0, 21)

        def final_value(k):
            return solve_ode(lambda y, t: -k * y, jnp.array(1.0), t)[-1]

        # y(1) = exp(-k) -> dy/dk = -exp(-k)
        assert jnp.allclose(grad(final_value)(0.5), -jnp.exp(-0.5), atol=1e-4)

    def test_scan_with_checkpointing_matches_plain_scan(self):
        w = random.normal(random.PRNGKey(0), (3, 3)) * 0.3
        xs = random.normal(random.PRNGKey(1), (11, 3))  # 11 is not a multiple of 4

        def body(h, x):
            h = jnp.tanh(w @ h + x)
            return h, h

        def run(fn):
            def loss(w_):
                def body_w(h, x):
                    h = jnp.tanh(w_ @ h + x)
                    return h, h
                carry, ys = fn(body_w)
                return jnp.sum(ys) + jnp.sum(carry)
            return loss

        plain = run(lambda b: jax.lax.scan(b, jnp.zeros(3), xs))
        remat = run(lambda b: scan_with_checkpointing(b, jnp.zeros(3), xs, checkpoint_every=4))
        assert jnp.allclose(plain(w), remat(w), atol=1e-6)
        assert jnp.allclose(grad(plain)(w), grad(remat)(w), atol=1e-5)

    def test_windowed_scan(self):
        xs = jnp.arange(6.0)
        _, out = windowed_scan(jnp.sum, xs, window_size=3, stride=1)
        assert jnp.allclose(out, jnp.array([3.0, 6.0, 9.0, 12.0]))


class TestControlFlow:
    def test_safe_divide(self):
        result = safe_divide(jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 0.0, 0.5]))
        assert jnp.allclose(result[0], 0.5) and jnp.allclose(result[2], 6.0, rtol=1e-6)
        assert jnp.isfinite(result[1])

    def test_safe_cond_reports_mismatch(self):
        with pytest.raises(TypeError, match="differ"):
            safe_cond(True, lambda x: x, lambda x: x.astype(jnp.int32), jnp.ones(2))
        assert safe_cond(jnp.array(False), lambda x: x + 1, lambda x: x - 1, jnp.ones(2))[0] == 0.0

    def test_switch_case(self):
        branches = [lambda x: x, lambda x: 2 * x, lambda x: 3 * x]
        assert switch_case(jnp.int32(2), branches, jnp.ones(1))[0] == 3.0
        assert jit(lambda i: switch_case(i, branches, jnp.ones(1)))(1)[0] == 2.0

    def test_while_loop_safe_cap(self):
        out = while_loop_safe(lambda v: v < 100, lambda v: v + 1, 0, max_iterations=10)
        assert out == 10

    def test_conditional_update(self):
        x = jnp.array([1.0, -2.0, 3.0])
        assert jnp.array_equal(conditional_update(x < 0, x, lambda v: -v), jnp.abs(x))

    def test_binary_search_and_iterative_solver(self):
        root = binary_search(lambda x: x**2, 2.0, 0.0, 2.0, tolerance=1e-5)
        assert jnp.allclose(root, jnp.sqrt(2.0), atol=1e-4)
        x, conv = iterative_solver(lambda x: jnp.cos(x), jnp.array(1.0), tolerance=1e-6, max_iterations=200)
        assert bool(conv) and jnp.allclose(x, jnp.cos(x), atol=1e-5)

    def test_gather_scatter_nd(self):
        params = jnp.arange(12.0).reshape(3, 4)
        idx = jnp.array([[0, 1], [2, 3]])
        assert jnp.array_equal(gather_nd(params, idx), jnp.array([1.0, 11.0]))
        out = scatter_add_nd(jnp.zeros((3, 4)), idx, jnp.array([1.0, 2.0]))
        assert out[0, 1] == 1.0 and out[2, 3] == 2.0

    def test_clip_gradient_elementwise(self):
        g = grad(lambda x: jnp.sum(clip_gradient(x, -1.0, 1.0) ** 2))(jnp.array([10.0, 0.1]))
        assert jnp.allclose(g, jnp.array([1.0, 0.2]))

    def test_clip_gradient_norm_transform(self):
        loss = lambda x: jnp.sum(x**2)  # noqa: E731
        x = jnp.array([10.0, -5.0, 2.0])
        g = grad(clip_gradient_norm(loss, max_norm=1.0))(x)
        assert jnp.linalg.norm(g) <= 1.0 + 1e-6
        assert jnp.allclose(g / jnp.linalg.norm(g), grad(loss)(x) / jnp.linalg.norm(grad(loss)(x)))
        g_small = grad(clip_gradient_norm(loss, max_norm=100.0))(x)
        assert jnp.allclose(g_small, grad(loss)(x))

    def test_stable_softmax_reexport(self):
        out = stable_softmax(jnp.array([1000.0, 999.0, 1001.0]))
        assert jnp.allclose(jnp.sum(out), 1.0) and jnp.all(jnp.isfinite(out))


class TestTransformComposition:
    def test_jit_vmap_composition(self):
        f = jit(vmap(lambda a, x: a @ x, in_axes=(0, 0)))
        k1, k2 = random.split(random.PRNGKey(0))
        a = random.normal(k1, (3, 4, 4))
        x = random.normal(k2, (3, 4))
        assert jnp.allclose(f(a, x), jnp.einsum("bij,bj->bi", a, x), atol=1e-5)

    def test_grad_jit_vmap_composition(self):
        def loss(w, x, y):
            return 0.5 * (jnp.dot(w, x) - y) ** 2

        f = jit(vmap(grad(loss), in_axes=(None, 0, 0)))
        k1, k2, k3 = random.split(random.PRNGKey(42), 3)
        grads = f(random.normal(k1, (5,)), random.normal(k2, (10, 5)), random.normal(k3, (10,)))
        assert grads.shape == (10, 5)

    def test_vmap_of_scan(self):
        def step(carry, x):
            return carry + x, carry + x

        f = vmap(lambda c, s: jax.lax.scan(step, c, s))
        finals, outs = f(jnp.array([0.0, 1.0, 2.0]), jnp.ones((3, 5)))
        assert jnp.allclose(finals, jnp.array([5.0, 6.0, 7.0]))
        assert outs.shape == (3, 5)


if __name__ == "__main__":
    pytest.main([__file__])
