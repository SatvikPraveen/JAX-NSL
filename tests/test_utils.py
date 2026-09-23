# tests/test_utils.py
"""Tests for jax_nsl.utils: pytree helpers and benchmarking."""

import jax
import jax.numpy as jnp
import pytest
from jax import random

from jax_nsl.utils.benchmarking import (
    PerformanceProfiler,
    benchmark_function,
    compare_implementations,
    count_flops,
    create_performance_report,
    profile_memory_usage,
    time_jit_compilation,
)
from jax_nsl.utils.tree_utils import (
    tree_apply_mask,
    tree_cast,
    tree_concatenate,
    tree_diff,
    tree_dot,
    tree_flatten_dict,
    tree_flatten_with_path,
    tree_norm,
    tree_paths,
    tree_random_like,
    tree_reduce,
    tree_select,
    tree_shapes,
    tree_slice,
    tree_stack,
    tree_statistics,
    tree_unflatten_dict,
    tree_unstack,
    tree_update_at_path,
)


@pytest.fixture
def tree():
    return {
        "layer1": {"W": jnp.ones((2, 3)), "b": jnp.zeros(3)},
        "layer2": [jnp.ones(2), jnp.array(5.0)],
    }


class TestTreeUtils:
    def test_paths_and_flat_dict_roundtrip(self, tree):
        assert tree_paths(tree) == ["layer1/W", "layer1/b", "layer2/0", "layer2/1"]
        flat = tree_flatten_dict(tree)
        assert set(flat) == {"layer1/W", "layer1/b", "layer2/0", "layer2/1"}
        nested = tree_unflatten_dict(flat)
        assert nested["layer1"]["W"].shape == (2, 3) and nested["layer2"]["1"] == 5.0

    def test_flatten_with_path_matches_jax(self, tree):
        pairs, treedef = tree_flatten_with_path(tree)
        assert len(pairs) == 4 and treedef == jax.tree_util.tree_structure(tree)

    def test_update_at_path_dict_list_namedtuple(self, tree):
        new = tree_update_at_path(tree, ["layer1", "b"], jnp.ones(3))
        assert jnp.all(new["layer1"]["b"] == 1.0) and jnp.all(tree["layer1"]["b"] == 0.0)
        new = tree_update_at_path(tree, ["layer2", 1], jnp.array(7.0))
        assert new["layer2"][1] == 7.0
        from collections import namedtuple

        S = namedtuple("S", ["a", "b"])
        s = tree_update_at_path(S(1, 2), ["b"], 9)
        assert s.b == 9
        # JAX key paths are accepted too.
        path = jax.tree_util.tree_leaves_with_path(tree)[0][0]
        assert tree_update_at_path(tree, path, jnp.zeros((2, 3)))["layer1"]["W"].sum() == 0

    def test_arithmetic(self, tree):
        assert jnp.allclose(tree_norm(tree), jnp.sqrt(6 + 2 + 25))
        assert jnp.allclose(tree_dot(tree, tree), 6 + 2 + 25)
        assert tree_reduce(tree, lambda a, b: a + jnp.sum(b), 0.0) == 6 + 2 + 5

    def test_random_like_and_cast(self, tree):
        r = tree_random_like(random.PRNGKey(0), tree)
        assert jax.tree_util.tree_structure(r) == jax.tree_util.tree_structure(tree)
        assert not jnp.allclose(r["layer1"]["W"], r["layer1"]["W"][::-1])
        c = tree_cast({"w": jnp.ones(2), "i": jnp.arange(2)}, jnp.bfloat16)
        assert c["w"].dtype == jnp.bfloat16 and c["i"].dtype == jnp.int32

    def test_select_and_mask(self, tree):
        matrices = tree_select(tree, lambda leaf: leaf.ndim == 2)
        assert matrices["layer1"]["b"] is None and matrices["layer1"]["W"] is not None
        masked = tree_apply_mask({"a": jnp.array([1.0, 2.0])}, {"a": jnp.array([True, False])})
        assert jnp.array_equal(masked["a"], jnp.array([1.0, 0.0]))

    def test_stack_unstack_concat_slice(self):
        trees = [{"w": jnp.full(2, float(i))} for i in range(3)]
        stacked = tree_stack(trees)
        assert stacked["w"].shape == (3, 2)
        assert jnp.allclose(tree_unstack(stacked)[2]["w"], 2.0)
        cat = tree_concatenate(trees)
        assert cat["w"].shape == (6,)
        assert jnp.array_equal(tree_slice(cat, slice(0, 2))["w"], jnp.zeros(2))

    def test_diff_and_statistics(self, tree):
        d = tree_diff(tree, tree)
        assert d["trees_equal"] and d["num_differences"] == 0
        other = tree_update_at_path(tree, ["layer1", "b"], jnp.ones(3))
        d = tree_diff(tree, other)
        assert not d["trees_equal"] and d["differences"][0]["path"] == "layer1/b"
        assert tree_diff(tree, {"x": 1})["structure_differs"]
        stats = tree_statistics(tree)
        assert stats["num_leaves"] == 4 and stats["max_depth"] == 2
        assert stats["array_statistics"]["total_elements"] == 6 + 3 + 2 + 1
        assert tree_shapes(tree)["layer1"]["W"] == ((2, 3), "float32")


class TestBenchmarking:
    def test_benchmark_function_blocks_on_trees(self):
        f = jax.jit(lambda x: (x + 1, x * 2))
        stats = benchmark_function(f, jnp.ones(16), num_runs=3, num_warmup=1)
        assert (
            stats["num_runs"] == 3 and stats["min_time"] <= stats["mean_time"] <= stats["max_time"]
        )

    def test_time_jit_compilation(self):
        stats = time_jit_compilation(lambda x: jnp.sin(x) ** 2, jnp.ones(32))
        assert stats["compile_time"] >= 0 and stats["execution_time"] > 0

    def test_count_flops_and_memory_profile(self):
        flops = count_flops(lambda a, b: a @ b, jnp.ones((16, 16)), jnp.ones((16, 16)))
        assert flops is None or flops >= 2 * 16**3 - 256
        mem = profile_memory_usage(lambda a: a @ a, jnp.ones((16, 16)))
        assert mem["result_bytes"] == 16 * 16 * 4

    def test_compare_and_report(self):
        results = compare_implementations(
            {"jit": jax.jit(jnp.sum), "eager": jnp.sum, "bad": lambda x: x @ jnp.ones(3)},
            jnp.ones((8, 8, 8)),
            num_runs=2,
        )
        assert "error" in results["bad"] and "speedup" in results["jit"]
        report = create_performance_report(results)
        assert "jit:" in report and "ERROR" in report

    def test_profiler_context(self):
        with PerformanceProfiler("op") as p:
            p.track(jax.jit(lambda x: x * 2)(jnp.ones(8)))
        assert p.duration is not None and p.duration > 0
