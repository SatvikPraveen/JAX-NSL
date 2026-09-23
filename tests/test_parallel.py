# tests/test_parallel.py
"""Tests for jax_nsl.parallel on 8 virtual CPU devices (see conftest.py)."""

import jax
import jax.numpy as jnp
import pytest
from jax import pmap, random
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from jax_nsl.models.mlp import create_mlp
from jax_nsl.parallel.collectives import (
    all_gather,
    all_reduce_mean,
    alltoall,
    broadcast,
    compute_communication_volume,
    distributed_dot,
    gradient_synchronization,
    reduce_scatter,
    ring_all_reduce,
    sync_batch_stats,
    tree_all_reduce,
)
from jax_nsl.parallel.pjit_utils import (
    check_sharding_compatibility,
    create_mesh,
    create_transformer_partition_specs,
    estimate_memory_per_device,
    fsdp_rules,
    make_sharded_train_step,
    partition_params,
    partition_specs,
    setup_model_parallelism,
    shard_array,
    sharded_matmul_row_parallel,
    sharded_matmul_shard_map,
    sharding_summary,
)
from jax_nsl.parallel.pmap_utils import (
    create_parallel_inference_fn,
    create_pmap_train_step,
    data_parallel_step,
    replicate_params,
    shard_batch,
    sync_gradients,
    unreplicate_params,
)
from jax_nsl.training.losses import cross_entropy_loss
from jax_nsl.training.optimizers import sgd_optimizer

N = jax.device_count()
multi = pytest.mark.skipif(N < 2, reason="needs >= 2 devices (set XLA_FLAGS, see conftest)")


class TestDeviceSetup:
    def test_virtual_devices_present(self):
        assert N == 8, "conftest should expose 8 virtual CPU devices"


@multi
class TestPmapUtils:
    def test_replicate_and_unreplicate(self):
        params = {"W": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.5, 1.5])}
        rep = replicate_params(params)
        assert rep["W"].shape == (N, 2, 2)
        assert len(rep["W"].sharding.device_set) == N
        assert jnp.allclose(unreplicate_params(rep)["W"], params["W"])

    def test_shard_batch(self):
        batch = {"x": jnp.ones((16, 3)), "y": jnp.ones(16)}
        sharded = shard_batch(batch)
        assert sharded["x"].shape == (N, 16 // N, 3)
        with pytest.raises(ValueError):
            shard_batch({"x": jnp.ones((N + 1, 3))})

    def test_sync_gradients(self):
        grads = jnp.arange(N * 3, dtype=jnp.float32).reshape(N, 3)
        synced = pmap(sync_gradients, axis_name="batch")(grads)
        for i in range(N):
            assert jnp.allclose(synced[i], jnp.mean(grads, axis=0))

    def test_data_parallel_step_keeps_replicas_in_sync(self):
        def loss_fn(params, batch):
            return jnp.mean((batch["x"] @ params["W"] - batch["y"]) ** 2)

        key = random.PRNGKey(0)
        params = {"W": random.normal(key, (3, 2))}
        batch = {"x": random.normal(key, (N, 4, 3)), "y": random.normal(key, (N, 4, 2))}
        new_params, loss = data_parallel_step(loss_fn, replicate_params(params), batch, lr=0.01)
        assert new_params["W"].shape == (N, 3, 2) and loss.shape == (N,)
        for i in range(1, N):
            assert jnp.allclose(new_params["W"][0], new_params["W"][i], atol=1e-6)
        # Equivalent to a single-device step on the concatenated batch.
        full = {"x": batch["x"].reshape(-1, 3), "y": batch["y"].reshape(-1, 2)}
        g = jax.grad(loss_fn)(params, full)
        assert jnp.allclose(new_params["W"][0], params["W"] - 0.01 * g["W"], atol=1e-5)

    def test_pmap_train_step_and_inference(self):
        params, forward_fn, _ = create_mlp([4, 8, 3], seed=0)
        init, update = sgd_optimizer(0.1)
        step = create_pmap_train_step(forward_fn, cross_entropy_loss, update)
        state = replicate_params(init(params))
        batch = shard_batch(
            {
                "inputs": random.normal(random.PRNGKey(0), (16, 4)),
                "labels": random.randint(random.PRNGKey(1), (16,), 0, 3),
            }
        )
        state, metrics = step(state, batch)
        assert metrics["loss"].shape == (N,)
        assert int(unreplicate_params(state).step) == 1
        infer = create_parallel_inference_fn(forward_fn)
        out = infer(unreplicate_params(state).params, jnp.ones((13, 4)))  # ragged -> padded
        assert out.shape == (13, 3)


@multi
class TestCollectives:
    def test_all_reduce_mean(self):
        values = jnp.arange(N, dtype=jnp.float32)
        out = pmap(all_reduce_mean, axis_name="batch")(values)
        assert jnp.allclose(out, jnp.mean(values))

    def test_distributed_dot(self):
        x = random.normal(random.PRNGKey(0), (N, 4))
        y = random.normal(random.PRNGKey(1), (N, 4))
        out = pmap(distributed_dot, axis_name="batch")(x, y)
        assert jnp.allclose(out, jnp.vdot(x, y), rtol=1e-5)

    def test_sync_batch_stats_pytree(self):
        stats = {
            "mean": random.normal(random.PRNGKey(0), (N, 3)),
            "var": random.uniform(random.PRNGKey(1), (N, 3), minval=0.1, maxval=2.0),
        }
        synced = pmap(sync_batch_stats, axis_name="batch")(stats)
        assert jnp.allclose(synced["mean"][3], jnp.mean(stats["mean"], axis=0))

    def test_tree_all_reduce_sum_and_max(self):
        tree = {"a": jnp.arange(N, dtype=jnp.float32)}
        s = pmap(lambda t: tree_all_reduce(t, "sum"), axis_name="batch")(tree)
        m = pmap(lambda t: tree_all_reduce(t, "max"), axis_name="batch")(tree)
        assert jnp.allclose(s["a"], N * (N - 1) / 2) and jnp.allclose(m["a"], N - 1)

    def test_all_gather_and_reduce_scatter(self):
        x = jnp.arange(N * 2, dtype=jnp.float32).reshape(N, 2)
        gathered = pmap(lambda v: all_gather(v, tiled=True), axis_name="batch")(x)
        assert gathered.shape == (N, 2 * N)
        assert jnp.allclose(gathered[0], x.ravel())
        big = random.normal(random.PRNGKey(0), (N, N * 3))
        scattered = pmap(reduce_scatter, axis_name="batch")(big)
        assert scattered.shape == (N, 3)
        total = jnp.sum(big, axis=0)
        for i in range(N):
            assert jnp.allclose(scattered[i], total[3 * i : 3 * (i + 1)], atol=1e-5)

    def test_alltoall(self):
        x = jnp.arange(N * N, dtype=jnp.float32).reshape(N, N)  # device i holds row i
        out = pmap(alltoall, axis_name="batch")(x)
        assert jnp.allclose(out, x.T)  # transpose across devices

    def test_broadcast(self):
        x = jnp.arange(N, dtype=jnp.float32) * 10
        out = pmap(lambda v: broadcast(v, root_rank=3), axis_name="batch")(x)
        assert jnp.allclose(out, 30.0)

    def test_gradient_synchronization_clips_averaged_grad(self):
        grads = {"w": jnp.full((N, 4), 10.0)}
        out = pmap(lambda g: gradient_synchronization(g, clip_norm=1.0), axis_name="batch")(grads)
        assert jnp.allclose(jnp.linalg.norm(out["w"][0]), 1.0, atol=1e-5)

    def test_ring_all_reduce_equals_psum(self):
        x = random.normal(random.PRNGKey(0), (N, N * 5, 3))
        ring = pmap(lambda v: ring_all_reduce(v, "batch", num_devices=N), axis_name="batch")(x)
        expected = jnp.sum(x, axis=0)
        for i in range(N):
            assert jnp.allclose(ring[i], expected, atol=1e-4)

    def test_communication_volume(self):
        v = compute_communication_volume([(1024, 1024)], num_devices=8)
        assert jnp.isclose(v["bytes_sent_per_device_mb"], 2 * 7 / 8 * 4)


@multi
class TestSharding:
    def test_create_mesh(self):
        mesh = create_mesh((N,), ("data",))
        assert isinstance(mesh, Mesh) and mesh.shape == {"data": N}
        mesh2 = create_mesh((2, N // 2), ("data", "model"))
        assert mesh2.shape == {"data": 2, "model": N // 2}

    def test_shard_array_with_and_without_context(self):
        mesh = create_mesh((N,), ("batch",))
        x = jnp.ones((16, 4))
        sharded = shard_array(x, P("batch", None), mesh)
        assert sharded.sharding.spec == P("batch", None)
        assert len(sharded.addressable_shards) == N
        assert sharded.addressable_shards[0].data.shape == (16 // N, 4)
        with jax.set_mesh(mesh):
            assert shard_array(x, P(None, None)).sharding.spec == P(None, None)
        with pytest.raises(ValueError):
            shard_array(x, P(None, None))

    def test_check_sharding_compatibility(self):
        mesh = create_mesh((N,), ("batch",))
        assert check_sharding_compatibility(jnp.ones((16, 4)), P("batch", None), mesh)
        assert not check_sharding_compatibility(jnp.ones((N + 1, 4)), P("batch", None), mesh)

    def test_partition_params_by_regex(self):
        mesh = create_mesh((N,), ("model",))
        params = {
            "embeddings": jnp.ones((N * 4, 8)),
            "dense": jnp.ones((8, N * 2)),
            "bias": jnp.ones(N * 2),
        }
        rules = {r"embeddings": P("model", None), r"dense": P(None, "model")}
        specs = partition_specs(params, rules)
        assert specs["embeddings"] == P("model", None) and specs["bias"] == P()
        sharded = partition_params(params, rules, mesh)
        assert sharded["dense"].sharding.spec == P(None, "model")
        assert sharded["bias"].sharding.spec == P()
        summary = sharding_summary(sharded)
        assert "['dense']" in summary and "model" in summary["['dense']"]

    def test_fsdp_rules_shard_largest_axis(self):
        params = {"w": jnp.ones((4, N * 8)), "b": jnp.ones(3)}
        specs = fsdp_rules("data", min_size=16)(params)
        assert specs["w"] == P(None, "data") and specs["b"] == P()

    def test_transformer_rules_apply_to_stacked_layers(self):
        from jax_nsl.models.transformer import create_transformer

        mesh = create_mesh((N,), ("model",))
        params, _ = create_transformer(
            d_model=N * 2, num_heads=2, num_layers=2, vocab_size=8, max_seq_len=4
        )
        specs = partition_specs(params, create_transformer_partition_specs("model"))
        assert specs["layers"]["attention"]["query"] == P(None, None, "model")
        assert specs["layers"]["ffn"]["W2"] == P(None, "model", None)
        assert specs["layers"]["ln1"]["scale"] == P()
        sharded = partition_params(params, create_transformer_partition_specs("model"), mesh)
        mem = estimate_memory_per_device(params, mesh, specs)
        assert mem["memory_reduction_factor"] > 1.0
        assert sharded["layers"]["attention"]["query"].addressable_shards[0].data.shape == (
            2,
            N * 2,
            2,
        )

    def test_setup_model_parallelism_matches_single_device(self):
        mesh = create_mesh((2, N // 2), ("data", "model"))
        key = random.PRNGKey(0)
        x = random.normal(key, (8, 16))
        w = random.normal(key, (16, 8 * (N // 2)))
        fn = setup_model_parallelism(
            lambda x, w: jax.nn.relu(x @ w),
            mesh,
            in_specs=(P("data", None), P(None, "model")),
            out_specs=P("data", "model"),
        )
        out = fn(x, w)
        assert out.sharding.spec == P("data", "model")
        assert jnp.allclose(out, jax.nn.relu(x @ w), atol=1e-5)

    def test_shard_map_matmuls(self):
        mesh = create_mesh((N,), ("model",))
        x = random.normal(random.PRNGKey(0), (4, N * 2))
        w = random.normal(random.PRNGKey(1), (N * 2, N * 3))
        col = sharded_matmul_shard_map(mesh)(x, w)
        row = sharded_matmul_row_parallel(mesh)(x, w)
        assert jnp.allclose(col, x @ w, atol=1e-4)
        assert jnp.allclose(row, x @ w, atol=1e-4)

    def test_sharded_train_step_matches_single_device(self):
        mesh = create_mesh((N,), ("data",))
        params, forward_fn, _ = create_mlp([4, 8, 3], seed=0)
        init, update = sgd_optimizer(0.1)

        def loss_fn(params, batch):
            return cross_entropy_loss(forward_fn(params, batch["inputs"]), batch["labels"])

        batch = {
            "inputs": random.normal(random.PRNGKey(0), (16, 4)),
            "labels": random.randint(random.PRNGKey(1), (16,), 0, 3),
        }
        param_specs = jax.tree_util.tree_map(lambda _: P(), params)
        step = make_sharded_train_step(
            loss_fn, update, mesh, param_specs, {"inputs": P("data", None), "labels": P("data")}
        )
        new_state, loss = step(init(params), batch)
        ref = update(init(params), jax.grad(loss_fn)(params, batch))
        for a, b in zip(
            jax.tree_util.tree_leaves(new_state.params), jax.tree_util.tree_leaves(ref.params)
        ):
            assert jnp.allclose(a, b, atol=1e-5)
        assert jnp.allclose(loss, loss_fn(params, batch), atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__])
