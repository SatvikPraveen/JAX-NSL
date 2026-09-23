# tests/test_training.py
"""Tests for jax_nsl.training: losses, optimisers, schedules, train loops."""

import jax
import jax.numpy as jnp
import pytest
from jax import grad, random

from jax_nsl.models.mlp import create_mlp
from jax_nsl.training.losses import (
    binary_cross_entropy,
    cross_entropy_loss,
    focal_loss,
    huber_loss,
    info_nce_loss,
    kl_divergence,
    mse_loss,
    quantile_loss,
)
from jax_nsl.training.optimizers import (
    adagrad_optimizer,
    adam_optimizer,
    adamw_optimizer,
    clip_grads_by_global_norm,
    create_learning_rate_schedule,
    ema_update,
    get_learning_rate,
    lion_optimizer,
    momentum_optimizer,
    rmsprop_optimizer,
    sgd_optimizer,
)
from jax_nsl.training.train_loop import (
    TrainState,
    accumulate_gradients,
    create_train_state,
    load_checkpoint,
    make_accumulating_train_step,
    make_eval_step,
    make_train_step,
    save_checkpoint,
    scaled_loss_and_grad,
    split_into_microbatches,
    train_epoch,
    training_loop,
    with_mixed_precision,
)

# ============================================================
# Losses
# ============================================================


class TestLosses:
    def test_cross_entropy_matches_jax_reference(self):
        logits = random.normal(random.PRNGKey(0), (8, 3))
        labels = random.randint(random.PRNGKey(1), (8,), 0, 3)
        ref = -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(8), labels])
        assert jnp.allclose(cross_entropy_loss(logits, labels), ref, atol=1e-6)
        assert cross_entropy_loss(logits, labels, reduction="none").shape == (8,)

    def test_cross_entropy_extreme_logits_and_smoothing(self):
        logits = jnp.array([[1000.0, -1000.0, -1000.0]])
        assert jnp.allclose(cross_entropy_loss(logits, jnp.array([0])), 0.0, atol=1e-6)
        smooth = cross_entropy_loss(logits, jnp.array([0]), label_smoothing=0.1)
        assert jnp.isfinite(smooth) and smooth > 0

    def test_cross_entropy_weights(self):
        logits = jnp.zeros((2, 2))
        labels = jnp.array([0, 1])
        w = jnp.array([1.0, 0.0])
        assert jnp.allclose(cross_entropy_loss(logits, labels, weights=w), jnp.log(2.0))

    def test_binary_cross_entropy_matches_reference(self):
        logits = jnp.array([-50.0, -1.0, 0.0, 1.0, 50.0])
        labels = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0])
        ref = -(labels * jax.nn.log_sigmoid(logits) + (1 - labels) * jax.nn.log_sigmoid(-logits))
        assert jnp.allclose(binary_cross_entropy(logits, labels, reduction="none"), ref, atol=1e-6)

    def test_mse_and_huber(self):
        assert jnp.allclose(mse_loss(jnp.array([0.0, 0.0]), jnp.array([1.0, 1.0])), 1.0)
        assert jnp.allclose(huber_loss(jnp.array([0.5]), jnp.array([0.0])), 0.125, atol=1e-6)
        assert jnp.allclose(huber_loss(jnp.array([3.0]), jnp.array([0.0]), delta=1.0), 2.5)

    def test_focal_loss_reduces_to_ce_when_gamma_zero(self):
        logits = random.normal(random.PRNGKey(0), (8, 3))
        labels = random.randint(random.PRNGKey(1), (8,), 0, 3)
        assert jnp.allclose(
            focal_loss(logits, labels, alpha=1.0, gamma=0.0),
            cross_entropy_loss(logits, labels),
            atol=1e-6,
        )
        confident = jnp.array([[10.0, 0.0, 0.0]])
        assert focal_loss(confident, jnp.array([0]), gamma=2.0) < cross_entropy_loss(
            confident, jnp.array([0])
        )

    def test_kl_divergence(self):
        logits = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(kl_divergence(logits, logits), 0.0, atol=1e-6)
        assert kl_divergence(logits, jnp.zeros(3)) > 0

    def test_quantile_loss_asymmetry(self):
        pred, target = jnp.array([0.0]), jnp.array([1.0])
        assert quantile_loss(pred, target, quantile=0.9) > quantile_loss(pred, target, quantile=0.1)

    def test_info_nce_is_low_for_matched_pairs(self):
        z = random.normal(random.PRNGKey(0), (16, 8))
        assert info_nce_loss(z, z, temperature=0.05) < info_nce_loss(z, z[::-1], temperature=0.05)


# ============================================================
# Optimisers
# ============================================================


def _quadratic_problem():
    target = {"w": jnp.array([1.0, -2.0, 0.5]), "b": jnp.array([0.3])}
    params = {"w": jnp.zeros(3), "b": jnp.zeros(1)}

    def loss(p):
        return sum(jnp.sum((p[k] - target[k]) ** 2) for k in p)

    return params, loss


class TestOptimizers:
    @pytest.mark.parametrize(
        "factory",
        [
            lambda: sgd_optimizer(0.1),
            lambda: momentum_optimizer(0.05, nesterov=True),
            lambda: adam_optimizer(0.1),
            lambda: adamw_optimizer(0.1, weight_decay=0.0),
            lambda: rmsprop_optimizer(0.05),
            lambda: adagrad_optimizer(0.5),
            lambda: lion_optimizer(0.02),
        ],
    )
    def test_all_optimizers_decrease_loss_under_jit(self, factory):
        params, loss = _quadratic_problem()
        init, update = factory()
        state = init(params)
        step = jax.jit(lambda s: update(s, grad(loss)(s.params)))
        for _ in range(100):
            state = step(state)
        assert float(loss(state.params)) < 0.05 * float(loss(params))
        assert int(state.step) == 100

    def test_adam_state_fields_and_bias_correction(self):
        params = {"w": jnp.array([1.0, 2.0, 3.0])}
        init, update = adam_optimizer(learning_rate=0.01)
        state = update(init(params), {"w": jnp.ones(3)})
        assert int(state.step) == 1
        # After one step with bias correction the update is exactly lr * sign(g).
        assert jnp.allclose(state.params["w"], params["w"] - 0.01, atol=1e-6)
        assert set(state._fields) == {"step", "params", "mu", "nu"}

    def test_adamw_decoupled_weight_decay(self):
        params = {"w": jnp.array([1.0, 2.0])}
        init, update = adamw_optimizer(learning_rate=0.1, weight_decay=0.5)
        state = update(init(params), {"w": jnp.zeros(2)})
        assert jnp.allclose(state.params["w"], params["w"] * (1 - 0.1 * 0.5))

    def test_schedule_as_learning_rate(self):
        params = {"w": jnp.array([1.0])}
        schedule = create_learning_rate_schedule("linear", base_lr=1.0, total_steps=2, final_lr=0.0)
        init, update = sgd_optimizer(schedule)
        state = update(init(params), {"w": jnp.array([1.0])})  # lr(0) = 1.0
        assert jnp.allclose(state.params["w"], 0.0)
        state = update(state, {"w": jnp.array([1.0])})  # lr(1) = 0.5
        assert jnp.allclose(state.params["w"], -0.5)

    def test_clip_grads_by_global_norm(self):
        clipped = clip_grads_by_global_norm({"w": jnp.array([[10.0, -10.0], [10.0, -10.0]])}, 1.0)
        assert float(jnp.linalg.norm(clipped["w"])) <= 1.0 + 1e-5

    def test_ema_update(self):
        ema = ema_update({"w": jnp.zeros(2)}, {"w": jnp.ones(2)}, decay=0.9)
        assert jnp.allclose(ema["w"], 0.1)


class TestSchedules:
    @pytest.mark.parametrize(
        "schedule", ["constant", "cosine", "linear", "exponential", "step", "warmup_cosine"]
    )
    def test_schedules_positive_and_jittable(self, schedule):
        lr_fn = create_learning_rate_schedule(
            schedule, base_lr=0.1, total_steps=100, warmup_steps=10, final_lr=0.01
        )
        for step in [0, 5, 50, 99, 150]:
            assert float(jax.jit(lr_fn)(jnp.int32(step))) >= 0.0
        assert jnp.allclose(get_learning_rate(lr_fn, 50), lr_fn(50))

    def test_warmup_cosine_shape(self):
        lr_fn = create_learning_rate_schedule(
            "warmup_cosine", base_lr=1.0, warmup_steps=10, total_steps=110, final_lr=0.0
        )
        assert (
            jnp.allclose(lr_fn(0), 0.0)
            and jnp.allclose(lr_fn(5), 0.5)
            and jnp.allclose(lr_fn(10), 1.0)
        )
        assert jnp.allclose(lr_fn(60), 0.5, atol=1e-6) and jnp.allclose(lr_fn(110), 0.0, atol=1e-6)


# ============================================================
# Training loop
# ============================================================


def _toy_problem(seed=0):
    key = random.PRNGKey(seed)
    x = random.normal(key, (64, 4))
    labels = (x[:, 0] + x[:, 1] > 0).astype(jnp.int32)
    params, forward_fn, _ = create_mlp([4, 16, 2], seed=seed)
    return params, forward_fn, {"inputs": x, "labels": labels}


class TestTrainLoop:
    def test_create_train_state(self):
        params = {"w": jnp.ones(4)}
        init, _ = sgd_optimizer(0.01)
        state = create_train_state(params, init, random.PRNGKey(0))
        assert int(state.step) == 0 and "w" in state.params
        assert issubclass(TrainState, tuple)

    def test_train_step_reduces_loss(self):
        params, forward_fn, batch = _toy_problem()
        init, update = adam_optimizer(1e-2)
        state = create_train_state(params, init, random.PRNGKey(0))
        step = make_train_step(forward_fn, cross_entropy_loss, update, max_grad_norm=5.0)
        state, first = step(state, batch)
        for _ in range(50):
            state, metrics = step(state, batch)
        assert float(metrics["loss"]) < float(first["loss"])
        assert float(metrics["accuracy"]) > 0.8
        assert int(state.step) == 51

    def test_eval_step_and_epoch_helpers(self):
        params, forward_fn, batch = _toy_problem()
        init, update = sgd_optimizer(0.1)
        state = create_train_state(params, init, random.PRNGKey(0))
        step = make_train_step(forward_fn, cross_entropy_loss, update)
        eval_step = make_eval_step(forward_fn, cross_entropy_loss)
        state, metrics = train_epoch(state, [batch, batch], step)
        assert set(metrics) == {"loss", "accuracy"}
        logs = []
        state = training_loop(
            state,
            lambda: [batch],
            lambda: [batch],
            step,
            eval_step,
            num_epochs=2,
            log_fn=lambda e, m: logs.append(m),
        )
        assert len(logs) == 2 and "val_loss" in logs[0] and "train_loss" in logs[0]

    def test_gradient_accumulation_equals_full_batch(self):
        params, forward_fn, batch = _toy_problem()

        def loss(p, b):
            return cross_entropy_loss(forward_fn(p, b["inputs"]), b["labels"])

        full_loss, full_grads = jax.value_and_grad(loss)(params, batch)
        acc_loss, acc_grads = accumulate_gradients(loss, params, split_into_microbatches(batch, 4))
        assert jnp.allclose(full_loss, acc_loss, atol=1e-6)
        for a, b in zip(
            jax.tree_util.tree_leaves(full_grads), jax.tree_util.tree_leaves(acc_grads)
        ):
            assert jnp.allclose(a, b, atol=1e-6)

    def test_accumulating_train_step_matches_plain_step(self):
        params, forward_fn, batch = _toy_problem()
        init, update = sgd_optimizer(0.1)
        s0 = create_train_state(params, init, random.PRNGKey(0))
        plain = make_train_step(forward_fn, cross_entropy_loss, update)
        accum = make_accumulating_train_step(
            forward_fn, cross_entropy_loss, update, num_microbatches=4
        )
        s1, _ = plain(s0, batch)
        s2, _ = accum(s0, batch)
        for a, b in zip(jax.tree_util.tree_leaves(s1.params), jax.tree_util.tree_leaves(s2.params)):
            assert jnp.allclose(a, b, atol=1e-5)

    def test_mixed_precision_wrapper(self):
        params, forward_fn, batch = _toy_problem()
        mp_forward = with_mixed_precision(forward_fn, jnp.bfloat16)
        out = mp_forward(params, batch["inputs"])
        assert out.dtype == jnp.float32
        assert jnp.allclose(out, forward_fn(params, batch["inputs"]), atol=5e-2)

    def test_scaled_loss_and_grad_unscales(self):
        f = lambda x: jnp.sum(x**2)  # noqa: E731
        x = jnp.array([1.0, 2.0])
        loss, g = scaled_loss_and_grad(f, 512.0)(x)
        assert jnp.allclose(loss, 5.0) and jnp.allclose(g, 2 * x)

    def test_checkpoint_roundtrip(self, tmp_path):
        params, forward_fn, batch = _toy_problem()
        init, update = adam_optimizer(1e-2)
        state = create_train_state(params, init, random.key(0))
        step = make_train_step(forward_fn, cross_entropy_loss, update)
        state, _ = step(state, batch)
        path = save_checkpoint(state, str(tmp_path), epoch=1)
        restored = load_checkpoint(path)
        assert int(restored.step) == 1
        assert jax.dtypes.issubdtype(restored.rng.dtype, jax.dtypes.prng_key)
        for a, b in zip(
            jax.tree_util.tree_leaves(state.params), jax.tree_util.tree_leaves(restored.params)
        ):
            assert jnp.array_equal(a, b)
        # Training can continue from the restored state.
        step(restored, batch)


if __name__ == "__main__":
    pytest.main([__file__])
