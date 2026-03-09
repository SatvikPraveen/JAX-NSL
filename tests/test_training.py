# tests/test_training.py
"""Tests for src/training: losses, optimizers, and train_loop."""

import jax
import jax.numpy as jnp
import pytest
from jax import random, grad

from training.losses import (
    cross_entropy_loss, binary_cross_entropy, mse_loss,
    huber_loss, focal_loss, kl_divergence,
)
from training.optimizers import (
    sgd_optimizer, adam_optimizer, adamw_optimizer,
    clip_grads_by_global_norm, create_learning_rate_schedule,
)
from training.train_loop import (
    TrainState, create_train_state,
)


# ============================================================
# Losses
# ============================================================

class TestLosses:
    def test_cross_entropy_decreases_with_correct_prediction(self):
        # Perfect logits → very low loss
        logits_good = jnp.array([[10.0, -10.0, -10.0],
                                  [-10.0, 10.0, -10.0]])
        labels = jnp.array([0, 1])
        loss = cross_entropy_loss(logits_good, labels)
        assert float(loss) < 0.1

    def test_cross_entropy_shape_mean(self):
        logits = random.normal(random.PRNGKey(0), (8, 3))
        labels = random.randint(random.PRNGKey(1), (8,), 0, 3)
        loss = cross_entropy_loss(logits, labels, reduction="mean")
        assert loss.shape == ()

    def test_cross_entropy_shape_none(self):
        logits = random.normal(random.PRNGKey(0), (8, 3))
        labels = random.randint(random.PRNGKey(1), (8,), 0, 3)
        per_sample = cross_entropy_loss(logits, labels, reduction="none")
        assert per_sample.shape == (8,)

    def test_cross_entropy_label_smoothing(self):
        logits = jnp.ones((4, 3))
        labels = jnp.array([0, 1, 2, 0])
        loss_smooth = cross_entropy_loss(logits, labels, label_smoothing=0.1)
        loss_plain = cross_entropy_loss(logits, labels, label_smoothing=0.0)
        # Smoothing should make a uniform-logit case less extreme
        assert jnp.isfinite(loss_smooth)
        assert jnp.isfinite(loss_plain)

    def test_binary_cross_entropy_bounds(self):
        logits = jnp.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        labels = jnp.array([0.0, 0.0, 0.5, 1.0, 1.0])
        loss = binary_cross_entropy(logits, labels)
        assert jnp.isfinite(loss) and float(loss) >= 0.0

    def test_mse_loss(self):
        pred = jnp.array([1.0, 2.0, 3.0])
        target = jnp.array([1.0, 2.0, 3.0])
        assert jnp.allclose(mse_loss(pred, target), 0.0)

    def test_mse_loss_known_value(self):
        pred = jnp.array([0.0, 0.0])
        target = jnp.array([1.0, 1.0])
        assert jnp.allclose(mse_loss(pred, target), 1.0)

    def test_huber_loss_small_error(self):
        # For |e| <= delta, Huber == 0.5 * e^2
        pred = jnp.array([0.5])
        target = jnp.array([0.0])
        expected = 0.5 * 0.25
        assert jnp.allclose(huber_loss(pred, target, delta=1.0), expected, atol=1e-6)

    def test_focal_loss_finite(self):
        logits = random.normal(random.PRNGKey(0), (8, 3))
        labels = random.randint(random.PRNGKey(1), (8,), 0, 3)
        loss = focal_loss(logits, labels)
        assert jnp.isfinite(loss)

    def test_kl_divergence_self(self):
        # KL(p || p) == 0
        logits = jnp.array([1.0, 2.0, 3.0])
        kl = kl_divergence(logits, logits)
        assert jnp.allclose(kl, 0.0, atol=1e-5)


# ============================================================
# Optimizers
# ============================================================

class TestOptimizers:
    def _simple_params(self):
        return {"w": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array([0.5])}

    def test_sgd_decreases_loss(self):
        params = self._simple_params()
        init, update = sgd_optimizer(learning_rate=0.1)
        state = init(params)

        target = {"w": jnp.zeros(3), "b": jnp.zeros(1)}

        def loss_fn(p):
            return sum(jnp.sum((p[k] - target[k]) ** 2) for k in p)

        for _ in range(10):
            grads = grad(lambda p: loss_fn(p))(state.params)
            state = update(state, grads)

        assert float(loss_fn(state.params)) < float(loss_fn(params))

    def test_adam_step(self):
        params = self._simple_params()
        init, update = adam_optimizer(learning_rate=0.01)
        state = init(params)
        grads = jax.tree_util.tree_map(jnp.ones_like, params)
        new_state = update(state, grads)
        assert new_state.step == 1

    def test_adamw_weight_decay(self):
        params = {"w": jnp.array([1.0, 2.0])}
        init, update = adamw_optimizer(learning_rate=0.0, weight_decay=0.1)
        state = init(params)
        grads = {"w": jnp.zeros(2)}  # zero grads → only weight decay acts
        new_state = update(state, grads)
        # With zero lr and weight_decay, params should decrease toward zero
        assert jnp.all(jnp.abs(new_state.params["w"]) <= jnp.abs(params["w"]) + 1e-6)

    def test_clip_grads_by_global_norm(self):
        grads = {"w": jnp.array([[10.0, -10.0], [10.0, -10.0]])}
        clipped = clip_grads_by_global_norm(grads, max_norm=1.0)
        leaves = jax.tree_util.tree_leaves(clipped)
        global_norm = float(jnp.sqrt(sum(jnp.sum(l ** 2) for l in leaves)))
        assert global_norm <= 1.0 + 1e-5

    @pytest.mark.parametrize("schedule", ["cosine", "linear", "exponential"])
    def test_lr_schedule_positive(self, schedule):
        lr_fn = create_learning_rate_schedule(schedule, base_lr=0.1, total_steps=100)
        for step in [0, 50, 99]:
            assert float(lr_fn(step)) > 0.0


# ============================================================
# TrainState
# ============================================================

class TestTrainState:
    def test_create_train_state(self):
        params = {"w": jnp.ones(4)}
        key = random.PRNGKey(0)
        init, _ = sgd_optimizer(0.01)
        state = create_train_state(params, init, key)
        assert state.step == 0
        assert "w" in state.params

    def test_train_state_is_namedtuple(self):
        assert issubclass(TrainState, tuple)
