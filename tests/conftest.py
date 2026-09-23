# tests/conftest.py
"""Shared pytest fixtures for the JAX-NSL test suite.

The XLA flag below must be set *before* JAX is imported anywhere: it makes
the CPU backend expose 8 virtual devices so the parallelism utilities
(pmap, sharding, collectives, shard_map) are exercised for real on a laptop
or in CI, rather than skipped.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import pytest  # noqa: E402
from jax import random  # noqa: E402


@pytest.fixture(scope="session")
def num_devices() -> int:
    return jax.device_count()


# ---------------------------------------------------------------------------
# PRNG keys
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def rng():
    """Root PRNG key for the entire test session."""
    return random.PRNGKey(42)


@pytest.fixture
def rng_pair(rng):
    """Two independent PRNG subkeys."""
    return random.split(rng)


# ---------------------------------------------------------------------------
# Small arrays
# ---------------------------------------------------------------------------

@pytest.fixture
def vec3(rng):
    return random.normal(rng, (3,))


@pytest.fixture
def mat3x3(rng):
    return random.normal(rng, (3, 3))


@pytest.fixture
def batch_vec(rng):
    return random.normal(rng, (8, 4))


@pytest.fixture
def batch_mat(rng):
    return random.normal(rng, (4, 3, 3))


# ---------------------------------------------------------------------------
# Tiny MLP parameter tree
# ---------------------------------------------------------------------------

@pytest.fixture
def mlp_params(rng):
    """Minimal MLP parameter dict: two layers (4->8->2)."""
    k1, k2 = random.split(rng)
    return {
        "layer1": {"W": random.normal(k1, (4, 8)) * 0.1, "b": jnp.zeros(8)},
        "layer2": {"W": random.normal(k2, (8, 2)) * 0.1, "b": jnp.zeros(2)},
    }


# ---------------------------------------------------------------------------
# Tiny datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_batch(rng):
    """Small regression mini-batch: (x: [16,4], y: [16])."""
    k1, k2 = random.split(rng)
    x = random.normal(k1, (16, 4))
    w_true = jnp.array([1.0, -2.0, 0.5, 3.0])
    y = x @ w_true + 0.01 * random.normal(k2, (16,))
    return {"x": x, "y": y}


@pytest.fixture
def classification_batch(rng):
    """Small classification mini-batch: (x: [16,4], labels: [16])."""
    k1, k2 = random.split(rng)
    return {"x": random.normal(k1, (16, 4)), "labels": random.randint(k2, (16,), 0, 3)}
