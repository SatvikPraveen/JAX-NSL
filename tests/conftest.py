# tests/conftest.py
"""Shared pytest fixtures for the JAX-NSL test suite."""

import jax
import jax.numpy as jnp
from jax import random
import pytest


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
    """Float32 vector of length 3."""
    return random.normal(rng, (3,))


@pytest.fixture
def mat3x3(rng):
    """Float32 3×3 matrix."""
    return random.normal(rng, (3, 3))


@pytest.fixture
def batch_vec(rng):
    """Batch of 8 vectors of length 4."""
    return random.normal(rng, (8, 4))


@pytest.fixture
def batch_mat(rng):
    """Batch of 4 matrices of shape 3×3."""
    return random.normal(rng, (4, 3, 3))


# ---------------------------------------------------------------------------
# Tiny MLP parameter tree
# ---------------------------------------------------------------------------

@pytest.fixture
def mlp_params(rng):
    """Minimal MLP parameter dict: two layers (4→8→2)."""
    k1, k2, k3, k4 = random.split(rng, 4)
    return {
        "layer1": {
            "W": random.normal(k1, (4, 8)) * 0.1,
            "b": jnp.zeros(8),
        },
        "layer2": {
            "W": random.normal(k2, (8, 2)) * 0.1,
            "b": jnp.zeros(2),
        },
    }


# ---------------------------------------------------------------------------
# Tiny dataset
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
    x = random.normal(k1, (16, 4))
    labels = random.randint(k2, (16,), 0, 3)
    return {"x": x, "labels": labels}
