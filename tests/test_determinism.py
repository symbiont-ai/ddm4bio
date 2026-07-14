"""Unit tests for the determinism / seeding utilities.

Verifies ``check_determinism`` on a seeded stochastic callable, that
``with_seed`` restores the global RNG state on exit, and that
``seed_everything`` yields reproducible numpy streams.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.config import seed_everything
from ddm4bio.datasets.synthetic import make_sparse_signal
from ddm4bio.utils.seeds import check_determinism, with_seed


def test_check_determinism_passes_for_seeded_callable():
    # The callable seeds itself internally, so two calls must agree exactly.
    result = check_determinism(lambda: make_sparse_signal(100, 5, seed=1).signal)
    assert result is True


def test_with_seed_restores_global_rng_state():
    # Establish a global state, then draw a "before" chunk to advance it.
    np.random.seed(123)
    np.random.rand(5)

    # A with_seed block reseeds and consumes randomness, then restores state.
    with with_seed(999):
        np.random.rand(10)
    after = np.random.rand(5)

    # Control: identical setup without ever entering the with_seed block.
    np.random.seed(123)
    np.random.rand(5)
    control_after = np.random.rand(5)

    assert np.array_equal(after, control_after)


def test_with_seed_inner_stream_is_seeded():
    # Inside the block the stream is governed by the block's seed.
    with with_seed(42):
        inside_a = np.random.rand(4)
    with with_seed(42):
        inside_b = np.random.rand(4)
    assert np.array_equal(inside_a, inside_b)


def test_seed_everything_is_reproducible():
    seed_everything()
    first = np.random.rand(5)
    seed_everything()
    second = np.random.rand(5)
    assert np.array_equal(first, second)


def test_seed_everything_returns_seed():
    assert seed_everything(1234) == 1234
