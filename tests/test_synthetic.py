"""Unit tests for the synthetic ground-truth fixtures.

Every fixture exposes a known ground truth; these tests check the reported
shapes/fields and the exact invariants (mixing identity, prescribed spectrum,
sparse support, conserved SIR mass). All tests are deterministic (fixed seeds)
and run fully offline.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.datasets.synthetic import (
    FitzHughNagumo,
    LinearDynamics,
    MixedSources,
    SIRTrajectory,
    SparseSignal,
    make_fitzhugh_nagumo,
    make_linear_dynamics,
    make_mixed_sources,
    make_sir,
    make_sparse_signal,
    undersample,
)


def _sort_complex(values: np.ndarray) -> np.ndarray:
    """Sort complex eigenvalues by (real, imag) for a stable comparison."""
    values = np.asarray(values)
    return values[np.lexsort((values.imag, values.real))]


def test_make_mixed_sources_shapes_and_mixing_identity():
    n_sources, n_samples = 3, 1000
    m = make_mixed_sources(n_sources, n_samples, seed=0)

    assert isinstance(m, MixedSources)
    assert m.sources.shape == (n_sources, n_samples)
    assert m.mixing.shape == (n_sources, n_sources)
    assert m.observations.shape == (n_sources, n_samples)

    # observations == mixing @ sources (the defining ground-truth relation).
    assert np.allclose(m.observations, m.mixing @ m.sources)


def test_make_mixed_sources_is_deterministic():
    a = make_mixed_sources(2, 500, seed=7)
    b = make_mixed_sources(2, 500, seed=7)
    assert np.array_equal(a.observations, b.observations)
    assert np.array_equal(a.mixing, b.mixing)


def test_make_linear_dynamics_recovers_spectrum():
    eigs = np.array([0.5 + 0.3j, 0.5 - 0.3j, -0.7 + 0.0j])
    ld = make_linear_dynamics(eigs, n_steps=50, seed=1)

    assert isinstance(ld, LinearDynamics)
    d = eigs.shape[0]
    assert ld.A.shape == (d, d)
    assert ld.trajectory.shape == (50, d)
    # A is real even though the spectrum has a complex-conjugate pair.
    assert np.isrealobj(ld.A)

    computed = np.linalg.eigvals(ld.A)
    assert np.allclose(_sort_complex(computed), _sort_complex(eigs), atol=1e-6)


def test_make_linear_dynamics_real_spectrum():
    eigs = np.array([0.9, 0.4, -0.2, 0.05])
    ld = make_linear_dynamics(eigs, n_steps=30, seed=2)
    computed = np.linalg.eigvals(ld.A)
    assert np.allclose(np.sort(computed.real), np.sort(eigs), atol=1e-6)


def test_make_sparse_signal_support_exact():
    n, k = 100, 5
    s = make_sparse_signal(n, k, seed=3)

    assert isinstance(s, SparseSignal)
    assert s.signal.shape == (n,)
    assert s.support.shape == (k,)
    assert s.coeffs.shape == (k,)

    # Exactly k nonzeros, located at the reported support.
    nz = np.nonzero(s.signal)[0]
    assert nz.size == k
    assert np.array_equal(np.sort(nz), np.sort(s.support))
    # Values at the support match the reported coefficients.
    assert np.allclose(s.signal[s.support], s.coeffs)


def test_make_sir_conserves_mass():
    sir = make_sir(0.3, 0.1)

    assert isinstance(sir, SIRTrajectory)
    n = sir.t.shape[0]
    assert sir.susceptible.shape == (n,)
    assert sir.infected.shape == (n,)
    assert sir.recovered.shape == (n,)
    assert sir.beta == 0.3
    assert sir.gamma == 0.1

    total = sir.susceptible + sir.infected + sir.recovered
    assert np.allclose(total, 1.0, atol=1e-3)


def test_make_fitzhugh_nagumo_fields():
    fhn = make_fitzhugh_nagumo(n_steps=500)

    assert isinstance(fhn, FitzHughNagumo)
    n = fhn.t.shape[0]
    assert n == 500
    assert fhn.v.shape == (n,)
    assert fhn.w.shape == (n,)
    assert isinstance(fhn.params, dict)
    for key in ("a", "b", "tau", "i_ext"):
        assert key in fhn.params


def test_undersample_retains_expected_count():
    x = np.arange(200, dtype=float)
    samples, mask_indices = undersample(x, 0.25, seed=4)

    assert samples.shape == mask_indices.shape
    assert samples.shape[0] == 50
    # Indices are sorted, unique, and in range.
    assert np.all(np.diff(mask_indices) > 0)
    assert mask_indices.min() >= 0
    assert mask_indices.max() < x.size
    # Retained samples equal x at the retained indices.
    assert np.array_equal(samples, x[mask_indices])
