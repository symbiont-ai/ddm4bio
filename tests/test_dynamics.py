"""Ground-truth tests for the data-driven dynamics methods.

Each test drives a method against a synthetic fixture whose exact answer is
known: DMD must recover the system spectrum, SINDy must recover the true active
library terms on a clean linear system, and the Kalman filter must denoise a
noisy Lorenz trajectory. All tests are deterministic (fixed seeds) and run
offline.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.datasets.synthetic import make_linear_dynamics, make_lorenz
from ddm4bio.methods.dynamics import dmd, kalman_filter, sindy_fit
from ddm4bio.methods.validation import term_recovery


def _sort_complex(values: np.ndarray) -> np.ndarray:
    """Sort complex eigenvalues by (real, imag) for a stable comparison."""
    values = np.asarray(values)
    return values[np.lexsort((values.imag, values.real))]


def test_dmd_recovers_system_eigenvalues():
    # Discrete linear system with a complex-conjugate pair and a real mode,
    # all inside the unit circle so the trajectory stays bounded.
    eigs = np.array([0.95 + 0.1j, 0.95 - 0.1j, 0.8 + 0.0j])
    ld = make_linear_dynamics(eigs, n_steps=200, seed=1)

    # dmd uses the snapshot convention (n_features, n_time): states in rows.
    snapshots = ld.trajectory.T
    result = dmd(snapshots)

    true_eigs = _sort_complex(np.linalg.eigvals(ld.A))
    dmd_eigs = _sort_complex(result.eigenvalues)

    assert dmd_eigs.shape == true_eigs.shape
    assert np.allclose(dmd_eigs, true_eigs, atol=1e-6)


def test_sindy_recovers_active_terms():
    # Clean linear spiral system: dx0 = -0.2 x0 + x1, dx1 = -x0 - 0.2 x1.
    # The only active library terms are the linear monomials x0 and x1.
    a_mat = np.array([[-0.2, 1.0], [-1.0, -0.2]])
    t = np.linspace(0.0, 20.0, 4000)
    dt = t[1] - t[0]

    state = np.array([2.0, 0.0])
    traj = np.empty((t.size, 2), dtype=float)
    traj[0] = state
    for i in range(1, t.size):
        # Fourth-order Runge-Kutta step of a linear ODE (deterministic).
        k1 = a_mat @ state
        k2 = a_mat @ (state + 0.5 * dt * k1)
        k3 = a_mat @ (state + 0.5 * dt * k2)
        k4 = a_mat @ (state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[i] = state

    rng = np.random.default_rng(0)
    traj_noisy = traj + 1e-3 * rng.standard_normal(traj.shape)

    result = sindy_fit(traj_noisy, t, poly_degree=2, threshold=0.1)
    true_active = {"x0", "x1"}
    scores = term_recovery(true_active, result)

    assert scores["precision"] >= 0.8
    assert scores["recall"] >= 0.8


def test_kalman_filter_reduces_noise():
    n_steps = 2000
    noise_sd = 3.0

    clean = make_lorenz(t_max=20.0, n_steps=n_steps, noise=0.0)
    noisy = make_lorenz(t_max=20.0, n_steps=n_steps, noise=noise_sd, seed=42)

    clean_states = clean.states
    observations = noisy.states

    # Random-walk state model observed directly: F = H = I. The filter trades
    # a modest process covariance against the (known) measurement covariance.
    identity = np.eye(3)
    f_mat = identity
    h_mat = identity
    q_mat = 1.0 * identity
    r_mat = (noise_sd**2) * identity

    filtered = kalman_filter(observations, f_mat, h_mat, q_mat, r_mat)

    assert filtered.shape == clean_states.shape
    err_filtered = np.linalg.norm(filtered - clean_states)
    err_raw = np.linalg.norm(observations - clean_states)
    assert err_filtered < err_raw


def test_make_lorenz_shape_and_determinism():
    a = make_lorenz(t_max=10.0, n_steps=1000, noise=0.5, seed=7)
    b = make_lorenz(t_max=10.0, n_steps=1000, noise=0.5, seed=7)

    assert a.states.shape == (1000, 3)
    assert a.t.shape == (1000,)
    # Same seed and noise level -> bitwise-identical trajectory.
    assert np.array_equal(a.states, b.states)

    # A different noise seed changes the observations.
    c = make_lorenz(t_max=10.0, n_steps=1000, noise=0.5, seed=8)
    assert not np.array_equal(a.states, c.states)
