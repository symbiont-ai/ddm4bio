"""Synthetic ground-truth fixtures.

These generators are the correctness backbone of the course: each returns a
small dataclass that exposes the *known* ground truth alongside the generated
observations, so algorithms can be validated against exact answers.

Tier: synthetic (generated locally).
License: n/a (produced by this package).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MixedSources:
    """Ground truth for a blind-source-separation fixture.

    Attributes
    ----------
    sources: ``(n_sources, n_samples)`` true independent sources.
    mixing: ``(n_sources, n_sources)`` true mixing matrix.
    observations: ``(n_sources, n_samples)`` observed mixtures.
    """

    sources: np.ndarray
    mixing: np.ndarray
    observations: np.ndarray


@dataclass
class SIRTrajectory:
    """Ground truth for an SIR epidemic simulation.

    Attributes
    ----------
    t: ``(n_steps,)`` time grid.
    susceptible, infected, recovered: ``(n_steps,)`` S/I/R fractions.
    beta, gamma: true rate parameters.
    """

    t: np.ndarray
    susceptible: np.ndarray
    infected: np.ndarray
    recovered: np.ndarray
    beta: float
    gamma: float


@dataclass
class FitzHughNagumo:
    """Ground truth for a FitzHugh-Nagumo oscillator.

    Attributes
    ----------
    t: ``(n_steps,)`` time grid.
    v, w: ``(n_steps,)`` membrane and recovery variables.
    params: dict of the true model parameters.
    """

    t: np.ndarray
    v: np.ndarray
    w: np.ndarray
    params: dict


@dataclass
class SparseSignal:
    """Ground truth for a sparse-recovery fixture.

    Attributes
    ----------
    signal: ``(n,)`` full signal.
    support: ``(k,)`` indices of the nonzero coefficients.
    coeffs: ``(k,)`` values at the support.
    """

    signal: np.ndarray
    support: np.ndarray
    coeffs: np.ndarray


@dataclass
class LorenzTrajectory:
    """Ground truth for a Lorenz-system trajectory.

    Attributes
    ----------
    t: ``(n_steps,)`` uniform time grid.
    states: ``(n_steps, 3)`` state trajectory with columns ``[x, y, z]``.
    sigma, rho, beta: the true Lorenz parameters.
    """

    t: np.ndarray
    states: np.ndarray
    sigma: float
    rho: float
    beta: float


@dataclass
class LinearDynamics:
    """Ground truth for a linear dynamical system.

    Attributes
    ----------
    A: ``(d, d)`` system matrix.
    eigs: ``(d,)`` true eigenvalues.
    trajectory: ``(n_steps, d)`` generated state trajectory.
    """

    A: np.ndarray
    eigs: np.ndarray
    trajectory: np.ndarray


def _standardize(x: np.ndarray) -> np.ndarray:
    """Return ``x`` shifted/scaled to zero mean and unit variance."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    std = x.std()
    if std > 0:
        x = x / std
    return x


def make_mixed_sources(
    n_sources: int,
    n_samples: int,
    mixing: str = "random",
    seed: int | None = None,
) -> MixedSources:
    """Generate mixed independent sources for BSS/ICA validation.

    The sources are deliberately non-Gaussian and independent (a sine, a
    sawtooth ramp, a square wave, then Laplacian-noise sources for any extra
    components) so that ICA can identify and separate them. Each source is
    standardized to zero mean and unit variance, then linearly mixed by a
    well-conditioned random matrix.

    Parameters
    ----------
    n_sources: number of latent sources.
    n_samples: samples per source.
    mixing: mixing strategy, e.g. ``"random"``.
    seed: optional RNG seed for deterministic output.

    Returns
    -------
    MixedSources
        Dataclass exposing the known sources, mixing matrix, and observations.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n_samples, endpoint=False)

    shape_builders = [
        lambda: np.sin(2.0 * np.pi * 5.0 * t),
        lambda: 2.0 * (t * 7.0 - np.floor(t * 7.0)) - 1.0,
        lambda: np.sign(np.sin(2.0 * np.pi * 3.0 * t)),
    ]

    sources = np.empty((n_sources, n_samples), dtype=float)
    for i in range(n_sources):
        if i < len(shape_builders):
            raw = shape_builders[i]()
        else:
            raw = rng.laplace(0.0, 1.0, size=n_samples)
        sources[i] = _standardize(raw)

    if mixing == "random":
        mat = rng.standard_normal((n_sources, n_sources))
        # Nudge toward a well-conditioned matrix by strengthening the diagonal.
        mat = mat + n_sources * np.eye(n_sources)
    else:
        raise ValueError(f"unknown mixing strategy: {mixing!r}")

    observations = mat @ sources
    return MixedSources(sources=sources, mixing=mat, observations=observations)


def make_sir(
    beta: float,
    gamma: float,
    s0: float = 0.99,
    i0: float = 0.01,
    t_max: float = 100.0,
    n_steps: int = 1000,
) -> SIRTrajectory:
    """Simulate an SIR epidemic with known parameters.

    Parameters
    ----------
    beta: infection rate.
    gamma: recovery rate.
    s0, i0: initial susceptible/infected fractions.
    t_max, n_steps: integration horizon and resolution.

    Returns
    -------
    SIRTrajectory
        Dataclass exposing the trajectory and the true ``beta``/``gamma``.
    """
    from scipy.integrate import solve_ivp

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        s, i, _r = y
        ds = -beta * s * i
        di = beta * s * i - gamma * i
        dr = gamma * i
        return [ds, di, dr]

    r0 = 1.0 - s0 - i0
    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        [s0, i0, r0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    return SIRTrajectory(
        t=sol.t,
        susceptible=sol.y[0],
        infected=sol.y[1],
        recovered=sol.y[2],
        beta=beta,
        gamma=gamma,
    )


def make_fitzhugh_nagumo(
    a: float = 0.7,
    b: float = 0.8,
    tau: float = 12.5,
    i_ext: float = 0.5,
    t_max: float = 200.0,
    n_steps: int = 2000,
) -> FitzHughNagumo:
    """Simulate a FitzHugh-Nagumo neuron with known parameters.

    Parameters
    ----------
    a, b, tau, i_ext: FitzHugh-Nagumo model parameters.
    t_max, n_steps: integration horizon and resolution.

    Returns
    -------
    FitzHughNagumo
        Dataclass exposing the trajectory and the true parameter set.
    """
    from scipy.integrate import solve_ivp

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        v, w = y
        dv = v - v**3 / 3.0 - w + i_ext
        dw = (v + a - b * w) / tau
        return [dv, dw]

    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        [0.0, 0.0],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10,
    )
    params = {"a": a, "b": b, "tau": tau, "i_ext": i_ext}
    return FitzHughNagumo(t=sol.t, v=sol.y[0], w=sol.y[1], params=params)


def make_sparse_signal(
    n: int,
    k: int,
    seed: int | None = None,
) -> SparseSignal:
    """Generate a k-sparse signal with a known support.

    Parameters
    ----------
    n: signal length.
    k: number of nonzero coefficients.
    seed: optional RNG seed.

    Returns
    -------
    SparseSignal
        Dataclass exposing the signal, its support, and coefficient values.
    """
    rng = np.random.default_rng(seed)
    support = np.sort(rng.choice(n, size=k, replace=False))
    coeffs = rng.standard_normal(k)
    # Guard against a zero draw so every support entry is genuinely nonzero.
    zero = coeffs == 0.0
    if np.any(zero):
        coeffs[zero] = 1.0
    signal = np.zeros(n, dtype=float)
    signal[support] = coeffs
    return SparseSignal(signal=signal, support=support, coeffs=coeffs)


def undersample(
    x: np.ndarray,
    ratio: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly undersample a signal/array by a given ratio.

    Parameters
    ----------
    x: input array to undersample.
    ratio: fraction of entries to retain, in ``(0, 1]``.
    seed: optional RNG seed for deterministic sampling.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(samples, mask_indices)`` where ``samples`` are the retained values
        of ``x`` and ``mask_indices`` are the sorted indices that were
        retained, so a caller can reconstruct which entries were observed.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = x.size
    n_keep = max(1, int(round(ratio * n)))
    mask_indices = np.sort(rng.choice(n, size=n_keep, replace=False))
    samples = x.reshape(-1)[mask_indices]
    return samples, mask_indices


def make_linear_dynamics(
    eigs: np.ndarray,
    n_steps: int = 200,
    seed: int | None = None,
) -> LinearDynamics:
    """Generate a linear dynamical system with prescribed eigenvalues.

    A real system matrix ``A`` is built so that its eigenvalues equal the
    requested ``eigs``. Real eigenvalues sit on the diagonal of a real block
    matrix; complex-conjugate pairs are encoded as ``2x2`` real rotation-scale
    blocks so ``A`` stays real. The block matrix is then conjugated by a random
    orthogonal ``Q`` (which preserves eigenvalues) to remove structure. A
    trajectory is generated by iterating ``x_next = A @ x_prev``.

    Parameters
    ----------
    eigs: ``(d,)`` desired eigenvalues of the system matrix.
    n_steps: number of states in the generated trajectory.
    seed: optional RNG seed for deterministic output.

    Returns
    -------
    LinearDynamics
        Dataclass exposing the system matrix, its eigenvalues, and a
        generated state trajectory.
    """
    from scipy.linalg import qr

    rng = np.random.default_rng(seed)
    eigs = np.asarray(eigs)
    d = eigs.shape[0]

    # Build a real block-diagonal matrix realizing the requested spectrum.
    block = np.zeros((d, d), dtype=float)
    idx = 0
    used = np.zeros(d, dtype=bool)
    order = np.argsort(-np.abs(eigs.imag))  # pair up complex eigs first
    for j in order:
        if used[j]:
            continue
        val = eigs[j]
        if abs(val.imag) < 1e-12:
            block[idx, idx] = val.real
            used[j] = True
            idx += 1
        else:
            re, im = val.real, abs(val.imag)
            block[idx, idx] = re
            block[idx, idx + 1] = -im
            block[idx + 1, idx] = im
            block[idx + 1, idx + 1] = re
            used[j] = True
            # Mark the conjugate partner as consumed.
            for m in range(d):
                if not used[m] and abs(eigs[m] - np.conj(val)) < 1e-9:
                    used[m] = True
                    break
            idx += 2

    # Random orthogonal similarity transform preserves eigenvalues, keeps A real.
    q_mat, _ = qr(rng.standard_normal((d, d)))
    a_mat = q_mat @ block @ q_mat.T

    trajectory = np.empty((n_steps, d), dtype=float)
    state = rng.standard_normal(d)
    trajectory[0] = state
    for step in range(1, n_steps):
        state = a_mat @ state
        trajectory[step] = state

    return LinearDynamics(A=a_mat, eigs=eigs, trajectory=trajectory)


def make_lorenz(
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    x0: np.ndarray | None = None,
    t_max: float = 20.0,
    n_steps: int = 2000,
    noise: float = 0.0,
    seed: int | None = None,
) -> LorenzTrajectory:
    """Integrate the Lorenz system with known parameters.

    The Lorenz equations are ``dx = sigma (y - x)``,
    ``dy = x (rho - z) - y``, ``dz = x y - beta z``. The system is integrated on
    a uniform time grid; when ``noise > 0`` independent Gaussian noise is added
    to every state sample using a locally seeded generator.

    Parameters
    ----------
    sigma, rho, beta: true Lorenz parameters (defaults give the classic chaotic
        attractor).
    x0: ``(3,)`` initial condition; ``[1, 1, 1]`` when None.
    t_max, n_steps: integration horizon and number of uniform grid points.
    noise: standard deviation of additive Gaussian observation noise.
    seed: optional RNG seed for the noise draw (deterministic when set).

    Returns
    -------
    LorenzTrajectory
        Dataclass exposing the ``(n_steps, 3)`` states and the true parameters.
    """
    from scipy.integrate import solve_ivp

    def rhs(_t: float, y: np.ndarray) -> list[float]:
        x_c, y_c, z_c = y
        dx = sigma * (y_c - x_c)
        dy = x_c * (rho - z_c) - y_c
        dz = x_c * y_c - beta * z_c
        return [dx, dy, dz]

    init = np.array([1.0, 1.0, 1.0]) if x0 is None else np.asarray(x0, dtype=float)
    t_eval = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        init,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-9,
        atol=1e-12,
    )
    states = sol.y.T.copy()

    if noise > 0.0:
        rng = np.random.default_rng(seed)
        states = states + rng.normal(0.0, noise, size=states.shape)

    return LorenzTrajectory(
        t=sol.t,
        states=states,
        sigma=sigma,
        rho=rho,
        beta=beta,
    )
