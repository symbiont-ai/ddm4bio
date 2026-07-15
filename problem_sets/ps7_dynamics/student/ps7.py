"""PS7 student template -- data-driven dynamics (DMD, SINDy, Kalman filter).

Fill in the bodies of the public functions below. The imports, data-loading, and
QC-call plumbing are already wired for you in ``main`` -- you only need to
implement the method logic where the ``# TODO`` markers are.

Array conventions (see ddm4bio.methods.dynamics):
* DMD uses snapshots of shape ``(n_features, n_time)`` (state in rows).
* SINDy uses trajectories of shape ``(n_time, n_state)`` (time in rows).
* The Kalman filter takes observations of shape ``(n_time, n_state)``.

Everything must run OFFLINE and deterministically (seeds fixed). Do not add
network calls, dataset downloads, or torch.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.datasets import get_dataset
from ddm4bio.datasets.synthetic import (
    make_fitzhugh_nagumo,
    make_linear_dynamics,
    make_sir,
)
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.dynamics import SINDyResult, dmd, kalman_filter, sindy_fit  # noqa: F401
from ddm4bio.methods.validation import reconstruction_error, term_recovery  # noqa: F401
from ddm4bio.qc.signals import qc_signals

SEED = 0

# Ground-truth active library terms for the reference systems.
LINEAR_SPIRAL_TERMS = {"x0", "x1"}
SIR_TERMS = {"x0 x1", "x1"}


def _integrate_linear(a_mat: np.ndarray, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Deterministic RK4 integration of the linear ODE ``dx/dt = A x``.

    This helper is provided so your ground-truth systems are reproducible.

    Parameters
    ----------
    a_mat : np.ndarray, shape (d, d)
        System matrix.
    x0 : array-like, shape (d,)
        Initial condition.
    t : np.ndarray, shape (n_time,)
        Uniform time grid.

    Returns
    -------
    np.ndarray, shape (n_time, d)
        State trajectory (time in rows, state in columns).
    """
    dt = float(t[1] - t[0])
    state = np.asarray(x0, dtype=float)
    traj = np.empty((t.size, state.size), dtype=float)
    traj[0] = state
    for i in range(1, t.size):
        k1 = a_mat @ state
        k2 = a_mat @ (state + 0.5 * dt * k1)
        k3 = a_mat @ (state + 0.5 * dt * k2)
        k4 = a_mat @ (state + dt * k3)
        state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        traj[i] = state
    return traj


# ---------------------------------------------------------------------------
# Part A -- Method
# ---------------------------------------------------------------------------
def run_dmd(snapshots: np.ndarray, r: int | None = None, dt: float = 1.0) -> dict:
    """Dynamic Mode Decomposition of a snapshot matrix.

    Fit exact DMD and derive the physically meaningful quantities from the
    discrete-time eigenvalues ``lambda``: the continuous growth/decay rate
    ``log|lambda| / dt`` and the oscillation frequency ``angle(lambda) / dt``.

    Parameters
    ----------
    snapshots : np.ndarray, shape (n_features, n_time)
        Snapshot matrix, state variables in ROWS and time samples in COLUMNS.
    r : int, optional
        SVD truncation rank; full rank when ``None``.
    dt : float, default 1.0
        Sample spacing, used to convert discrete eigenvalues to rates.

    Returns
    -------
    dict
        Keys ``eigenvalues``, ``modes``, ``amplitudes``, ``growth_rates``,
        ``frequencies``.
    """
    # TODO: call dmd(snapshots, r=r); pull out eigenvalues/modes/amplitudes.
    # TODO: compute growth_rates = log|lambda| / dt and
    #       frequencies = angle(lambda) / dt, then return the dict.
    raise NotImplementedError("Implement run_dmd")


def sindy_terms(
    trajectory: np.ndarray,
    t: np.ndarray,
    poly_degree: int = 2,
    threshold: float = 0.1,
) -> SINDyResult:
    """Recover the governing equations of a trajectory with SINDy.

    Parameters
    ----------
    trajectory : np.ndarray, shape (n_time, n_state)
        Measured state trajectory.
    t : np.ndarray
        Time stamps (1-D) or scalar sample spacing.
    poly_degree : int, default 2
        Maximum monomial degree of the candidate library.
    threshold : float, default 0.1
        STLSQ sparsity threshold.

    Returns
    -------
    SINDyResult
        Fitted SINDy result (coefficients, feature names, active-term sets).
    """
    # TODO: call sindy_fit with the given poly_degree and threshold and return
    #       its result (keep the (n_time, n_state) trajectory convention).
    raise NotImplementedError("Implement sindy_terms")


def kalman_denoise(
    observations: np.ndarray,
    process_var: float = 1.0,
    meas_var: float = 1.0,
) -> np.ndarray:
    """Denoise a directly-observed trajectory with a random-walk Kalman filter.

    Use the matched linear model where each state follows a random walk
    (``F = I``) observed directly (``H = I``), with ``Q = process_var * I`` and
    ``R = meas_var * I``.

    Parameters
    ----------
    observations : np.ndarray, shape (n_time, n_state)
        Noisy observations (time in rows).
    process_var : float, default 1.0
        Diagonal process-noise variance.
    meas_var : float, default 1.0
        Diagonal measurement-noise variance.

    Returns
    -------
    np.ndarray, shape (n_time, n_state)
        Filtered (posterior) state estimates.
    """
    obs = np.asarray(observations, dtype=float)
    if obs.ndim == 1:
        obs = obs[:, np.newaxis]
    # TODO: build F = H = identity(n_state), Q = process_var * I, R = meas_var * I,
    #       then call kalman_filter(obs, F, H, Q, R) and return the result.
    raise NotImplementedError("Implement kalman_denoise")


# ---------------------------------------------------------------------------
# Part C -- Quality control
# ---------------------------------------------------------------------------
def sindy_noise_sensitivity(
    noise_levels: list[float],
    seed: int = SEED,
    threshold: float = 0.1,
) -> list[dict]:
    """Ground-truth QC sweep: term recovery vs. observation noise.

    Generate a clean 2-D linear spiral (``dx0 = -0.2 x0 + x1``,
    ``dx1 = -x0 - 0.2 x1``) whose only active terms are ``LINEAR_SPIRAL_TERMS``,
    add Gaussian noise at each level, refit SINDy, and score term recovery.

    Parameters
    ----------
    noise_levels : list of float
        Standard deviations of additive Gaussian noise to sweep.
    seed : int, default ``SEED``
        Base RNG seed; use a distinct offset per level for reproducibility.
    threshold : float, default 0.1
        STLSQ sparsity threshold passed to SINDy.

    Returns
    -------
    list of dict
        One record per noise level with keys ``noise``, ``precision``,
        ``recall``, ``f1``.
    """
    a_mat = np.array([[-0.2, 1.0], [-1.0, -0.2]])
    t = np.linspace(0.0, 20.0, 4000)
    clean = _integrate_linear(a_mat, np.array([2.0, 0.0]), t)

    records: list[dict] = []
    for i, sd in enumerate(noise_levels):
        rng = np.random.default_rng(seed + i + 1)
        noisy = clean + float(sd) * rng.standard_normal(clean.shape)  # noqa: F841
        # TODO: fit SINDy on `noisy` via sindy_terms, score it with
        #       term_recovery(LINEAR_SPIRAL_TERMS, result), and append a record
        #       {"noise", "precision", "recall", "f1"}.
        raise NotImplementedError("Implement sindy_noise_sensitivity")
    return records


def dmd_forecast(snapshots: np.ndarray, n_train: int, r: int | None = None) -> dict:
    """Out-of-sample DMD forecast with a held-out future horizon.

    Fit DMD on the first ``n_train`` snapshots only, then propagate the modal
    amplitudes forward with the DMD eigenvalues (a Vandermonde time-evolution:
    column ``k`` is ``lambda**k``) to predict the held-out future columns.

    Parameters
    ----------
    snapshots : np.ndarray, shape (n_features, n_time)
        Full snapshot matrix (state in rows, time in columns).
    n_train : int
        Number of leading columns used for fitting; the remainder are held out.
    r : int, optional
        SVD truncation rank for the DMD fit.

    Returns
    -------
    dict
        Keys ``forecast``, ``reconstruction``, ``train_error``, ``test_error``
        (the last two are relative-L2 errors from ``reconstruction_error``).
    """
    x_arr = np.asarray(snapshots, dtype=float)
    n_time = x_arr.shape[1]
    if not 1 < n_train < n_time:
        raise ValueError("n_train must satisfy 1 < n_train < n_time")

    train = x_arr[:, :n_train]  # noqa: F841
    # TODO: fit DMD on `train`; build the (r, n_time) Vandermonde evolution
    #       lambda**k scaled by the amplitudes; reconstruct = (modes @ evolution).real.
    # TODO: split off forecast = reconstruction[:, n_train:], compute train_error
    #       and test_error with reconstruction_error(..., "rel_l2"), return the dict.
    raise NotImplementedError("Implement dmd_forecast")


# ---------------------------------------------------------------------------
# Part B -- Application
# ---------------------------------------------------------------------------
def fit_epidemic_dynamics(sir, threshold: float = 0.05) -> dict:
    """Recover SIR governing terms from synthetic case-count data with SINDy.

    Use the (S, I) sub-state: with ``dS = -beta S I`` and
    ``dI = beta S I - gamma I`` the active terms are ``SIR_TERMS`` = the bilinear
    ``x0 x1`` and the linear ``x1``. R is dropped to avoid the ``S+I+R``
    collinearity.

    Parameters
    ----------
    sir : SIRTrajectory
        Ground-truth SIR trajectory from ``make_sir``.
    threshold : float, default 0.05
        STLSQ sparsity threshold.

    Returns
    -------
    dict
        Keys ``result`` (the ``SINDyResult``), ``scores`` (term_recovery dict),
        ``true_terms`` (the ground-truth active-term set).
    """
    state = np.column_stack([sir.susceptible, sir.infected])  # noqa: F841
    # TODO: fit SINDy on `state` (time grid sir.t) via sindy_terms; score with
    #       term_recovery(SIR_TERMS, result); return the result/scores/true_terms.
    raise NotImplementedError("Implement fit_epidemic_dynamics")


def filter_physiological_signal(
    clean: np.ndarray,
    noisy: np.ndarray,
    process_var: float = 0.05,
    meas_var: float = 0.25,
) -> dict:
    """Kalman-filter a noisy physiological signal and compare to the raw signal.

    Apply the filter to ``noisy`` and compare the filtered estimate against the
    known ``clean`` signal, reporting L2 error for both raw and filtered.

    Parameters
    ----------
    clean : np.ndarray, shape (n_time, n_state)
        Ground-truth clean signal.
    noisy : np.ndarray, shape (n_time, n_state)
        Noisy observations of the same signal.
    process_var : float, default 0.05
        Random-walk process variance for the filter.
    meas_var : float, default 0.25
        Measurement-noise variance for the filter.

    Returns
    -------
    dict
        Keys ``filtered``, ``error_raw``, ``error_filtered``, ``improved``.
    """
    clean_arr = np.asarray(clean, dtype=float)  # noqa: F841
    # TODO: filter `noisy` with kalman_denoise(process_var, meas_var); compute
    #       error_raw and error_filtered as L2 norms vs clean_arr; set
    #       improved = error_filtered < error_raw; return the dict.
    raise NotImplementedError("Implement filter_physiological_signal")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    """Run the full PS7 pipeline and print QC + interpretation output."""
    rng = np.random.default_rng(SEED)

    # --- Part A: DMD on a synthetic linear spatio-temporal system ------------
    eigs = np.array([0.97 + 0.08j, 0.97 - 0.08j, 0.85 + 0.0j])
    linear = make_linear_dynamics(eigs, n_steps=200, seed=3)
    snapshots = linear.trajectory.T
    dmd_out = run_dmd(snapshots)
    print("=== Part A: DMD ===")
    print("DMD |eig|       :", np.round(np.sort(np.abs(dmd_out["eigenvalues"])), 4))

    # --- Part A: SINDy recovers the governing terms --------------------------
    a_mat = np.array([[-0.2, 1.0], [-1.0, -0.2]])
    t = np.linspace(0.0, 20.0, 4000)
    clean_spiral = _integrate_linear(a_mat, np.array([2.0, 0.0]), t)
    spiral_noisy = clean_spiral + 1e-3 * rng.standard_normal(clean_spiral.shape)
    sindy_result = sindy_terms(spiral_noisy, t, poly_degree=2, threshold=0.1)
    spiral_scores = term_recovery(LINEAR_SPIRAL_TERMS, sindy_result)
    print("\n=== Part A: SINDy ===")
    print("recovered terms :", sorted(sindy_result.active_terms))
    print(
        "precision/recall:",
        round(spiral_scores["precision"], 3),
        "/",
        round(spiral_scores["recall"], 3),
    )

    # --- Part C: QC BEFORE real data (noise sensitivity) ---------------------
    noise_levels = [0.0, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1]
    sweep = sindy_noise_sensitivity(noise_levels, seed=SEED, threshold=0.1)
    print("\n=== Part C: SINDy noise sensitivity (ground truth) ===")
    for record in sweep:
        print(f"noise={record['noise']:<6} P={record['precision']:.2f} R={record['recall']:.2f}")

    # --- Part C: held-out DMD forecast ---------------------------------------
    forecast_out = dmd_forecast(snapshots, n_train=150)
    print("\n=== Part C: DMD held-out forecast ===")
    print("train rel-L2    :", f"{forecast_out['train_error']:.2e}")
    print("test  rel-L2    :", f"{forecast_out['test_error']:.2e}")

    # --- Part B: epidemic dynamics from case counts --------------------------
    sir = make_sir(beta=0.6, gamma=0.2, t_max=100.0, n_steps=1000)
    epi = fit_epidemic_dynamics(sir, threshold=0.05)
    print("\n=== Part B: SIR governing equations (SINDy) ===")
    print("recovered terms :", sorted(epi["result"].active_terms))
    print(
        "precision/recall:",
        round(epi["scores"]["precision"], 3),
        "/",
        round(epi["scores"]["recall"], 3),
    )

    # --- Part B: filter a noisy physiological signal -------------------------
    fhn = make_fitzhugh_nagumo(t_max=200.0, n_steps=2000)
    clean_sig = np.column_stack([fhn.v, fhn.w])
    noise_sd = 0.5
    noisy_sig = clean_sig + noise_sd * rng.standard_normal(clean_sig.shape)

    # QC golden rule: report signal quality before analysis.
    qc = qc_signals(noisy_sig.T, fs=float(fhn.t.size) / 200.0, reference=clean_sig.T)
    print("\n=== Part B: physiological-signal QC ===")
    print(qc.render())

    phys = filter_physiological_signal(clean_sig, noisy_sig, process_var=0.05, meas_var=0.25)
    print("\n=== Part B: Kalman filtering ===")
    print("raw   L2 error  :", round(phys["error_raw"], 3))
    print("filt  L2 error  :", round(phys["error_filtered"], 3))
    print("filter improved :", phys["improved"])

    # --- Part B (real data): apply DMD to a REAL epidemic curve --------------
    # After validating DMD on a synthetic linear system, apply your dmd_forecast
    # to a real epidemic curve loaded through the data layer. get_dataset returns
    # the archived JHU CSSE COVID-19 series when reachable, else a deterministic
    # synthetic fallback with the SAME (date, cases) shape -- this block runs
    # identically either way, and prints the source so you know which you got.
    covid = get_dataset("jhu_covid")
    print("\n=== Part B (real): epidemic curve via DMD ===")
    print("data source     :", covid.source)
    print("provenance      :", covid.provenance)
    cases = np.asarray(covid.payload["cases"], dtype=float)
    incidence = np.clip(np.diff(cases), 0.0, None)  # daily new cases
    smoothed = np.convolve(incidence, np.ones(7) / 7.0, mode="valid")
    log_inc = np.log1p(smoothed)
    onset = int(np.argmax(smoothed > 0.01 * smoothed.max()))
    early = log_inc[onset:onset + min(60, log_inc.size - onset)]
    n_delays = 10
    cols = early.size - n_delays + 1
    covid_snaps = np.stack([early[i:i + cols] for i in range(n_delays)])
    epi = dmd_forecast(covid_snaps, n_train=int(0.7 * covid_snaps.shape[1]))
    print("train    rel-L2 :", f"{epi['train_error']:.3f}")
    print("held-out rel-L2 :", f"{epi['test_error']:.3f}  (short-horizon forecast)")

    # --- Part B (real data): Kalman-filter a REAL physiological signal -------
    # A single MIT-BIH ECG lead (real via wfdb, else a deterministic ECG-like
    # fallback with the same payload shape). There is no clean ground truth, so we
    # report the drop in sample-to-sample roughness (a reference-free noise proxy)
    # rather than an error against a known signal.
    ecg = get_dataset("mitbih")
    print("\n=== Part B (real): ECG via Kalman filter ===")
    print("data source     :", ecg.source)
    print("provenance      :", ecg.provenance)
    ecg_lead = np.asarray(ecg.payload["signal"], dtype=float)[:1500, :1]
    ecg_qc = qc_signals(ecg_lead.T, fs=float(ecg.payload["fs"]))
    print(ecg_qc.render())
    ecg_filtered = kalman_denoise(ecg_lead, process_var=0.02, meas_var=0.5)
    raw_rough = float(np.mean(np.abs(np.diff(ecg_lead, axis=0))))
    filt_rough = float(np.mean(np.abs(np.diff(ecg_filtered, axis=0))))
    print("raw  roughness  :", round(raw_rough, 4))
    print("filt roughness  :", round(filt_rough, 4))
    print("roughness drop  :", f"{raw_rough / filt_rough:.1f}x")

    # --- Part D: honest interpretation block ---------------------------------
    print("\n=== Part D: Interpretation ===")
    block = interpretation_block(
        claim="TODO: state your headline claim about DMD/SINDy/Kalman results",
        confidence="moderate",
        limitations_list=["TODO: name the real limitations you observed"],
        evidence="TODO: cite the numbers that justify your confidence",
    )
    print(block)


if __name__ == "__main__":
    main()
