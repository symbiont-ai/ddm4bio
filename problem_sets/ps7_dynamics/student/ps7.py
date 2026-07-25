"""PS7: the predictability horizon (how far can you forecast before chaos wins?).

Module 7 fits data-driven models (DMD, SINDy, Kalman) and reports a one-shot forecast
error at a fixed horizon. This problem set -- the course finale -- asks the question
*underneath* every forecast: **how far ahead can any model see before chaos wins?** For a
chaotic system the answer is a finite horizon set by the rate at which nearby trajectories
diverge (the largest Lyapunov exponent); for a non-chaotic system the horizon is unbounded.

- Part A -- **measure the divergence rate on systems with KNOWN ground truth**: run
  twin-trajectory ("butterfly") experiments, estimate the divergence rate lambda, and
  derive the forecast horizon -- recovering Lorenz's known lambda and separating chaotic
  (finite horizon) from non-chaotic (unbounded) systems.
- Part B -- **apply the horizon idea to a real forecast**: measure the empirical horizon of
  a data-driven model on real COVID incidence, and distinguish *intrinsic* predictability
  (the system's Lyapunov limit) from *model-limited* predictability (this forecaster's skill).

Fill in every function body marked ``# TODO``. The systems, the integrators, the
twin-trajectory plumbing, the fit-window heuristic, the real forecaster, and ``main`` are
provided -- this problem set is about the four estimator functions, not the dynamics. The
autograder imports these functions by name, so keep the signatures exactly as given. Run with
``python ps7.py``; it stops at the first unimplemented function.
"""

from __future__ import annotations

import numpy as np

from ddm4bio.config import GLOBAL_SEED, seed_everything
from ddm4bio.interpret import interpretation_block

INF = float("inf")


# --------------------------------------------------------------------------- #
# Provided: systems, integrators, twin-trajectory plumbing (do not edit)       #
# --------------------------------------------------------------------------- #


def integrate_system(
    rhs, x0: np.ndarray, t_max: float, n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """(provided) Integrate ``dx/dt = rhs(t, x)`` from ``x0``; return ``(states, t)``.

    ``states`` has shape ``(n_steps, n_state)``. A thin deterministic ``solve_ivp`` wrapper
    so you can propagate perturbed initial conditions for any system.
    """
    from scipy.integrate import solve_ivp

    t = np.linspace(0.0, t_max, n_steps)
    sol = solve_ivp(rhs, [0.0, t_max], np.asarray(x0, dtype=float), t_eval=t, rtol=1e-9, atol=1e-9)
    return sol.y.T, t


def _fhn_rhs(t, s, a=0.7, b=0.8, tau=12.5, i_ext=0.5):
    v, w = s
    return [v - v**3 / 3.0 - w + i_ext, (v + a - b * w) / tau]


def lorenz_twin_ensemble(
    n_pairs: int = 120,
    eps: float = 1e-7,
    t_max: float = 8.0,
    n_steps: int = 800,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """(provided) Twin-trajectory separations for the chaotic Lorenz system.

    Samples ``n_pairs`` on-attractor start points, perturbs each by ``eps`` in a random
    direction, integrates both members, and returns ``(sep_curves, t)`` where ``sep_curves``
    has shape ``(n_pairs, n_steps)`` -- one separation curve per twin pair.
    """
    from ddm4bio.datasets.synthetic import make_lorenz

    rng = np.random.default_rng(seed)
    attractor = make_lorenz(t_max=40.0, n_steps=4000, seed=seed + 1).states
    starts = attractor[rng.integers(1000, 4000, size=n_pairs)]
    seps = np.empty((n_pairs, n_steps), dtype=float)
    for i, x0 in enumerate(starts):
        d = rng.standard_normal(3)
        d = eps * d / np.linalg.norm(d)
        a = make_lorenz(x0=x0.copy(), t_max=t_max, n_steps=n_steps, seed=0).states
        b = make_lorenz(x0=(x0 + d).copy(), t_max=t_max, n_steps=n_steps, seed=0).states
        seps[i] = np.linalg.norm(b - a, axis=1)
    return seps, np.linspace(0.0, t_max, n_steps)


def fhn_twin_ensemble(
    n_pairs: int = 40,
    eps: float = 1e-6,
    t_max: float = 60.0,
    n_steps: int = 1500,
    seed: int = GLOBAL_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """(provided) Twin-trajectory separations for the non-chaotic FitzHugh-Nagumo limit cycle."""
    from ddm4bio.datasets.synthetic import make_fitzhugh_nagumo

    fh = make_fitzhugh_nagumo(t_max=400.0, n_steps=4000)
    on_cycle = np.array([fh.v[-1], fh.w[-1]])
    rng = np.random.default_rng(seed)
    seps = np.empty((n_pairs, n_steps), dtype=float)
    for i in range(n_pairs):
        x0 = on_cycle + 0.5 * rng.standard_normal(2)
        d = rng.standard_normal(2)
        d = eps * d / np.linalg.norm(d)
        a, _ = integrate_system(_fhn_rhs, x0, t_max, n_steps)
        b, t = integrate_system(_fhn_rhs, x0 + d, t_max, n_steps)
        seps[i] = np.linalg.norm(b - a, axis=1)
    return seps, np.linspace(0.0, t_max, n_steps)


def linear_twin_ensemble(
    n_pairs: int = 30, eps: float = 1e-6, n_steps: int = 120, seed: int = GLOBAL_SEED
) -> tuple[np.ndarray, np.ndarray]:
    """(provided) Twin-trajectory separations for a stable linear system (per-step time axis)."""
    from ddm4bio.datasets.synthetic import make_linear_dynamics

    a_mat = make_linear_dynamics(eigs=np.array([0.95, 0.90]), n_steps=1, seed=seed).A
    rng = np.random.default_rng(seed)
    seps = np.empty((n_pairs, n_steps), dtype=float)
    for i in range(n_pairs):
        x0 = rng.standard_normal(2)
        xa, xb = x0.copy(), x0 + eps * rng.standard_normal(2)
        for k in range(n_steps):
            seps[i, k] = np.linalg.norm(xb - xa)
            xa, xb = a_mat @ xa, a_mat @ xb
    return seps, np.arange(n_steps, dtype=float)


def select_exponential_window(
    t: np.ndarray, mean_log_sep: np.ndarray, floor_frac: float = 0.12, sat_frac: float = 0.6
) -> tuple[int, int]:
    """(provided) A fit window above the initial transient and below saturation.

    Returns ``(start, end)`` indices bracketing the exponential-growth region: it ends where
    the mean-log separation first reaches ``sat_frac`` of its total rise, and starts a
    fraction ``floor_frac`` of the way into that span.
    """
    mls = np.asarray(mean_log_sep, dtype=float)
    lo, hi = mls.min(), mls.max()
    if hi <= lo:
        return 0, len(mls)
    level = lo + sat_frac * (hi - lo)
    end = int(np.searchsorted(mls, level))
    end = max(end, 3)
    start = int(floor_frac * end)
    return start, end


def hankel_dmd_forecast(
    series: np.ndarray, embed: int = 14, r: int = 6, n_train: int = 400, n_forecast: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """(provided) A time-delay (Hankel) DMD forecast of a 1-D series.

    Builds a delay-embedded matrix from ``series[:n_train]``, fits DMD, evolves it forward
    from the last training snapshot, and returns ``(y_true, y_pred)`` for the ``n_forecast``
    steps after ``n_train``.
    """
    from ddm4bio.methods.dynamics import dmd

    series = np.asarray(series, dtype=float)
    train = series[:n_train]
    ncol = len(train) - embed + 1
    hankel = np.array([train[i : i + ncol] for i in range(embed)])  # (embed, ncol)
    res = dmd(hankel, r=r)
    eig, modes = res.eigenvalues, res.modes
    x_last = hankel[:, -1]
    b_last = np.linalg.lstsq(modes, x_last, rcond=None)[0]
    steps = np.arange(1, n_forecast + 1)
    vander = eig[:, None] ** steps[None, :]
    recon = (modes @ (vander * b_last[:, None])).real  # (embed, n_forecast)
    y_pred = recon[-1]
    y_true = series[n_train : n_train + n_forecast]
    m = min(len(y_true), len(y_pred))
    return y_true[:m], y_pred[:m]


def load_covid_incidence() -> tuple[np.ndarray, str]:
    """(provided) Real JHU COVID-19 daily incidence, log1p-compressed. Returns ``(y, source)``."""
    from ddm4bio.datasets import get_dataset

    ds = get_dataset("jhu_covid")
    cases = np.asarray(ds.payload["cases"], dtype=float)
    daily = np.clip(np.diff(cases), 0.0, None)  # cumulative -> daily new (clip correction dips)
    return np.log1p(daily), ds.source


def load_ecg_series() -> tuple[np.ndarray, float, str]:
    """(provided) Real MIT-BIH ECG (first channel). Returns ``(signal, fs, source)``."""
    from ddm4bio.datasets import get_dataset

    ds = get_dataset("mitbih")
    sig = np.asarray(ds.payload["signal"], dtype=float)[:, 0]
    return sig, float(ds.payload["fs"]), ds.source


def run_qc(series: np.ndarray, label: str) -> None:
    """(provided) Print a QC line before any results."""
    s = np.asarray(series, dtype=float)
    print(
        f"QC [{label}]: length {s.size}, finite {np.isfinite(s).mean():.0%}, "
        f"range [{np.nanmin(s):.3g}, {np.nanmax(s):.3g}]."
    )


# --------------------------------------------------------------------------- #
# Part A -- The divergence rate and the horizon  (you implement)               #
# --------------------------------------------------------------------------- #


def ensemble_log_divergence(sep_curves: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    """Mean over pairs of ``ln(separation)`` at each time (the Lyapunov ensemble estimator).

    Given ``(n_pairs, n_time)`` separation curves, average the *logs* (not the raw
    separations), flooring at ``floor`` so a zero separation is finite. Returns ``(n_time,)``.
    """
    # TODO: take np.log(np.maximum(sep_curves, floor)) and average over the pair axis (axis=0).
    # Averaging the LOGS (not logging the average) is what makes this a Lyapunov estimator.
    raise NotImplementedError("Implement ensemble_log_divergence.")


def finite_time_rate(t: np.ndarray, mean_log_sep: np.ndarray, half: int = 5) -> np.ndarray:
    """Local (finite-difference) divergence rate at each time; its PLATEAU is ``lambda``.

    A centered slope with half-width ``half`` (clipped at the ends): its early rise exposes
    the alignment transient and its late fall exposes attractor saturation. Returns ``(n_time,)``.
    """
    # TODO: for each index i, use lo=max(0,i-half), hi=min(n-1,i+half) and compute the local
    # slope (mean_log_sep[hi] - mean_log_sep[lo]) / (t[hi] - t[lo]). Return the (n_time,) array.
    raise NotImplementedError("Implement finite_time_rate.")


def forecast_horizon(lam: float, eps: float, tol: float) -> float:
    """Predictability horizon ``T = (1/lam) * ln(tol / eps)``.

    The lead time at which an initial uncertainty ``eps`` grows to tolerance ``tol``. Returns
    ``inf`` (use ``INF``/``float('inf')``) when ``lam <= 0`` (error never diverges).
    """
    # TODO: if lam <= 0 return INF; else return (1/lam) * ln(tol/eps) as a float.
    raise NotImplementedError("Implement forecast_horizon.")


def empirical_forecast_horizon(
    y_true: np.ndarray, y_pred: np.ndarray, t: np.ndarray, tol: float
) -> float:
    """Model-limited horizon: the first time the forecast error exceeds ``tol``.

    Returns ``t[k]`` at the first ``k`` where ``|y_pred - y_true| > tol``; if the error never
    crosses, returns the censored endpoint ``t[-1]``.
    """
    # TODO: err = |y_pred - y_true|; find the first index where err > tol and return t there;
    # if there is no crossing, return t[-1] (censored).
    raise NotImplementedError("Implement empirical_forecast_horizon.")


# The twin-pair separation and the Lyapunov slope are *standard NumPy one-liners*, so the
# driver below computes them directly -- ``np.linalg.norm(b - a, axis=1)`` for the per-timestep
# separation, and ``np.polyfit(t[s:e], mls[s:e], 1)[0]`` for the least-squares divergence rate
# over the provided fit window. Nothing to implement here.


# --------------------------------------------------------------------------- #
# Provided: driver                                                             #
# --------------------------------------------------------------------------- #


def _lambda_of(ensemble_fn) -> tuple[float, np.ndarray, np.ndarray]:
    seps, t = ensemble_fn()
    mls = ensemble_log_divergence(seps)
    s, e = select_exponential_window(t, mls)
    lam = float(np.polyfit(t[s:e], mls[s:e], 1)[0])
    return lam, t, mls


def main() -> None:
    """Measure the predictability horizon on known systems, then on a real forecast."""
    seed_everything()

    print("== Part A: unit checks on closed-form fixtures ==")
    grid = np.linspace(0.0, 5.0, 50)
    a_demo = np.zeros((3, 3))
    b_demo = a_demo + [3, 4, 0]
    print(
        f"    separation of constant offset [3,4,0] = "
        f"{np.linalg.norm(b_demo - a_demo, axis=1)[0]:.1f} (=5)"
    )
    print(
        f"    divergence_rate of exp(0.9 t)         = "
        f"{np.polyfit(grid, np.log(1e-7) + 0.9 * grid, 1)[0]:.3f} (=0.9)"
    )
    print(
        f"    forecast_horizon(0.9, 1e-7, 1)        = {forecast_horizon(0.9, 1e-7, 1.0):.1f}  |  "
        f"non-chaotic (lam=-0.03) = {forecast_horizon(-0.03, 1e-7, 1.0)}"
    )

    print("\n== Part A: divergence rate on systems with KNOWN ground truth ==")
    lam_lorenz, t_l, _ = _lambda_of(lorenz_twin_ensemble)
    horizon_lorenz = forecast_horizon(lam_lorenz, eps=1e-7, tol=1.0)
    print(
        f"    chaotic  Lorenz : lambda={lam_lorenz:+.3f}  (true ~0.905)  "
        f"horizon(eps=1e-7,tol=1)={horizon_lorenz:.1f} time units"
    )
    lam_fhn, _, _ = _lambda_of(fhn_twin_ensemble)
    lam_lin, _, _ = _lambda_of(linear_twin_ensemble)
    h_fhn = forecast_horizon(lam_fhn, 1e-6, 1.0)
    h_lin = forecast_horizon(lam_lin, 1e-6, 1.0)
    print(f"    non-chaos FHN   : lambda={lam_fhn:+.3f}  horizon={h_fhn:.0f} (unbounded)")
    print(f"    non-chaos linear: rate/step={lam_lin:+.3f}  horizon={h_lin}")
    print("    -> chaos has a FINITE predictability horizon; the non-chaotic systems do not.")

    print("\n== Part B: the empirical horizon of a real forecast (COVID) ==")
    y, source = load_covid_incidence()
    print(f"[jhu_covid] source={source}")
    run_qc(y, "covid log-incidence")
    tol = 0.75  # log units (~2.1x multiplicative) -- where the forecast stops being useful
    horizons = np.array(
        [
            empirical_forecast_horizon(
                *hankel_dmd_forecast(y, embed=14, r=6, n_train=start, n_forecast=28),
                np.arange(1, 29),
                tol=tol,
            )
            for start in range(360, len(y) - 30, 45)
        ]
    )
    med, hlo, hhi = int(np.median(horizons)), int(horizons.min()), int(horizons.max())
    print(f"    Hankel-DMD forecast horizon (error > {tol} log units, {len(horizons)} windows):")
    print(f"      median ~{med} days, range {hlo}-{hhi} days -- it shrinks toward ~1 day near")
    print("      epidemic turning points, and is longest in stable stretches.")

    ecg, fs, ecg_source = load_ecg_series()
    yt, yp = hankel_dmd_forecast(ecg[:2000], embed=20, r=8, n_train=1400, n_forecast=200)
    h_ecg = empirical_forecast_horizon(
        yt, yp, np.arange(1, len(yt) + 1) / fs, tol=float(np.std(ecg[:2000]))
    )
    print(f"    [mitbih] source={ecg_source}: a linear model reverts to the mean, so its")
    print(f"      horizon {h_ecg * 1000:.0f} ms is just the quiet inter-beat gap (an honest null).")

    print("\n== Interpretation ==")
    block = interpretation_block(
        claim=(
            f"A chaotic system has a finite predictability horizon set by its largest Lyapunov "
            f"exponent (Lorenz lambda~{lam_lorenz:.2f} gives ~{horizon_lorenz:.0f} time units), "
            f"while non-chaotic systems (lambda<=0) are predictable indefinitely; a real forecast "
            f"adds a further model-limited horizon (~{med} days on COVID incidence)."
        ),
        limitations_list=[
            "The finite-time Lyapunov estimate has an alignment-transient bias, so the fit window "
            "matters; lambda is reported within a tolerance band, not to many digits.",
            "Intrinsic predictability (the Lyapunov limit) and model-limited predictability (a "
            "particular forecaster's skill) differ; the real horizon is the latter.",
            "The linear Hankel-DMD model has essentially no skill on the ECG, and COVID incidence "
            "is non-stationary, so the empirical horizon collapses exactly when a wave turns.",
        ],
    )
    print(block)


if __name__ == "__main__":
    main()
