# PS7 — The Predictability Horizon: How Far Can You Forecast Before Chaos Wins?

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/symbiont-ai/ddm4bio/blob/main/problem_sets/ps7_dynamics/ps7_colab.ipynb)

**Work in the browser:** click the badge to open this problem set in Google Colab — no local setup required. You can also work locally (see below).

**Reading:** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 7
(dynamical systems, DMD/SINDy), plus any introduction to chaos and the Lyapunov exponent.

Week 7 **fits** data-driven models (DMD, SINDy, Kalman) and reports a one-shot forecast
error at a fixed horizon. This problem set — the course **finale** — asks the question
*underneath* every forecast: **how far ahead can any model see before chaos wins?** For a
chaotic system the answer is a **finite** horizon set by the rate at which nearby
trajectories diverge (the largest Lyapunov exponent); for a non-chaotic system it is
unbounded.

> The systems, the integrators, the twin-trajectory plumbing, the fit-window heuristic, and
> the real forecaster are all **provided**; you build the six estimator functions. The core
> synthetic result needs *no model fit at all* — just twin-trajectory divergence and a slope.

Fill in the functions marked `# TODO` in `student/ps7.py`. Each is checked against a
**closed-form** answer, so a copied physical number cannot game the grader.

## Data

- **Chaotic Lorenz** (`make_lorenz`, ρ=28) — known largest Lyapunov exponent ~0.905.
- **Non-chaotic controls** — the FitzHugh–Nagumo limit cycle and a stable linear system
  (λ ~ 0 / < 0), for the finite-vs-unbounded contrast.
- **Real 1-D series** — JHU COVID daily incidence (`load_covid_incidence`) and MIT-BIH ECG
  (`load_ecg_series`), for the empirical horizon of an actual forecast.

Seed everything through `ddm4bio.seed_everything()` (called in `main`).

## Part A — The divergence rate and the horizon

- `separation_curve(traj_a, traj_b)` — per-timestep `‖b−a‖` of a twin pair.
- `ensemble_log_divergence(sep_curves, floor)` — average the **logs** of the separations.
- `divergence_rate(t, mean_log_sep, window)` — the least-squares slope over the growth
  window: the largest Lyapunov exponent.
- `finite_time_rate(t, mean_log_sep, half)` — the **local** slope (its plateau is λ).
- `forecast_horizon(lam, eps, tol)` — `T = (1/λ)·ln(tol/eps)`, or `inf` for `λ ≤ 0`.
- `empirical_forecast_horizon(y_true, y_pred, t, tol)` — the first time a real forecast's
  error exceeds `tol` (censored if it never does).

Recovering Lorenz's λ within a tolerance band (the finite-time estimator sits a little below
the true ~0.905) and separating chaotic (finite horizon ~18 time units) from non-chaotic
(unbounded) is the graded contrast.

## Part B — The empirical horizon of a real forecast

Apply the calibrated `empirical_forecast_horizon` to a Hankel-DMD forecast of real COVID
incidence: a finite **model-limited** horizon (~9–13 days) that shrinks toward ~1 day near
epidemic turning points. The ECG is an honest null (a linear model has no skill).

## Quality control & interpretation (required)

The provided `run_qc` reports the data source, length, and range before any result. The
provided `main` closes with a `ddm4bio.interpret.interpretation_block`: distinguish
**intrinsic** predictability (the Lyapunov limit) from **model-limited** predictability (a
particular forecaster's skill), at an honest confidence level, and name the real
limitations — the finite-time estimator's alignment-transient bias, the Gaussian/linearity
assumptions, and COVID non-stationarity.

## Files

- `student/ps7.py` — your working file; fill in every `# TODO`.
- `rubric.md` — how this problem set is graded.
- `ps7_colab.ipynb` — one-click Google Colab launcher (badge at the top).
- The reference solution and the autograder are provided through the course and
  run automatically by GitHub Classroom.

## Running

```bash
python student/ps7.py          # runs until the first unimplemented function
```

To work in the browser instead, click the Colab badge at the top of this file.
The autograder runs automatically when you push to GitHub Classroom.
