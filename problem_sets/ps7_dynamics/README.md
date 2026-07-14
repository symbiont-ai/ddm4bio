# PS7 -- Data-Driven Dynamics: DMD, SINDy, and the Kalman Filter

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation*, Chapter 15
(dynamic mode decomposition), and Chapters 20-21 (sparse identification of
nonlinear dynamics and data assimilation / Kalman filtering). Read these before
you start; the assignment asks you to implement the core ideas yourself rather
than call a turnkey library, so you should be comfortable with the snapshot
matrix, the sequential-thresholded-least-squares idea behind SINDy, and the
predict/update recursion of the Kalman filter.

This problem set is about **learning dynamics from data**: given a measured
trajectory, can you recover the operator that generated it, the sparse governing
equations behind it, or a clean state estimate from noisy observations? Each
tool answers a different version of that question, and each comes with its own
failure mode that you are expected to find and report.

## Data

Everything is **offline**. There are no downloads and no network calls. All
signals come from `ddm4bio.datasets.synthetic`, which returns dataclasses that
carry the *known ground truth* alongside the generated observations:

- `make_linear_dynamics(eigs, ...)` -- a real linear system with a prescribed
  spectrum, plus a generated trajectory (for DMD).
- `make_sir(beta, gamma, ...)` -- an SIR epidemic with known rate parameters
  (synthetic "case-count" style data for SINDy).
- `make_fitzhugh_nagumo(...)` -- a FitzHugh-Nagumo neuron trace used here as a
  stand-in *physiological signal* (for Kalman filtering).
- `make_lorenz(..., noise=...)` -- a Lorenz trajectory with optional additive
  observation noise (available if you want to explore the chaotic case).

Because the ground truth is known, you can score your recovered dynamics exactly
with `ddm4bio.methods.validation.term_recovery` and
`reconstruction_error`. Use the library methods
`ddm4bio.methods.dynamics.dmd`, `sindy_fit`, and `kalman_filter`, and run the QC
golden rule (`ddm4bio.qc.signals.qc_signals`) before analyzing any signal.

## What to implement

Fill in the function bodies in `student/ps7.py`. The imports, data loading, and
QC plumbing are already wired; you implement only the method logic where the
`# TODO` markers are. Keep the public function signatures exactly as given.

### Part A -- Method

1. **DMD.** Implement `run_dmd(snapshots, r=None, dt=1.0)`. Fit exact dynamic
   mode decomposition to a snapshot matrix (state variables in rows, time in
   columns), then read the dynamics off the discrete-time eigenvalues: report
   the spatial modes, the eigenvalues, the growth/decay rate
   `log|lambda| / dt`, and the oscillation frequency `angle(lambda) / dt`.
   Verify on a synthetic linear system that the recovered spectrum matches the
   true eigenvalues.
2. **SINDy.** Implement `sindy_terms(trajectory, t, poly_degree, threshold)` to
   recover the sparse governing equations of a synthetic system from its
   time-series, and confirm which library terms come back active. On a clean
   linear spiral the only true terms are the linear monomials.
3. **Kalman filter.** Implement `kalman_denoise(observations, process_var,
   meas_var)`: assimilate a noisy trajectory into a denoised state estimate
   using a matched random-walk model (`F = H = I`). You will apply it to a noisy
   Lorenz-style / physiological trajectory in Part B.

### Part B -- Application

4. **Epidemic dynamics.** Implement `fit_epidemic_dynamics(sir, threshold)`:
   fit SINDy to synthetic SIR case-count data and recover the governing terms
   (the bilinear infection term and the linear recovery term). Interpret what
   the recovered equation says about transmission and recovery.
5. **Physiological signal.** Implement `filter_physiological_signal(clean,
   noisy, ...)`: Kalman-filter a noisy physiological signal (a FitzHugh-Nagumo
   trace) and compare the filtered estimate against the raw signal, quantifying
   the improvement in L2 error.

### Part C -- Quality control (required)

6. **Test SINDy on ground truth first.** Implement
   `sindy_noise_sensitivity(noise_levels, ...)`: before touching any "real"
   data, run SINDy on a system whose answer you *know* and report term-recovery
   precision/recall as a function of observation noise. Identify the noise level
   at which recovery breaks down.
7. **Hold out the future.** Implement `dmd_forecast(snapshots, n_train, r=None)`:
   fit DMD on the leading `n_train` snapshots only, forecast the held-out future
   columns, and report the **out-of-sample** relative-L2 error separately from
   the in-sample reconstruction error. Never report only training error.

### Part D -- Interpretation & confidence

End your run by printing an interpretation block via
`ddm4bio.interpret.interpretation_block(...)`. State plainly:

- **Which dynamical terms you trust** and why (tie it to the term-recovery
  scores, not a hunch).
- **The noise level at which SINDy breaks** -- name the number from your Part C
  sweep.
- **Your forecast confidence bounds** -- how far ahead the DMD forecast stays
  trustworthy, and why a linear forecast should not be extrapolated into genuine
  chaos.

Be honest about limitations: everything here is synthetic ground truth, the
random-walk Kalman model is a deliberate simplification, and clean recovery on a
fixture does not guarantee clean recovery on field data.

## Running

```bash
python student/ps7.py            # runs the whole pipeline + prints QC and interpretation
pytest tests/test_ps7.py         # autograder (interfaces + ground-truth thresholds)
```

The reference solution lives in `solution/ps7_solution.py`. The autograder
imports the solution by default; for GitHub Classroom the import target is
swapped to your `student/ps7.py`.
