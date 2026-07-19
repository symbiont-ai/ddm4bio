"""ddm4bio capstone reference pipeline: warfarin PK/PD -> a dosing policy.

A designed, multi-step hybrid that carries REAL warfarin data from raw
measurements to a defensible dosing policy:

    load -> quality control (BEFORE results)
         -> fit a one-compartment PK model        (model-driven; validate t1/2)
         -> characterize the PD effect endpoint    (dose -> exposure -> effect)
         -> calibrate a dosing environment from the fitted clearance spread (data-driven)
         -> learn a dosing policy with RL           (value iteration + Q-learning)
         -> validate: policy holds effect in-range vs a fixed dose
         -> interpret with named limitations

The two paradigms are braided: a mechanistic PK model *fit from data* becomes the
environment a data-driven policy is learned on. It runs end-to-end offline -- if
the real warfarin fetch is unavailable, the loader returns a synthetic PK/PD table
with the same schema, so the pipeline behaves identically (the provenance line
says which one you got).

Run:  python src/pipeline.py   (or ``make run``)
"""

from __future__ import annotations

import numpy as np

from ddm4bio.datasets import get_dataset
from ddm4bio.interpret import interpretation_block
from ddm4bio.methods.control import PKDosingEnv, policy_value, q_learning, value_iteration

# Nominal dosing interval used to turn a continuous elimination rate (per hour)
# into a per-step clearance factor for the discrete dosing environment.
DOSING_INTERVAL_H = 24.0


def load_data():
    """Load the warfarin PK/PD dataset and print its provenance."""
    ds = get_dataset("warfarin")
    print(f"[load] warfarin source={ds.source}")
    print(f"[load] {ds.provenance}\n")
    return ds.payload


def quality_control(df) -> None:
    """Inspect the data BEFORE any modeling (non-negotiable #1: QC first)."""
    n_subj = df["id"].nunique()
    n_cp = int((df["dvid"] == "cp").sum())
    n_pca = int((df["dvid"] == "pca").sum())
    missing = int(df["dv"].isna().sum())
    doses = df.loc[df["amt"] > 0, "amt"]
    print("[QC] before any modeling:")
    print(f"[QC]   subjects={n_subj}  concentration obs={n_cp}  effect obs={n_pca}")
    print(f"[QC]   missing dv={missing}  dose range={doses.min():.0f}-{doses.max():.0f} mg")
    print(f"[QC]   weight range={df['wt'].min():.0f}-{df['wt'].max():.0f} kg\n")
    if n_subj < 10 or n_cp == 0 or n_pca == 0:
        raise ValueError("QC gate failed: too few subjects or a missing channel.")


def _oral_pk(t, ka, ke, amplitude):
    """One-compartment oral model (dose-normalized concentration)."""
    return amplitude * (np.exp(-ke * t) - np.exp(-ka * t))


def fit_pk(df) -> np.ndarray:
    """Group A / model-driven: fit a one-compartment PK model per subject.

    Returns the per-subject elimination rate ``ke`` (1/h). Validation: the median
    elimination half-life must land in warfarin's known ~1-2.5 day range.
    """
    from scipy.optimize import curve_fit

    cp = df[(df["dvid"] == "cp") & (df["evid"] == 0) & (df["time"] > 0)]
    dose = df[df["amt"] > 0].set_index("id")["amt"]
    kes = []
    for sid, g in cp.groupby("id"):
        if len(g) < 5:
            continue
        y = g["dv"].to_numpy() / float(dose[sid])
        try:
            params, _ = curve_fit(
                _oral_pk,
                g["time"].to_numpy(),
                y,
                p0=[0.5, 0.02, 0.02],
                bounds=([1e-3, 1e-4, 1e-4], [5.0, 1.0, 1.0]),
                maxfev=20000,
            )
            kes.append(params[1])
        except RuntimeError:
            continue
    kes = np.asarray(kes)
    t_half_h = float(np.log(2) / np.median(kes))
    print("[Group A: PK] one-compartment oral fit")
    print(f"[Group A: PK]   n={kes.size}  half-life={t_half_h:.1f} h = {t_half_h / 24:.2f} d")
    print(f"[Group A: PK]   clearance variability CV(ke)={kes.std() / kes.mean():.0%}\n")
    if not (20.0 <= t_half_h <= 80.0):
        raise ValueError(f"PK validation failed: t1/2={t_half_h:.1f} h outside warfarin's range.")
    return kes


def characterize_pd(df) -> float:
    """The PD effect endpoint: does more drug exposure produce a deeper effect?

    Correlates each subject's peak concentration (exposure) with the depth of
    their anticoagulation effect (baseline minus PCA nadir). A positive
    correlation is the dose -> exposure -> effect chain that justifies dosing to a
    therapeutic *effect* target. Returns the Pearson correlation.
    """
    cp = df[(df["dvid"] == "cp") & (df["evid"] == 0)]
    pca = df[df["dvid"] == "pca"]
    exposure, effect = [], []
    for sid in df["id"].unique():
        c = cp[cp["id"] == sid]["dv"]
        p = pca[pca["id"] == sid]["dv"]
        if c.empty or p.empty:
            continue
        exposure.append(float(c.max()))
        effect.append(100.0 - float(p.min()))  # depth below a ~100% baseline
    r = float(np.corrcoef(exposure, effect)[0, 1])
    print("[PD] effect endpoint (anticoagulation)")
    print(f"[PD]   corr(peak exposure, effect depth) = {r:+.2f}  (dose -> exposure -> effect)\n")
    return r


def calibrate_environment(kes: np.ndarray) -> PKDosingEnv:
    """Turn the fitted clearance spread into the dosing environment's variability.

    Each subject's elimination rate becomes a per-interval clearance factor
    ``exp(-ke * interval)``; three percentiles of that spread become the
    environment's stochastic clearance -- so the control problem inherits *real*
    inter-patient variability. Concentration is normalized to the therapeutic
    target (window around 1.0); the dynamics come from the warfarin fit.
    """
    factors = np.exp(-np.percentile(kes, [75, 50, 25]) * DOSING_INTERVAL_H)
    decays = tuple(np.sort(factors))
    print(f"[calibrate] per-interval clearance from real ke = {np.round(decays, 3).tolist()}\n")
    return PKDosingEnv(decays=decays, decay_probs=(0.25, 0.50, 0.25))


def learn_policy(env: PKDosingEnv):
    """The hybrid: value iteration (ground truth) + Q-learning (model-free) on the
    calibrated environment. Returns the learned and optimal policies."""
    v_star, pi_star = value_iteration(env.transition_matrix, env.expected_reward, env.gamma)
    q_table = q_learning(env, seed=0)
    pi_q = q_table.argmax(axis=1)
    v_q = policy_value(pi_q, env.transition_matrix, env.expected_reward, env.gamma)
    frac_opt = float(np.mean(v_q >= v_star - 1e-6))
    print("[Group B: RL] value iteration + Q-learning")
    print(f"[Group B: RL]   Q-learning reaches the optimal value on {frac_opt:.0%} of states\n")
    return pi_q, pi_star, frac_opt


def _time_in_range(env: PKDosingEnv, policy, n: int = 300, horizon: int = 30) -> float:
    """Fraction of steps the concentration sits in the therapeutic window."""
    rng = np.random.default_rng(0)
    in_range = 0
    for _ in range(n):
        s = 0
        for _ in range(horizon):
            s, _ = env.step(s, int(policy[s]), rng)
            in_range += env.c_low <= env.states[s] <= env.c_high
    return in_range / (n * horizon)


def validate(env: PKDosingEnv, pi_q) -> tuple[float, float]:
    """Ground-truth validation: does the learned policy beat the best fixed dose
    at holding the effect in the therapeutic window?"""
    best_fixed = max(
        range(env.n_actions),
        key=lambda a: _time_in_range(env, np.full(env.n_states, a)),
    )
    tir_learned = _time_in_range(env, pi_q)
    tir_fixed = _time_in_range(env, np.full(env.n_states, best_fixed))
    print("[validate] time in therapeutic range")
    print(f"[validate]   learned policy   = {tir_learned:.0%}")
    print(f"[validate]   best fixed dose  = {tir_fixed:.0%}\n")
    return tir_learned, tir_fixed


def interpret(t_half_d: float, frac_opt: float, tir_learned: float, tir_fixed: float) -> str:
    """Close with an explicit claim and named limitations."""
    return interpretation_block(
        claim=(
            f"A dosing policy learned on a warfarin PK model fit from real data "
            f"holds the anticoagulation effect in its therapeutic window "
            f"{tir_learned:.0%} of the time, versus {tir_fixed:.0%} for the best "
            f"fixed dose; the PK fit recovers warfarin's ~{t_half_d:.1f}-day "
            f"half-life, and the model-free learner reaches the model-based "
            f"optimum on {frac_opt:.0%} of states."
        ),
        limitations_list=[
            "The PK model is one-compartment and the control problem is a "
            "discretized MDP with a normalized concentration axis; real warfarin "
            "PK/PD is multi-compartment with a delayed (turnover) effect.",
            "Inter-patient variability is summarized as a three-point clearance "
            "distribution from 32 subjects; a larger, covariate-aware model would "
            "individualize dosing further.",
            "The therapeutic window and toxicity penalty are modeling choices, not "
            "calibrated INR targets; different clinical objectives change the policy.",
            "This is a pedagogical pipeline, NOT a clinical dosing tool -- real "
            "warfarin dosing uses validated nomograms and INR monitoring.",
        ],
    )


def main() -> None:
    """Run the full capstone pipeline end to end."""
    print("=" * 68)
    print("ddm4bio capstone -- warfarin PK/PD -> dosing policy (reference pipeline)")
    print("=" * 68 + "\n")

    df = load_data()
    quality_control(df)  # QC BEFORE results.
    kes = fit_pk(df)  # Group A: mechanistic PK fit (validated).
    characterize_pd(df)  # the effect endpoint that defines the target.
    env = calibrate_environment(kes)  # real variability -> environment.
    pi_q, _, frac_opt = learn_policy(env)  # Group B: RL (the hybrid).
    tir_learned, tir_fixed = validate(env, pi_q)  # policy vs fixed dose.

    t_half_d = float(np.log(2) / np.median(kes) / 24.0)
    print("-" * 68)
    print("INTERPRETATION")
    print("-" * 68)
    print(interpret(t_half_d, frac_opt, tir_learned, tir_fixed))


if __name__ == "__main__":
    main()
