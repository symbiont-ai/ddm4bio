# Real Data and Synthetic Ground Truth

One of [the course standards](how-it-works.md) is *ground-truth validation before
trusting real data*. This page explains what that means for the data you'll
actually touch — because this course does two things that can look contradictory
until you see why they belong together. It runs on **real biomedical data**, and
it deliberately uses **synthetic data** in specific places. Here is the rule that
reconciles them:

> **Score a result on real data whenever the result *can* be scored on real
> data. Reach for synthetic data only when the question is "did my method recover
> a known answer?" — because real biological data doesn't come with the answer.**

## Real data is the default

Every lesson and lab pulls a real dataset through the course data layer —
BloodMNIST cell crops, the Wisconsin breast-cytology panel, UCI heart-disease
records, MIT-BIH ECG, 10x pbmc3k single cells, the JHU COVID-19 series, and more.
Whenever that dataset carries the label or target your claim is about, you score
your result **on the real data itself** — real inputs, real labels, a real
held-out test. When you classify blood-cell images (Week 1), diagnose heart
disease (Problem Set 3), or read a breast-cytology panel (Problem Set 6), the
accuracy or AUC you report is measured on real samples and real patients.

## Synthetic data is for checking recovery

Some methods don't predict a labelled target — they try to **recover something
you can't observe directly**: the hidden sources mixed into a signal
(independent component analysis), the equations behind a trajectory (SINDy, DMD),
or a sparse signal behind incomplete measurements (compressed sensing). For these,
the only way to ask "is my answer right?" is to compare it against a **known**
answer — and real biological data has none. Nobody knows the true independent
sources inside a real single-cell dataset, or the true governing equation of a
real epidemic.

So you do what working scientists do: **plant a known answer in synthetic data,
prove your method recovers it, and only then turn the method loose on real
data**, where you make a smaller, honest claim.

- In **Week 5** you score ICA on a synthetic mixture whose sources you injected,
  then apply PCA to real pbmc3k cells (whose true sources are unknown, so that
  part stays qualitative).
- In **Week 7** you score SINDy and DMD on systems with known dynamics, then
  forecast a real COVID-19 curve and smooth a real ECG.
- In **Week 4** you score exact recovery on a known sparse signal, then denoise a
  real MIT-BIH heartbeat.

If your method can't recover a signal you planted yourself, it has no business
being trusted on one you didn't. This is a **necessity, not a shortcut**: swapping
in real data here wouldn't make the recovery score *more real* — it would make it
*impossible to compute*, because there would be nothing to check it against.

## You can always see which is which

The course never hides the ball:

- **Every dataset prints its source.** You'll see `source=real` with a named
  origin (PhysioNet, 10x Genomics, UCI), or `source=fallback` if a real download
  wasn't reachable.
- **Every result states its evidence and limits.** When a scored number comes
  from a synthetic fixture, the interpretation block says so — for example,
  "result is on a synthetic fixture that is exactly *k*-sparse."

And the published site is checked automatically: if any lesson couldn't load its
real data, the build **fails** rather than quietly showing you a synthetic
stand-in. What you read here ran on the real thing.

## The short version

| When your claim is… | Score it on… | Because… |
|---|---|---|
| "this predicts a labelled target" | **real data** | the label is in the real data, so a real test is possible |
| "this recovered a hidden truth" | **synthetic data** (then apply to real) | scoring recovery needs a known answer, which real data lacks |

Ground-truth validation first, real-data application second — that order is the
habit this course is trying to put into your fingers.

---

*For the full policy and the per-dataset compliance record, see the repository's
[`docs/DATA_AND_GROUND_TRUTH.md`](https://github.com/symbiont-ai/ddm4bio/blob/main/docs/DATA_AND_GROUND_TRUTH.md)
and [`data/DATA_CARD.md`](https://github.com/symbiont-ai/ddm4bio/blob/main/data/DATA_CARD.md).*
