---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.0
kernelspec:
  display_name: "Python 3"
  language: python
  name: python3
---

# Week 3 - Machine Learning as Curve Fitting: Generalization on Small-n Data

Strip away the vocabulary and most supervised machine learning is a single
old idea: *fit a flexible curve to observed points*. A polynomial, a random
forest, a neural network -- each is a family of functions with adjustable
knobs, and "training" is choosing the knobs so the curve passes near the data.
That framing is clarifying but also dangerous, because a curve that hugs your
training points beautifully can be worthless the moment you ask it about a
point it never saw. The gap between *memorizing the training set* and
*predicting new data* is the single most important idea in this course, and in
the life sciences -- where a cohort of 60 patients is a large study, not a
small one -- it is where careers and clinical claims quietly go wrong.

This lesson makes that gap visible and then measurable. We first watch a
high-degree polynomial and a small neural network both fit a synthetic curve
almost perfectly *inside* the sampled range and then fail catastrophically
just outside it -- the difference between interpolation and extrapolation. We
then move to a real biomedical dataset and quantify generalization honestly:
k-fold cross-validation for an unbiased performance estimate, a learning curve
to see at what sample size the model stops overfitting, and a permutation test
to prove the model has learned something a shuffled-label model could not.
Throughout we watch the train/validation/test gap and guard against the most
common way small-n studies lie to themselves: information leakage.

**Reading.** Kutz, *Data-Driven Modeling & Scientific Computation*, 2nd ed.,
Chapters 6 and 13 (regression, model selection, and the bias-variance view of
learning). Read those for the derivations; everything below is explained in
our own terms and run against our own fixtures.

**Learning goals.**

- See supervised learning as curve fitting, and distinguish interpolation
  (predicting inside the sampled domain) from extrapolation (predicting
  outside it), where flexible models fail.
- Estimate generalization performance without fooling yourself, using k-fold
  cross-validation and a held-out test set, and read the train/validation/test
  gap as a diagnosis of over- versus under-fitting.
- Use a learning curve to judge at what sample size a model stops overfitting.
- Confirm a model beats chance with a permutation test, and close with an
  explicit confidence-and-limitations statement.

## Setup

We seed all random number generators and apply the course plotting style so
the figures below are deterministic and reproducible from a cold kernel.

```{code-cell} ipython3
import numpy as np

import ddm4bio
from ddm4bio import seed_everything
from ddm4bio.viz.style import set_style

seed_everything()
set_style()

print(f"ddm4bio version: {ddm4bio.__version__}")
```

## 1. Curve fitting on known ground truth: interpolation vs. extrapolation

To see the interpolation/extrapolation distinction cleanly we need a problem
whose true answer we already know. We define a smooth ground-truth response
curve `f(x)` -- think of it as the true relationship between a log-dose of a
compound and a normalized assay readout -- and pretend our instrument can only
measure it over a limited *calibrated range*. Inside that range we collect
noisy training points; outside it we have no data at all. An honest model
should reproduce `f` where it was trained and should not be trusted where it
was not.

```{code-cell} ipython3
def f_true(x):
    """Smooth ground-truth response curve (known exactly for scoring)."""
    return np.sin(1.3 * x) + 0.3 * x

rng = np.random.default_rng(0)

# The instrument is calibrated only over x in [-3, 3].
x_min, x_max = -3.0, 3.0
x_train = np.sort(rng.uniform(x_min, x_max, size=40))
y_train = f_true(x_train) + 0.15 * rng.standard_normal(x_train.size)

# A dense grid that reaches BEYOND the calibrated range, into extrapolation.
x_grid = np.linspace(x_min - 3.0, x_max + 3.0, 400)
y_grid_true = f_true(x_grid)

# Book-keeping masks: which grid points are interpolation vs extrapolation.
interp_mask = (x_grid >= x_min) & (x_grid <= x_max)
extrap_mask = ~interp_mask

print(f"Training points: {x_train.size} over x in [{x_min}, {x_max}]")
print(f"Evaluation grid: {x_grid.size} points over "
      f"[{x_grid.min():.0f}, {x_grid.max():.0f}] (interp + extrap)")
```

We fit two very different flexible models to exactly these points. The first
is a **degree-12 polynomial** -- a classic high-variance curve fitter. The
second is a **small multilayer perceptron** (a two-hidden-layer neural
network) from scikit-learn. We standardize the input for the network inside a
pipeline so the scaler is estimated from the training inputs only; the same
transform is then applied to the grid.

```{code-cell} ipython3
from numpy.polynomial import Polynomial
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# High-degree polynomial fit (least squares on a well-conditioned basis).
poly = Polynomial.fit(x_train, y_train, deg=12)
y_poly = poly(x_grid)

# Small MLP: two hidden layers, fixed random_state for determinism.
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation="tanh",
        max_iter=5000,
        random_state=0,
    ),
)
mlp.fit(x_train.reshape(-1, 1), y_train)
y_mlp = mlp.predict(x_grid.reshape(-1, 1))

print("Both models fit; evaluating on the interpolation and extrapolation grids.")
```

Now plot the truth, the training points, and both fitted curves. The shaded
band marks the calibrated range where training data exist; everything outside
it is extrapolation.

```{code-cell} ipython3
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9, 5))
ax.axvspan(x_min, x_max, color="0.85", label="calibrated range (training)")
ax.plot(x_grid, y_grid_true, color="black", linewidth=2, label="ground truth $f(x)$")
ax.scatter(x_train, y_train, s=18, color="0.35", zorder=5, label="training points")
ax.plot(x_grid, y_poly, linestyle="--", linewidth=1.8, label="degree-12 polynomial")
ax.plot(x_grid, y_mlp, linestyle="-.", linewidth=1.8, label="small MLP")

# Keep the y-axis readable despite wild extrapolation blow-ups.
pad = 2.0
ax.set_ylim(y_grid_true.min() - pad, y_grid_true.max() + pad)
ax.set_xlabel("input $x$ (e.g. log-dose)")
ax.set_ylabel("response")
ax.set_title("Interpolation is easy; extrapolation is where flexible models break")
ax.legend(loc="upper left", fontsize=8)
fig;
```

Visually both curves track the truth almost perfectly inside the shaded band
and then peel away from it -- often violently -- the instant they leave it. We
quantify this against the *known* `f(x)` by splitting the grid error into an
interpolation part and an extrapolation part.

```{code-cell} ipython3
def rmse(pred, truth, mask):
    err = (pred - truth)[mask]
    return float(np.sqrt(np.mean(err ** 2)))

print("Root-mean-square error against ground truth:\n")
print(f"{'model':<20}{'interpolation':>15}{'extrapolation':>15}")
for name, pred in [("degree-12 poly", y_poly), ("small MLP", y_mlp)]:
    ri = rmse(pred, y_grid_true, interp_mask)
    re = rmse(pred, y_grid_true, extrap_mask)
    print(f"{name:<20}{ri:>15.3f}{re:>15.3f}")
```

**QC note.** For both models the interpolation error is small (comparable to
the 0.15 noise we injected) while the extrapolation error is larger by one to
several orders of magnitude. This is not a bug to be tuned away -- it is the
fundamental fact that a curve fitted to data in one region carries almost no
information about a region it never saw. The practical rule for biology: state
the domain your model was trained on, and refuse to predict outside it.

## 2. Honest generalization on a real biomedical dataset

We now move from a curve to a classifier and from a toy to a real biomedical
dataset: the UCI Heart Disease (processed Cleveland) cohort. Each of the 303
patients is described by 13 clinical features -- age, sex, chest-pain type,
resting blood pressure, cholesterol, maximum heart rate, exercise-induced
angina, ST-segment measurements, and so on -- and the label marks the presence
of heart disease. There is no ground-truth "curve" here, so the whole burden of
trust falls on *how we measure generalization*.

We load the data through the course dataset layer. `get_dataset` fetches and
caches the real UCI CSV when the network and its optional dependencies are
available, and otherwise returns a clearly labeled synthetic fallback with the
*same payload shape* (`X` a 303x13 table, `y` a binary label). The analysis
below runs identically on either, so we always print the provenance first and
let the reader see which one they got.

```{code-cell} ipython3
from ddm4bio.datasets import get_dataset

ds = get_dataset("heart_uci", download=True)
print(f"data source: {ds.source}")
print(f"provenance:  {ds.provenance}")

X = ds.payload["X"]           # 303 x 13 clinical DataFrame (may hold missing values)
y = ds.payload["y"].to_numpy()

print(f"\nFeature matrix X: {X.shape} (samples x features)")
print(f"Missing values:   {int(X.isna().sum().sum())} entries (imputed inside the pipeline)")
print(f"Labels y:         {y.shape}, classes = {np.unique(y)}")
print(f"Class balance:    {np.bincount(y)} (no disease, disease)")
```

Our model is a standardized logistic regression, wrapped in a pipeline. Because
the raw clinical table has a few missing entries, the first pipeline step is a
median imputer; the scaler and classifier follow. Wrapping imputation *and*
scaling *inside* the estimator is not cosmetic: it guarantees that when
cross-validation or the learning curve refits the model on a training fold, the
imputation and standardization statistics are computed from that fold alone.
Fitting an imputer or scaler on all the data before splitting is the textbook
form of **information leakage**, and it silently inflates every score below.

```{code-cell} ipython3
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

def make_model():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000, random_state=0),
    )

print("Model: SimpleImputer -> StandardScaler -> LogisticRegression (leakage-safe)")
```

### 2.1 The train / validation / test gap

Before any cross-validation, we set aside a **test set** that the model will
not see until the very end, and we look at three numbers: accuracy on the data
the model was fit on (train), a cross-validated estimate on the training
portion (validation), and accuracy on the untouched test set. The spread
between them is the generalization gap.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from ddm4bio.methods.learning import cross_validate

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=0
)

model = make_model()
model.fit(X_tr, y_tr)

train_acc = model.score(X_tr, y_tr)
test_acc = model.score(X_te, y_te)

# Validation estimate: 5-fold CV *within the training set only*.
cv = cross_validate(make_model(), X_tr, y_tr, cv=5, scoring="accuracy", seed=0)

print(f"Train accuracy:      {train_acc:.3f}")
print(f"Validation (5-fold): {cv['mean']:.3f} +/- {cv['std']:.3f}")
print(f"Test accuracy:       {test_acc:.3f}  (held out until now)")
print(f"\nTrain - validation gap: {train_acc - cv['mean']:+.3f}")
```

**QC note.** A small train-minus-validation gap says the model is *not* badly
overfitting: it performs about as well on held-out folds as on the data it
trained on. The test accuracy landing close to the cross-validated estimate is
the confirmation we wanted -- our validation procedure was an honest predictor
of performance on genuinely unseen patients, and the untouched test set was never
allowed to leak into model selection.

### 2.2 A learning curve: when does overfitting stop?

The learning curve plots training and validation score as a function of how
many samples the model is allowed to learn from. When training score sits far
above validation score, the model is overfitting -- it is memorizing a small
sample. As the two curves converge, overfitting gives way to a stable,
generalizing fit. Reading *where* they converge answers a concretely useful
question: how much data is enough?

```{code-cell} ipython3
from ddm4bio.methods.learning import learning_curve

lc = learning_curve(
    make_model(), X, y,
    train_sizes=np.linspace(0.1, 1.0, 8),
    cv=5, seed=0,
)

train_mean = lc["train_scores"].mean(axis=1)
train_std = lc["train_scores"].std(axis=1)
val_mean = lc["val_scores"].mean(axis=1)
val_std = lc["val_scores"].std(axis=1)
sizes = lc["train_sizes"]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sizes, train_mean, marker="o", label="training score")
ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
ax.plot(sizes, val_mean, marker="s", label="validation score")
ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
ax.set_xlabel("number of training samples")
ax.set_ylabel("accuracy")
ax.set_title("Learning curve: the train/validation gap closes as n grows")
ax.legend(loc="lower right")
fig;
```

```{code-cell} ipython3
gap = train_mean - val_mean
print(f"{'n_train':>8}{'train':>10}{'val':>10}{'gap':>10}")
for n, tr, vl, g in zip(sizes, train_mean, val_mean, gap):
    print(f"{int(n):>8}{tr:>10.3f}{vl:>10.3f}{g:>10.3f}")

# Smallest training size at which the gap is under 2 accuracy points.
small = sizes[gap <= 0.02]
if small.size:
    print(f"\nGap falls below 0.02 at n = {int(small[0])} training samples.")
else:
    print("\nGap never falls below 0.02 over the sampled range.")
```

**QC note.** At the smallest training sizes the model can nearly memorize the
data, so training accuracy is high and the gap to validation is widest -- the
signature of overfitting on small n. As `n` increases the gap shrinks and the
two curves settle toward a common ceiling; that convergence point is where the
model stops overfitting and extra data buys little. This is the quantitative
version of the folk warning that small cohorts overfit.

### 2.3 A permutation test: does the model beat chance?

A model can post a respectable accuracy simply because a dataset is easy or
imbalanced. The permutation test answers the sharper question directly: is the
learned relationship between features and labels real? We repeatedly shuffle
the labels -- destroying any true association while preserving class
proportions -- refit under cross-validation, and build the null distribution of
scores achievable by chance. The observed score is significant only if it sits
far out in the tail of that null.

```{code-cell} ipython3
from ddm4bio.methods.learning import permutation_test

perm = permutation_test(
    make_model(), X, y,
    n_perm=200, cv=5, scoring="accuracy", seed=0,
)

print(f"Observed CV accuracy:     {perm['observed_score']:.3f}")
print(f"Permutation null mean:    {perm['permutation_scores'].mean():.3f}")
print(f"Permutation null max:     {perm['permutation_scores'].max():.3f}")
print(f"p-value:                  {perm['p_value']:.4f}")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(perm["permutation_scores"], bins=20, color="0.7",
        edgecolor="white", label="null (shuffled labels)")
ax.axvline(perm["observed_score"], color="crimson", linewidth=2,
           label=f"observed = {perm['observed_score']:.3f}")
ax.set_xlabel("cross-validated accuracy")
ax.set_ylabel("count")
ax.set_title("Permutation test: observed score vs. the chance distribution")
ax.legend(loc="upper center")
fig;
```

**QC note.** The null distribution clusters near the majority-class rate -- the
best you can do while ignoring the features -- while the observed score sits far
to its right, yielding a tiny p-value. The lowest attainable p-value is
`1 / (n_perm + 1)`, so with 200 permutations a result reported as the minimum
means only "smaller than we can resolve at this permutation count," not exactly
zero. Either way, the model has learned a real feature-label relationship, not
an artifact of class balance.

## 3. Interpretation

Every ddm4bio analysis closes with an explicit interpretation block: a single
claim, an honest confidence level backed by named evidence, and a list of
stated limitations. This forces us to write down not just *what* we found but
*how much* we should believe it and *where* it could break.

```{code-cell} ipython3
from ddm4bio.interpret import interpretation_block

block = interpretation_block(
    claim="The standardized logistic-regression model generalizes on the "
          "heart-disease data: it beats chance decisively and its validation "
          "accuracy tracks its training accuracy rather than overfitting, "
          "though the train/validation gap stays visible on this small cohort.",
    confidence="moderate",
    limitations_list=[
        f"Generalization is estimated, not guaranteed: the held-out test "
        f"accuracy ({test_acc:.3f}) and the 5-fold validation estimate "
        f"({cv['mean']:.3f}) agree here, but both are single draws on one "
        f"cohort of {X.shape[0]} patients.",
        "The permutation p-value is bounded below by 1/(n_perm+1) = "
        f"{1.0 / (200 + 1):.4f}; it shows significance, not an exact value.",
        "The interpolation/extrapolation demo shows flexible models are "
        "untrustworthy outside their training domain -- the same caution "
        "applies to this classifier on patients unlike the training cohort.",
        "No leakage was introduced (scaler fit inside CV folds), but results "
        "still assume the samples are independent and identically "
        "distributed, which clinical cohorts often violate.",
    ],
    evidence=f"5-fold CV accuracy = {cv['mean']:.3f} +/- {cv['std']:.3f}, "
             f"held-out test accuracy = {test_acc:.3f}, permutation "
             f"p-value = {perm['p_value']:.4f} against a shuffled-label null.",
)
print(block)
```

## Exercises

Your graded work for this week is **Problem Set 3 (PS3)**, distributed and
auto-graded through GitHub Classroom. Building on this lesson, PS3 asks you to:

- Sweep the polynomial degree and the MLP width on the Section 1 fixture, and
  chart how interpolation and extrapolation error diverge as model flexibility
  grows -- the bias-variance trade-off made concrete.
- Deliberately introduce information leakage (fit the scaler on all the data
  before splitting) and measure how much it inflates the cross-validated score
  versus the leakage-safe pipeline -- then explain the mechanism.
- Rerun the learning curve on progressively smaller subsamples of the
  heart-disease data and report the sample size at which the train/validation
  gap stops closing, using `ddm4bio.methods.learning.learning_curve`.
- Run a permutation test with `ddm4bio.methods.learning.permutation_test` on a
  deliberately uninformative feature subset and confirm the p-value behaves as
  the null predicts.
- Write an interpretation block for each result using
  `ddm4bio.interpret.interpretation_block`, with a defensible confidence level.

Refer to the PS3 repository README for the submission and auto-grading details.
