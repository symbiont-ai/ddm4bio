# Syllabus

## Format

This is a **self-paced, 8-week online course** — enroll and start any time, and move at
whatever pace suits you. "Week" is simply a unit of coherent material, not a fixed
schedule. Each week has two components:

- **A lesson** that develops the ideas and the mathematics in plain language, with fully
  worked code. Read it on the page, or open the same notebook in Google Colab with one
  click to run and modify it yourself in the browser.
- **A problem set** that puts the method to work on a *new* problem the lesson does not
  solve for you: the reusable pieces are provided, and you implement the missing logic
  yourself. It is distributed and graded automatically through GitHub Classroom.

## Prerequisites & who this is for

This is an **intermediate-to-advanced quantitative** course — not a first course in
programming, statistics, or machine learning. If you already have linear algebra,
statistics, and Python, you're in the right place. It's meant to be welcoming, not
gatekeeping.

You'll move fastest if you're comfortable with:

- **Linear algebra** — vectors, matrices, eigenvalues/eigenvectors, and ideally having
  *seen* the SVD. Week 1 sets up and solves linear systems and reads conditioning off the
  singular values from day one; Week 5 then uses the SVD and eigendecomposition fluently for PCA.
- **Probability & statistics** — distributions, variance, hypothesis testing and
  p-values, and the idea of a *null distribution*. Weeks 3 and 6 lean on base rates
  and Bayes, permutation nulls, Type-I error, and selective inference.
- **Python + NumPy** — comfortable writing *and modifying* scientific code, not writing
  your first program.
- **Calculus** — gradients and derivatives (gradient descent, finite-time rates,
  Jacobians conceptually).

A prior **machine-learning course is recommended, not required.** With one, Weeks 2–3
and the deep-learning week (8) go smoothly; without one, those are the steep parts —
reachable, just steeper. Everything else assumes only the bar above.

**Rusty? Brush up on** solving linear systems and least squares (the normal equations and the
pseudoinverse), the SVD and eigendecomposition (what singular values mean and how
to read them), hypothesis testing and null / permutation distributions, and NumPy array
indexing and linear-algebra calls. Those carry the most weight early on.

**One-glance self-check.** You should be comfortable reading code like this:

```python
import numpy as np

X = np.random.randn(100, 5)          # 100 samples, 5 features
U, s, Vt = np.linalg.svd(X, full_matrices=False)
print(s)                             # singular values, largest first
print(s**2 / np.sum(s**2))           # variance explained per component
```

If that reads clearly, you're ready. If not, start with the SVD and NumPy pointers above.

## Tools

You will work entirely in the modern scientific-Python ecosystem:

- **NumPy** and **SciPy** for numerical arrays, linear algebra, and signal processing.
- **scikit-learn** for classical machine learning.
- **PyTorch** for neural networks and deep learning — the Week-8 lesson trains small
  models device-agnostically (on a GPU when one is available, otherwise CPU) and is
  designed to run on CPU.
- **Jupyter** notebooks as the interactive computing environment (run in Google Colab).
- **GitHub** for version control, reproducibility, and sharing your work.

## What the course emphasizes

The heart of the course is the family of **data-driven methods** — the material that in
Kutz's textbook lives largely in its data-methods chapters. We spend our time on:

- **Dimensionality reduction and factorization** — the singular value decomposition
  (SVD), principal component analysis (PCA), robust PCA, independent component
  analysis (ICA), linear discriminant analysis (LDA), and nonlinear neighbor
  embeddings (t-SNE, UMAP).
- **Signals and sparsity** — Fourier and wavelet analysis, and compressed
  sensing.
- **Machine learning** — both unsupervised (clustering, source separation) and
  supervised (classification, regression, generalization).
- **Dynamics learned from data** — dynamic mode decomposition (DMD), sparse
  identification of nonlinear dynamics (SINDy), and data assimilation /
  Kalman filtering.
- **Deep learning** — linear and nonlinear autoencoders, a convolutional classifier, and
  sensing-and-reconstruction (SHRED) architectures in Week 8. Reinforcement learning — a
  distinct paradigm (learning to act) — is previewed and developed in the capstone.
  Transformers and foundation-model embeddings sit beyond this course's scope.

## What the course de-emphasizes

We deliberately spend **less** time on the internals of classical numerical solvers for
differential equations — finite-difference and spectral schemes, stability analysis of
time-steppers, and the like. These are important and beautiful, but they are the subject
of a different course. Here they appear as **references only**: we point you to the
relevant textbook chapters when a deeper treatment would help, and otherwise treat
solvers as reliable tools rather than objects of study.

## Learning outcomes

By the end of the course, you should be able to:

1. **Distinguish the two reasoning paradigms** — model-driven and data-driven — and
   judge which is appropriate for a given biomedical problem, or how to combine them.
2. **Wield large-scale linear algebra** — set up and solve linear systems, reason about
   eigenvalues and singular values, and use factorizations as practical analysis tools.
3. **Reduce dimensionality and extract structure** from high-dimensional biological data
   using SVD, PCA, robust PCA, ICA, LDA, and nonlinear neighbor embeddings (t-SNE, UMAP), and
   interpret what the resulting components mean — including how neighbor embeddings can
   distort global structure.
4. **Analyze signals** with Fourier and wavelet transforms, and reconstruct
   signals from incomplete measurements using compressed sensing.
5. **Build and validate machine-learning models** — choosing models sensibly, guarding
   against overfitting through cross-validation, and honestly assessing generalization,
   especially in the small-sample regime typical of clinical data.
6. **Discover dynamics from measurements** using DMD, SINDy, and data-assimilation /
   filtering methods, and use them to forecast and to filter noisy time series.
7. **Apply modern deep learning** — train autoencoders (linear and nonlinear), a
   convolutional classifier, and sensing-and-reconstruction (SHRED) models, and learn a
   control policy by reinforcement learning (previewed and developed in the
   capstone) — with an eye to when the added complexity is warranted. Transformers and
   foundation-model embeddings sit beyond this course's scope.
