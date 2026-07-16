# Syllabus

## Format

This is a **self-paced, 8-week online course** — enroll and start any time, and move at
whatever pace suits you. "Week" is simply a unit of coherent material, not a fixed
schedule. Each week has two components:

- **A lesson** that develops the ideas and the mathematics in plain language, with fully
  worked code. Read it on the page, or open the same notebook in Google Colab with one
  click to run and modify it yourself in the browser (with a free GPU for the
  deep-learning week).
- **A problem set** that puts the method to work on a *new* problem the lesson does not
  solve for you: the reusable pieces are provided, and you implement the missing logic
  yourself. It is distributed and graded automatically through GitHub Classroom.

## Tools

You will work entirely in the modern scientific-Python ecosystem:

- **NumPy** and **SciPy** for numerical arrays, linear algebra, and signal processing.
- **scikit-learn** for classical machine learning.
- **PyTorch** for neural networks and deep learning (optional — the deep-learning lab
  runs a dependency-light NumPy demonstration, with PyTorch as an extension).
- **Jupyter** notebooks as the interactive computing environment (run in Google Colab).
- **GitHub** for version control, reproducibility, and sharing your work.

## What the course emphasizes

The heart of the course is the family of **data-driven methods** — the material that in
Kutz's textbook lives largely in its data-methods chapters. We spend our time on:

- **Dimensionality reduction and factorization** — the singular value decomposition
  (SVD), principal component analysis (PCA), robust PCA, and independent component
  analysis (ICA).
- **Signals, images, and sparsity** — Fourier and wavelet analysis, and compressed
  sensing.
- **Machine learning** — both unsupervised (clustering, source separation) and
  supervised (classification, regression, generalization).
- **Dynamics learned from data** — dynamic mode decomposition (DMD), Koopman-operator
  ideas, sparse identification of nonlinear dynamics (SINDy), and data assimilation /
  Kalman filtering.
- **Deep learning** — autoencoders, sensing-and-reconstruction architectures,
  reinforcement learning, and transformer / foundation models.

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
   using SVD, PCA, robust PCA, and ICA, and interpret what the resulting components mean.
4. **Analyze signals and images** with Fourier and wavelet transforms, and reconstruct
   signals from incomplete measurements using compressed sensing.
5. **Build and validate machine-learning models** — choosing models sensibly, guarding
   against overfitting through cross-validation, and honestly assessing generalization,
   especially in the small-sample regime typical of clinical data.
6. **Discover dynamics from measurements** using DMD, SINDy, and data-assimilation /
   filtering methods, and use them to forecast and to filter noisy time series.
7. **Apply modern deep learning** — autoencoders, reinforcement learning, and
   transformer-based models — to representation learning, control, and clinical-text and
   sequence problems, with an eye to when the added complexity is warranted.
