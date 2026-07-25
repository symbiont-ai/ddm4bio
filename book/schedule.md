# Schedule

The eight modules below — and the capstone that follows them — pair the methods you will
learn with the biomedical application that motivates them and the dataset it runs on. Each
row is a self-contained arc: the
lesson introduces the mathematics, and its one-click Colab lab exercises it on the
listed data. Where a method needs a known answer to be scored, the lab pairs a real
dataset with a **synthetic ground-truth fixture** — see
[Data & ground truth](data-and-ground-truth.md) for why.

| Module | Methods | Biomedical application | Dataset |
|------|---------|------------------------|---------|
| [**M1**](lessons/wk1_linear_systems.ipynb) | Foundations: solving Ax = b — determined linear systems, the least-squares solution (pseudoinverse / normal equations), and conditioning read from the singular values. | Registering medical images: recover an affine transform from landmark correspondences by least squares, and judge when the solve can be trusted. | **retinal fundus** image (scikit-image) + known-transform fixture |
| [**M2**](lessons/wk2_curvefit_sparsity.ipynb) | Curve fitting, noisy differentiation, and sparsity: nonlinear least-squares curve fitting (with uncertainty), regularized (Tikhonov) differentiation of noisy signals, and L1-penalized (sparse) regression. | Fitting dose-response curves, and selecting a small, sparse panel of biomarkers from many candidate measurements. | **CCLE** drug dose-response · **Wisconsin (WDBC)** breast-cytology panel |
| [**M3**](lessons/wk3_ml_generalization.ipynb) | Machine learning as curve fitting: interpolation versus extrapolation, generalization, overfitting, and cross-validation. | Predicting clinical outcomes when only a small number of patients are available. | **UCI Heart Disease** cohort |
| [**M4**](lessons/wk4_signals_cs.ipynb) | Signals: the Fourier transform, wavelets, and compressed sensing. | Time-frequency analysis and wavelet denoising of ECG, and reconstructing sparse signals from far fewer measurements than the Nyquist rate. | **MIT-BIH ECG** (PhysioNet) · synthetic sparse-signal fixtures |
| [**M5a**](lessons/wk5a_linear_subspaces.ipynb) | Linear subspaces: the SVD and PCA, eigen-cells (PCA on images), robust PCA, and linear discriminant analysis (LDA). | Low-dimensional structure in single-cell expression; recognizing cells from a compact eigen-cell basis. | **10x PBMC3k** single-cell · **BloodMNIST** blood-cell images |
| [**M5b**](lessons/wk5b_sources_manifolds.ipynb) | Beyond PCA: independent component analysis (ICA) for source separation, and nonlinear neighbor embeddings (t-SNE, UMAP). | Separating statistically independent mixed sources; laying out single-cell data while distrusting the distances it distorts. | synthetic ICA sources · **10x PBMC3k** single-cell |
| [**M6**](lessons/wk6_cluster_classify.ipynb) | Learning from data — unsupervised and supervised: clustering, classifiers, and correction for multiple testing / false-discovery rate. | Discovering cell types and disease subtypes, and building diagnostic classifiers. | **10x PBMC3k** single-cell · **Wisconsin (WDBC)** |
| [**M7**](lessons/wk7_dynamics.ipynb) | Dynamics from data: dynamic mode decomposition (DMD), SINDy, and Kalman-filter data assimilation. | Modeling epidemic and physiological dynamics, and filtering noisy vital-sign streams. | **JHU COVID-19** case series · **MIT-BIH ECG** · synthetic dynamical systems |
| [**M8**](lessons/wk8_deep_learning.ipynb) | Deep learning for representations: linear and nonlinear autoencoders (a linear autoencoder is exactly PCA) and a convolutional net on blood-cell images. | Learning a nonlinear latent that follows a curved manifold, and recognizing blood cells with learned convolutional features. | Synthetic manifold fixture · Wisconsin (WDBC) · **BloodMNIST** blood-cell images |
| [**Capstone**](capstone.md) | Model-informed reinforcement learning for precision dosing: fit a mechanistic one-compartment PK model and an indirect-response PD model (model-driven), characterize inter-patient variability (data-driven), calibrate a dosing environment, and learn a dosing policy with value iteration and Q-learning (see the [RL preview](capstone_rl_preview.ipynb)). | Designing a personalized **warfarin** dosing policy — the course's model-driven ↔ data-driven hybrid in one project. | **Warfarin PK/PD** (nlmixr2data) · synthetic PK/PD fixture |

```{note}
Where a lesson cites a chapter number, that number refers to J. N. Kutz's textbook
*Data-Driven Modeling & Scientific Computation* (2nd ed.), offered as optional reading
for a deeper treatment. This course explains every concept in its own words and does not
reproduce that text.
```
