# Schedule

The eight weeks below pair the methods you will learn with the biomedical application
that motivates them and the dataset it runs on. Each row is a self-contained arc: the
lesson introduces the mathematics, and its one-click Colab lab exercises it on the
listed data. Where a method needs a known answer to be scored, the lab pairs a real
dataset with a **synthetic ground-truth fixture** — see
[Data & ground truth](data-and-ground-truth.md) for why.

| Week | Methods | Biomedical application | Dataset |
|------|---------|------------------------|---------|
| **W1** | Foundations and the two paradigms: solving linear systems (Ax = b), eigenvalues and eigenvectors, and eigen-decomposition-based image recognition; reproducible workflows with Git and GitHub. | Recognizing cells by their principal directions — an "eigen-cells" take on the classic eigenface idea, applied to cell imagery. | **BloodMNIST** blood-cell images |
| **W2** | Curve fitting, optimization, and sparsity: least-squares fitting, general optimization, and L1-penalized (sparse) regression. | Fitting dose-response curves, and selecting a small, sparse panel of biomarkers from many candidate measurements. | **GDSC** drug dose-response · **Wisconsin (WDBC)** breast-cytology panel |
| **W3** | Machine learning as curve fitting; introduction to neural networks: generalization, overfitting, and cross-validation. | Predicting clinical outcomes when only a small number of patients are available. | **UCI Heart Disease** cohort |
| **W4** | Signals and images: the Fourier transform, wavelets, and compressed sensing. | Time-frequency analysis and wavelet denoising of ECG, and reconstructing sparse signals from far fewer measurements than the Nyquist rate. | **MIT-BIH ECG** (PhysioNet) · synthetic sparse-signal fixtures |
| **W5** | Dimensionality reduction: the SVD, PCA, robust PCA, and ICA. | Finding the low-dimensional structure in single-cell expression data, and separating statistically independent mixed sources. | **10x PBMC3k** single-cell · synthetic ICA sources |
| **W6** | Learning from data — unsupervised and supervised: clustering, classifiers, and correction for multiple testing / false-discovery rate. | Discovering cell types and disease subtypes, and building diagnostic classifiers. | **10x PBMC3k** single-cell · **Wisconsin (WDBC)** |
| **W7** | Dynamics from data: dynamic mode decomposition (DMD), Koopman-operator methods, SINDy, and Kalman-filter data assimilation. | Modeling epidemic and physiological dynamics, and filtering noisy vital-sign streams. | **JHU COVID-19** case series · **MIT-BIH ECG** · synthetic dynamical systems |
| **W8** | Deep learning for dynamics, reinforcement learning, and foundation models: autoencoders, shallow sensing-and-reconstruction (SHRED) architectures, reinforcement learning, and transformers. | Learning latent representations, optimizing dosing policies, and modeling clinical text with large language models. | Synthetic demonstrations (dependency-light) |

```{note}
Where a lesson cites a chapter number, that number refers to J. N. Kutz's textbook
*Data-Driven Modeling & Scientific Computation* (2nd ed.), offered as optional reading
for a deeper treatment. This course explains every concept in its own words and does not
reproduce that text.
```
