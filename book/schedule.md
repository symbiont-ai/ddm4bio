# Schedule

The eight weeks below pair the methods you will learn with the biomedical application
that motivates them. Each row is a self-contained arc: the lesson introduces the
mathematics, and its one-click Colab lab exercises it on the corresponding data.

| Week | Methods | Biomedical application |
|------|---------|------------------------|
| **W1** | Foundations and the two paradigms: solving linear systems (Ax = b), eigenvalues and eigenvectors, and eigen-decomposition-based image recognition; reproducible workflows with Git and GitHub. | Recognizing images by their principal directions — an "eigen-cells" take on the classic eigenface idea, applied to cell imagery. |
| **W2** | Curve fitting, optimization, and sparsity: least-squares fitting, general optimization, and L1-penalized (sparse) regression. | Fitting dose-response curves, and selecting a small, sparse panel of biomarkers from many candidate measurements. |
| **W3** | Machine learning as curve fitting; introduction to neural networks: generalization, overfitting, and cross-validation. | Predicting clinical outcomes when only a small number of patients are available. |
| **W4** | Signals and images: the Fourier transform, wavelets, and compressed sensing. | Time-frequency analysis of EEG and ECG recordings; MRI denoising and accelerated image reconstruction from undersampled data. |
| **W5** | Dimensionality reduction: the SVD, PCA, robust PCA, and ICA. | PCA of single-cell data; separating mixed sources in EEG and fMRI signals. |
| **W6** | Learning from data — unsupervised and supervised: clustering, classifiers, and correction for multiple testing / false-discovery rate. | Discovering cell types and disease subtypes, and building diagnostic classifiers. |
| **W7** | Dynamics from data: dynamic mode decomposition (DMD), Koopman-operator methods, SINDy, and Kalman-filter data assimilation. | Modeling epidemic and physiological dynamics, and filtering noisy vital-sign streams. |
| **W8** | Deep learning for dynamics, reinforcement learning, and foundation models: autoencoders, shallow sensing-and-reconstruction (SHRED) architectures, reinforcement learning, and transformers. | Learning latent representations, optimizing dosing policies, and modeling clinical text with large language models. |

```{note}
Where a lesson cites a chapter number, that number refers to J. N. Kutz's textbook
*Data-Driven Modeling & Scientific Computation* (2nd ed.), offered as optional reading
for a deeper treatment. This course explains every concept in its own words and does not
reproduce that text.
```
