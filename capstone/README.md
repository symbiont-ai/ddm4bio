# Capstone — Data-Driven Methods for the Life Sciences

The capstone is where the course stops being a tour of methods and becomes a
piece of your own scientific work. You will **design and execute a
reproducible pipeline** that applies course methods to a biomedical question
you care about, and you will defend not just your answer but the *process* that
produced it.

The course follows J. N. Kutz, *Data-Driven Modeling & Scientific Computation*.
This capstone reuses the teaching library (`ddm4bio`) you have used all term:
the same loaders, QC checks, decompositions, dynamics/learning methods,
validation metrics, and interpretation helpers. You are not asked to reinvent
them — you are asked to *compose* them into a defensible analysis.

---

## What you must build

A self-contained project that takes a biomedical dataset from raw input to a
calibrated, interpreted conclusion, wired together as a single runnable
pipeline. Concretely, your pipeline must:

1. **Frame** a specific biomedical question that a data-driven method can
   actually answer (not "explore the data").
2. **Load** a dataset (real, or a defensible synthetic/semi-synthetic proxy)
   through a reproducible loader.
3. **Quality-control the data before drawing any conclusion** — and report the
   QC, including what you decided to drop or keep and why.
4. Apply **at least TWO course methods**, one from each group below.
5. **Validate against ground truth** before you trust results on real data
   (a synthetic fixture with a known answer, a held-out split, a positive/
   negative control, or a published benchmark).
6. **Interpret** the result with an explicit, calibrated confidence level and a
   stated list of limitations.
7. Ship it so **someone else can reproduce your numbers** from a clean clone.

### The two-method requirement

You must use **one method from each column**. Using two methods from the same
column does not satisfy the requirement.

| Group A — representation / decomposition | Group B — learning / dynamics |
|------------------------------------------|-------------------------------|
| SVD / PCA                                | Supervised or unsupervised ML |
| Robust PCA                               | DMD / SINDy                   |
| ICA                                      | Kalman filtering              |
| Wavelets / compressed sensing            | Autoencoder                   |

The two methods must be *connected*: the Group A representation should feed, or
be justified by, the Group B step (e.g. PCA denoising before a classifier; SVD
truncation before DMD; ICA source separation before a state-space model). A
pipeline where the two methods never interact will not score well on framing.

Every method you report must map to a named function in `ddm4bio` (or a clearly
documented, tested extension of it). **No silent method swaps**: if you change
the method after your proposal, say so and say why.

---

## Suggested tracks

Pick one track, or propose your own with equivalent rigor. Each row is a
starting point, not a script — the example question is meant to be sharpened.

| Track | Typical data | Group A → Group B pipeline | Example question |
|-------|--------------|----------------------------|------------------|
| **Single-cell / omics** | scRNA-seq counts (e.g. PBMC), bulk expression | QC + normalize → **PCA/SVD** embedding → **unsupervised clustering** (+ marker interpretation) | Do the transcriptomic clusters correspond to known cell types, and how stable are they to subsampling? |
| **Neuro signals** | EEG/LFP/MEG multichannel time series | **ICA** artifact/source separation → **supervised ML** or **Kalman** state tracking | Can independent components separate a stimulus-locked source from blink/line-noise artifacts? |
| **Physiological dynamics** | ECG/respiration/glucose, wearable streams | **wavelets/CS** denoise & compress → **DMD/SINDy** dynamics | What low-dimensional dynamics govern the beat-to-beat signal, and are the modes stable across segments? |
| **Medical imaging** | MRI slices, MedMNIST tiles | **robust PCA** background/foreground split → **autoencoder** or **classifier** | Does a low-rank + sparse decomposition isolate lesion-like structure that a classifier can then use? |
| **Epidemic / population** | Case/incidence time series (e.g. SIR-like) | **SVD/PCA** of the trajectory ensemble → **DMD/SINDy** to recover rates | Can we recover epidemic growth/decay modes from noisy incidence, and how does noise degrade them? |
| **Sequence / foundation** | Embeddings from a sequence/foundation model | **PCA/ICA** of the embedding space → **supervised ML** probe | Do learned embedding axes carry a biologically meaningful, linearly decodable signal? |

Each track has ground-truth options: a synthetic generator in
`ddm4bio.datasets.synthetic` with a known answer, plus a held-out real split.
Use the synthetic one first to prove your pipeline is correct, *then* run it on
real data.

---

## Timeline & deliverables

Three graded milestones. Dates are set on the course schedule; the deliverables
are fixed.

| Milestone | Deliverable | What it must contain |
|-----------|-------------|----------------------|
| **1 — Proposal** | `proposal.md` (1–2 pages) | The biomedical question; chosen track; the two methods (one per group) named as `ddm4bio` functions; the dataset and its access/licence; the **ground-truth validation plan**; the risk you are most worried about. |
| **2 — Checkpoint** | Running pipeline on **synthetic/ground-truth data** + short progress note | End-to-end pipeline that loads → QC → both methods → interpretation on a fixture with a *known* answer, plus the validation metric showing the pipeline recovers it. This is the "prove it works before you trust it" gate. |
| **3 — Final** | Final report + code + presentation | `report.md` filled from the template (all mandated sections), the reproducible pipeline (`make data && make run && make report` from a clean clone), and a short presentation of framing, method, result, confidence, and limitations. |

**Reproducibility bar for the final:** a grader clones your repo, runs
`make data && make run`, and reproduces every headline number in your report
(within a documented tolerance), using pinned seeds and pinned dependencies.

---

## Getting started

```bash
cp -r capstone/template capstone/submissions/<your-name>
cd capstone/submissions/<your-name>
make data      # materialize demo/synthetic inputs
make run       # load -> QC -> decomposition + dynamics/learning -> interpretation
make report    # check the report has every mandated section
```

The template runs end-to-end on bundled synthetic data with only the core
`ddm4bio` stack installed — no network, no heavy extras. Swap in your dataset
and your two methods, keep the section structure, and keep it reproducible.

Read [`rubric.md`](rubric.md) before you start: the **non-negotiable standards**
(QC before conclusions, no silent method swaps, ground-truth validation before
trusting real data, reproducibility) are pass/fail gates, not point deductions.
