# Data-Driven Methods for the Life Sciences

*From linear algebra to dynamics to deep learning, on real biomedical data*

Welcome. This is an applications-first course in modern computational science, told
through the lens of biology and medicine. Rather than starting from abstract theory
and hoping the examples land later, we start from problems a life scientist actually
cares about — reading out cell types from single-cell data, denoising a noisy ECG,
forecasting an epidemic, picking a handful of biomarkers out of thousands — and then
build up the mathematical and computational machinery needed to attack them.

## What this course is

The last few decades have quietly rewired how quantitative science gets done. Where we
once wrote down governing equations from first principles and solved them, we now
increasingly *learn* structure, dynamics, and predictions directly from measurements.
This course is a biomedical re-casting of that shift. It teaches the methods of
data-driven modeling and scientific computation — linear algebra done at scale,
dimensionality reduction, signal and image processing, machine learning, dynamical
systems discovered from data, and deep learning — with every technique motivated,
demonstrated, and practiced on a mix of open clinical data and synthetic ground-truth
fixtures where the true answer is known.

## Two ways of reasoning

It helps to name the two intellectual traditions the course sits between.

The first is **deductive**, or model-driven, reasoning. Here you begin with known
mechanism — a rate law, a conservation principle, a differential equation — and deduce
what the system must do. This is the classical scientific-computation worldview, and it
is powerful when the governing rules are known and trustworthy.

The second is **inductive**, or data-driven, reasoning. Here you begin with
measurements and let the data reveal the patterns, coordinates, and dynamics, often
without committing to a mechanism up front. This is the worldview behind modern machine
learning and statistics.

Neither is complete on its own, and the most interesting work braids them together. But
because data are now abundant in biology and mechanistic models often are not, this
course leans deliberately toward the **data-driven half** of the spectrum. We treat
first-principles modeling with respect, and we reference it, but our center of gravity
is learning from data.

## Who it is for

Quantitatively-trained people — CS, math, physics, and engineering students, working data
scientists, and computationally-inclined biologists and clinicians who already have linear
algebra, statistics, and Python. It also works as a **refresher for practising ML
scientists and engineers moving into life-sciences and biomedical work**: you get the
biomedical framing, the datasets, and the domain-specific pitfalls — base-rate traps,
double-dipping, real-data QC — mapped onto methods you may already know.

No biology background is required. Every biomedical example is introduced from scratch,
in plain language, with just enough domain context to make the modeling choices sensible.

## How it works

The course is **self-paced and evergreen** — there is no cohort, no start date, and no
deadline but your own. Learning happens by doing:

- **Every lab runs in the browser.** Each computational lab opens in Google Colab with a
  single click, so there is nothing to install and nothing to configure. You write and
  run real code against real data from the first week.
- **Problem sets are auto-graded.** You get immediate, objective feedback, which means
  you can iterate quickly and know when you have actually gotten something right.

The sequence of topics follows the structure of J. N. Kutz's textbook *Data-Driven
Modeling & Scientific Computation: Methods for Integrating Dynamics of Complex Systems
and Big Data* (2nd ed.), which we recommend as companion reading; all explanations,
code, and examples here are original and this course does not reproduce that text.
