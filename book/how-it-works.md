# How This Course Works

Welcome. This is a self-paced, evergreen course, which means there is no cohort to wait for and no start date to miss. You enroll whenever you're ready, and you move at whatever pace fits your life. Some learners finish a week in an evening; others spread it across a fortnight. Both are fine. The material doesn't expire, and neither does your access to it.

Here's what a typical week looks like and how everything fits together.

## Read, then build

Each week begins with a lesson. Read it first — it introduces the ideas in plain language, motivates why the method matters for real biological data, and sets up the reasoning you'll need before you touch any code. Skimming works less well here than in most places, because running the code yourself assumes you've absorbed the conceptual thread.

Once the lesson clicks, you run it. The lesson *is* a notebook: the same page you just read opens in **Google Colab** with a single click, so you can run and modify every example yourself. There is nothing to install, no environment to configure, no dependency conflicts to debug at midnight. Colab hosts the notebook in your browser, and for the deep-learning week it gives you **free GPU access**, so you can train real models without owning a GPU or paying for cloud time. You write and run code in the very place you read the explanation of what it does.

## The `ddm4bio` library

At the top of every lesson notebook, you'll import the course's own Python library, `ddm4bio`. Think of it as the vetted toolkit that lets you focus on the science instead of reinventing plumbing. It gives you:

- **Data generators with known ground truth** — synthetic datasets where you already know the right answer, so you can check whether your method actually recovers it.
- **Decomposition, QC, and validation tools** — the reliable building blocks for cleaning data, breaking signals into components, and testing whether a result holds up.
- **Interpretation helpers** — the `ddm4bio.interpret` module, which makes the good habits below almost automatic.

Because every learner imports the same vetted building blocks, your results are comparable to everyone else's, and you spend your energy on understanding rather than on boilerplate.

## The problem sets

There are **seven problem sets**, delivered through **GitHub Classroom**. When you accept an assignment, you get your own private copy of a repository to work in. When you push your solution, the course's **pytest-based autograders** run automatically and give you **immediate feedback** — you see exactly which checks passed and which need more work, so you can iterate right away instead of waiting on a human grader. Pass all seven and you earn a **completion certificate**.

The autograders test the substance of your work, not just its surface. They're built from the same validation philosophy the course teaches.

## The standards we hold — and why they matter

The lessons and the problem sets alike hold you to a small set of non-negotiable standards. These aren't bureaucratic hoops. Each one exists because skipping it is exactly how real scientific analyses go wrong. Getting them into your fingers is arguably the most valuable thing this course can give you.

**Quality control before results.** You run QC on your data before you interpret anything. Raw data is almost never clean — there are dropouts, batch effects, and artifacts hiding in it. A beautiful result built on unchecked data is a beautiful mistake, and it's the kind that survives peer review and wastes years. QC first means you find the problem before it finds you.

**Honest method labeling.** You call each method by its true name, always. If you ran a PCA, you say PCA — not "clustering," not something that sounds fancier. Mislabeling a method makes results impossible for anyone (including future you) to reproduce or trust, and it quietly corrupts the literature. Precision in naming is precision in thinking.

**A stated, calibrated confidence with named limitations.** Every result you report carries an explicit confidence level and a list of the specific things that could undermine it. "It works" is not a finding. Real science is quantified uncertainty: a result you're 60% sure of is genuinely useful *if you say so*, and dangerous if you present it as certain. Naming your limitations is how you stay honest and how others know where to push. The `ddm4bio.interpret` helpers make this habit easy — they give you a structured way to attach a confidence and its caveats to any result, so the honest version is also the convenient version.

**Ground-truth validation before trusting real data.** Before you trust a method on real biological data, you prove it on synthetic data where you already know the answer. If your method can't recover a signal you planted yourself, it certainly can't be trusted to find one you didn't. This is why the library ships with ground-truth data generators: validation isn't an afterthought, it's the first step. For exactly when this course uses real versus synthetic data — and why both belong — see **[Real Data and Synthetic Ground Truth](data-and-ground-truth.md)**.

**Full reproducibility.** Anyone should be able to re-run your analysis and get your result. Fixed random seeds, recorded versions, code that runs top to bottom. A result that can't be reproduced isn't a result yet — it's an anecdote. Reproducibility is what separates science from storytelling.

## In short

Enroll anytime. Read each lesson, open it in Colab with one click, import `ddm4bio`, and build. Submit the seven problem sets through GitHub Classroom, get instant autograded feedback, and earn your certificate. Along the way, practice the habits that make analysis trustworthy — QC first, honest labels, calibrated confidence with named limitations, ground-truth validation, and full reproducibility — until they stop feeling like rules and start feeling like how you think.

Welcome aboard. Start whenever you like.
