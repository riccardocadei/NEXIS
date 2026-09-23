# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to work in this repo

### Session role
- The main session is an ORCHESTRATOR, not a worker. Keep its context clean:
  spawn subagents for substantial (or even medium) work, read back their reports,
  decide, repeat.
- Never do large amounts of exploration, implementation, or launching directly in the
  main session. Direct work is fine only for trivial housekeeping (a git branch, a
  one-line check).

### Subagent policy
- ALL real work (exploration, implementation, running/launching experiments, collecting
  results) goes to subagents.
- Model choice by task difficulty:
  - **Opus** — design, planning, implementation of nontrivial code, debugging.
  - **Sonnet** or **Haiku** — surveys/recon, mechanical edits, launching and monitoring
    jobs, collecting results into tables.
  - **Never Fable subagents.**
- Give workers self-contained prompts: the goal, the hard rules (code style, honesty
  rules), pointers to files/plan, and an explicit "report back" spec with measured
  numbers, so the orchestrator never has to re-explore.
- Treat subagents as unreliable employees. Give them specific, realistic tests and ask
  for confirmation that they implemented what you asked.

### Communication
The reader is a human being, not an LLM. Mind their cognitive load and attention limits;
they do not remember everything from earlier in the session.

- **Self-contained messages.** Restate the necessary context, file names, and objective
  in each message rather than assuming previous conversation is top-of-mind.
- **No unexplained jargon.** Define acronyms and shorthand in context. Simple, direct
  technical English — no convoluted sentences, no unnatural prose, no dead prose.
- **Lead with the outcome**, then the detail.

### Interaction style
- Interactive by default: decisions are made WITH the user (AskUserQuestion at real
  forks), then workers execute what was decided.
- Check in at every gate before spending compute.

### Standing execution rules (research)
- State the session goal explicitly and keep it in view; escalate methods until the goal
  is met rather than settling for a weaker result.
- Honest reporting at every step: only measured numbers, identity-at-init / no metric
  gaming, read logs before claiming success, fail loud.

## What this is, and where it is now

Research code for the NEXIS paper (*From Tokens to Policy: Causal and Interpretable
Heterogeneous Treatment Effects Identification*). It is a **method implementation plus
three applications**, not a library with a public API — there is no test suite, no linter
config, and no CI. Correctness is checked by re-running pipelines and comparing against
the numbers in `docs/*_experiment_brief.md`.

Two distinct kinds of work are live, and they have different rules:

1. **NEXIS paper — done.** The paper is online. What remains is cleanup and folding in
   the rebuttal material sitting uncleaned on the `rebuttal-NeurIPS` branch. The method
   (`src/method/nexis.py`) and the CelebA/Uganda applications are effectively **frozen**:
   touch them to clean, reproduce, or answer a reviewer, not to redesign. Any change that
   moves a published number needs to be deliberate and flagged.
2. **Ghana LEAP-1000 deep-dive — active.** This is the current focus and where new work
   lands, on the `ghana` branch (~50 commits ahead of `main`). Expect to be working in
   `src/apps/ghana/`, `src/apps/covariates.py`, `data/ghana/README.md`, and
   `docs/ghana_experiment_brief.md`. Default here is to extend, not preserve.

## Environment

There is no `requirements.txt` / lockfile. The conda env every SLURM script hardcodes is:

```bash
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
```

Use that interpreter (or `PYTHON=... bash scripts/...` to override) rather than bare
`python`. `pip install -e .` installs the repo as the `nexis` package (`src/` layout),
but app modules are normally run as scripts from the repo root, and they insert the repo
root into `sys.path` themselves.

## Running things

Pipelines are SLURM batch scripts under `scripts/<app>/` that also run fine under plain
`bash` from the repo root. Logs go to `logs/`.

```bash
# Ghana — the active application
bash scripts/ghana/slurm_train_sae.sh
bash scripts/ghana/slurm_stats.sh          # NEXIS only (interpret.py --no-interpret)
bash scripts/ghana/slurm_interpret_7b.sh   # + Qwen-VL 7B   (slurm_interpret.sh = 72B)
bash scripts/ghana/run_figure_neural.sh

# Uganda / CelebA — reproduce-only
bash scripts/uganda/run.sh --models=prithvi --all-outcomes
bash scripts/uganda/reanalyze.sh --models=prithvi --all-outcomes   # skip embed/SAE
bash scripts/celeba/submit_embed.sh && bash scripts/celeba/submit_sae.sh
bash scripts/celeba/submit_experiment.sh && python src/apps/celeba/figure_main.py
```

Individual modules are directly runnable for iteration, which is the usual Ghana loop:
`$PYTHON src/apps/ghana/interpret.py --mode both --alpha 0.05 --no-interpret`.

`scripts/uganda/run.sh` is the reference for the step-skipping convention: a step is
skipped when its output file already exists unless `--overwrite[=step,...]` is passed.
Preserve that idempotence when adding steps — these get re-run constantly and the GPU
steps are expensive.

GPU is required only for embedding extraction, SAE training, and VLM interpretation. The
NEXIS selection step itself is CPU-only and fast.

## Architecture

Four layers, in dependency order:

**`src/method/nexis.py`** — the method, and the only file that carries the paper's
statistical claims. `nexis(y, t, w, ...)` runs forward(-backward) selection over candidate
effect modifiers, each round gated by a conditional interaction test (`test="linear"`:
parametric t-test with optional CR1S/HC1 SEs; `test="GCM"`: R-learner pseudo-outcome with
cross-fitted nuisance), a Bonferroni/BH multiplicity gate, and a spectral-gap threshold
`rho`. Returns a `SelectionResult`. App-agnostic, with no imports from `src/apps/`.
Frozen — see above.

**`src/causality/estimation.py`** — generic HC1-robust OLS, ATE, and the GATE/CATE split
logic (`classify_feature`, `feature_gate`) used to report per-modifier effects.

**`src/apps/covariates.py`** — the `Covariate`/`Dataset` registry, and the main surface
the Ghana deep-dive extends. Read its module docstring before touching covariate
definitions; it is the design document for the data layer. Key points:
- **`w` is one flat pool.** No household-vs-community, no W-vs-Z split at the API level.
  `Dataset.X` is a single `(n, p)` DataFrame; `level` is metadata on each column, not a
  structural fork. The old two-matrix `nexis(y, t, w, z)` API was deliberately retired
  ("refactor: retire the household=W/community=Z naming, W is everything") — do not
  reintroduce a second matrix.
- Six tagging axes: `level`, `origin`, `support`, `domain`, `access`, `timing`.
- **`origin` is the only axis with behavioral consequences.** `nexis()` reads
  `X.attrs['origin']` and screens every non-`learned` column in a cheaper preliminary
  phase (its own smaller Bonferroni pool) before the joint round where everything competes
  symmetrically. `X.attrs['cluster']` supplies CR1S cluster IDs. Both flow from
  `Dataset.__post_init__`, so passing `dataset.X` to `nexis()` is all that's needed.
- `support` drives `Covariate.binarize` (binary/sparse split at 0, else at the median).

**`src/apps/{ghana,uganda,celeba,synthetic}/`** — one package per application, each the
same pipeline shape: `download_*.py` → `extract_satellite_features.py` → `train_sae.py`
(TopK SAE, Gao et al. 2024) → NEXIS selection → `interpret.py` (VLM labels the selected
neurons a posteriori) → `figure_*.py` / `plot_*.py`.

Only Ghana is migrated to the `Covariate` registry. Uganda and CelebA still have their own
differently-shaped `data.py` and pass plain arrays. When editing shared code, check all
three call sites: Uganda/CelebA depend on the untagged-ndarray path staying
behavior-identical (single flat search, no staging), and they back published numbers.

## Ghana conventions

The deep-dive's process, mostly about not letting the covariate pool drift:

- **`data/ghana/README.md` is a source registry that must be updated whenever a source is
  added, rejected, or changed.** Every `Covariate.source` string matches a section there.
  It documents *sources* only — derived artifacts (embeddings, SAE weights, figures) live
  under `results/ghana/` and are deliberately not tracked in it.
- Add one column-producing block per source inside
  `external_data.py::load_effect_modifiers` rather than inventing per-source merge logic.
  Each source gets a matching `download_<source>.py`.
- A covariate is excluded from the pool only when treatment could plausibly have caused it
  (post-treatment/collider risk). Survey covariates are baseline-2015 only; exogenous
  sources (rainfall, prices) carry no such risk regardless of measurement year, which is
  what the `timing` axis records.
- **Rejected and reverted sources get documented too**, with the reasoning — see the
  `docs(ghana): reject ...` and `revert(ghana): drop ...` commits. The negative results
  are part of the deep-dive's record; record *why*, not just *that*.
- `data.py` imports `covariates.py` via its fully-qualified `src.apps.covariates` path on
  purpose: a bare `from covariates import ...` creates a second module object with
  distinct `Enum` classes, silently breaking every `c.level is Level.HOUSEHOLD` identity
  check. Keep qualified imports.

## Docs, notebooks, and outputs

- `docs/ghana_experiment_brief.md` and `docs/uganda_experiment_brief.md` are the
  frozen-numbers briefs behind the paper. If a change moves a number that appears there,
  update the brief in the same change.
- `docs/index.html` + `docs/assets/` is the published project website; figure scripts
  write into it, including the Ghana covariate-flows sankey
  (`docs/assets/ghana_covariate_flows.html`).
- `notebooks/*.ipynb` are interactive views over the same modules and assume a
  `notebooks/` working directory (relative paths like `../data/ghana`).
- Results land in `results/<app>/...` (Uganda/CelebA nest as `<model>_<sae_dim>/<outcome>/`).
  Treat `results/` and `data/` as regenerable outputs, not sources.

## Commit style

Conventional commits, scoped by app: `feat(ghana):`, `fix(ghana):`, `docs(ghana):`,
`refactor:`, `revert(ghana):`.
