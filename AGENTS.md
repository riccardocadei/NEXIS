# AGENTS.md

Guidance for coding agents (Claude Code, Codex) working in this repository.

**This file is local-only** (untracked, listed in `.gitignore` since 2026-09-27). The
GitHub repo `riccardocadei/NEXIS` is public and shows the author names, while the paper is
under double-blind review. Every tracked file, commit message, tag and branch name must
therefore stay free of the venue (ICLR, NeurIPS), the review process (rebuttal, anonymous
submission, reviewers, OpenReview) and the private `paper/` folder. Cite paper items
neutrally ("paper Appendix C", "Table tab:ghana_temporal", "the paper's main text").
Check before every push:
`git grep -n -i -E "iclr|neurips|rebuttal|anonymous|double-blind|reviewer|openreview|paper/"`.
This file may mention them freely.

## Where we are

Research code for the NEXIS paper (*From Tokens to Policy: Causal and Interpretable
Heterogeneous Treatment Effects Identification*): the method plus three applications
(CelebA semi-synthetic, Uganda YOP, Ghana LEAP 1000). There is no test suite, linter or CI.
Correctness is checked by re-running pipelines and comparing against the numbers in the
per-application READMEs `src/apps/{celeba,uganda,ghana}/README.md` (formerly
`docs/*_experiment_brief.md`), which document the ICLR'27 paper's numbers and their
sources.

The NeurIPS rebuttal branch is merged into `main` and deleted. The pre-cleanup state is
tag `pre-cleanup-2026-09` (commit `2da8785`; it was called `neurips-rebuttal-final` until
2026-09-27, renamed because tags are public). `paper/iclr27` is the final version of the
paper and the ground truth for every number. The method (`src/method/nexis.py`) and the
CelebA/Uganda applications are **frozen**: touch them only to clean or reproduce, and flag
any change that moves a published number. Machine-specific notes (backups, local paths)
are in the "Local notes" section below.

Plan, in order:

1. Update the project website (`docs/index.html`, `docs/assets/`, `animations/`). Done
   (branch `website` merged in `4066243`).
2. Merge the `ghana` branch (the LEAP 1000 deep dive, ~50 commits ahead). That becomes
   the active project, and its detailed conventions get added back here. See "Ghana
   merge" below: it will not be a clean merge.
3. Make `src/apps/celeba/benchmark/` the canonical CelebA pipeline. The move from
   `iclr/` to that neutral path is done (2026-09-27). **TODO after the ghana merge:**
   switch the package from its own `nexis/` copy of the method to the single
   `src/method/nexis.py` (the ghana branch changes `nexis.py`, so do it after the merge),
   check that `main` and `violation` still reproduce the paper, then check the other
   blocks.

### Ghana merge

The 2026-09-27 cleanup on `main` causes these conflicts with the `ghana` branch:

- `AGENTS.md`: modify/delete. `main` untracked it; `ghana` adds ~193 lines of Ghana
  conventions. Merge those lines **by hand into this local file**, then resolve with
  `git rm --cached AGENTS.md` (keep it untracked).
- `notebooks/*.ipynb`: modify/delete. `main` removed the notebooks (archived); resolve by
  deleting them again.
- `README.md` and `.gitignore`: both changed on both sides; merge by hand and keep the
  public wording neutral (no venue, no notebooks, `AGENTS.md` ignored).
- `src/method/nexis.py` and `src/apps/celeba/experiment.py`: `main` only changed a
  docstring line and a comment outside the ghana hunks, so they should merge cleanly.
- `docs/assets/ghana_covariate_flows.html` (added on `ghana`) contains 4 matches of the
  scrub words (probably inside its bundled script): check and clean before pushing.
  `ghana`'s `docs/*_experiment_brief.md` are unchanged since the fork, so the moves to
  `src/apps/*/README.md` carry over; the Ghana findings of the branch belong in
  `src/apps/ghana/README.md` and `data/ghana/README.md`.
- After merging, rerun the scrub `git grep` above before pushing.

## The paper

`paper/` is a separate git clone of the Overleaf project. It is ignored by this repo and
has its own history. Follow `paper/AGENTS.md` there. Sync it with `git overleaf pull` and
`git overleaf push`, and only when the user asks. From the code side, the only thing to
touch there is `paper/iclr27/figures/` (the ICLR 2027 version, the only one left), and
only when the user asks: regenerate a figure with its plotting script and commit it in the
paper repo. `README.md` and the per-application READMEs map every figure to its script.

## How to work

- **Orchestrate, don't grind.** Keep the main session's context clean: hand substantial
  work (exploration, implementation, launching jobs, collecting results) to subagents
  with self-contained prompts. Each prompt covers the goal, the hard rules, pointers to
  files and what to report back. Use Opus for design, nontrivial code and debugging, and
  Sonnet or Haiku for recon, mechanical edits and monitoring jobs. Never use Fable
  subagents. Verify what subagents claim.
- **Decide with the user.** Ask at real forks, and check in before spending compute.
- **Honest reporting.** Only measured numbers, read the logs before claiming success,
  fail loudly.
- **Write for a human.** Each message should stand on its own: lead with the outcome,
  restate the needed context and file names, define acronyms, and use plain technical
  English.

## Environment and running

Use the conda interpreter that the SLURM scripts hardcode, not bare `python`:

```bash
PYTHON=/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3
```

Pipelines are SLURM scripts under `scripts/<app>/`. They also run with plain `bash` from
the repo root, and logs go to `logs/`. See `README.md` for the per-app commands. Steps are
skipped when their output exists unless `--overwrite` is passed (see
`scripts/uganda/run.sh`). Keep that idempotence, because the GPU steps are expensive.
`results/` and `data/` are regenerable outputs, not sources, except the SAE checkpoints
and codes behind published coordinate indices (restore them from the backup, see "Local
notes").

## Code map

- `src/method/nexis.py`: the method, and the only file that carries the paper's
  statistical claims. It is app-agnostic. Frozen. Its defaults are the NeurIPS ones; the
  paper's algorithm is `nexis(backward=False, terminal_filter=True)` (see `README.md`).
- `src/causality/estimation.py`: HC1-robust OLS, ATE and GATE/CATE reporting.
- `src/train/`: the TopK SAE (`overcomplete`) used by the CelebA pipeline.
- `src/apps/{celeba,uganda,ghana}/`: one pipeline per application
  (download → embed → SAE → NEXIS → VLM interpretation → figures), each with a
  `README.md` holding the paper's numbers, their source files and the design choices. If a
  change moves a number, update that README in the same commit. When editing shared
  code, check every call site: Uganda and CelebA back published numbers.
  (`src/apps/synthetic/` and `notebooks/` were removed on 2026-09-27; copies in the backup.)
- `src/apps/celeba/benchmark/`: self-contained reproduction package for every CelebA
  experiment (its own `nexis/` copy of the method, `python run.py <block>` from that
  folder; outputs to `data/`, `results/`, `logs/` next to it unless `--data-dir` /
  `--results-dir`). Only `main` and `violation` are checked against the paper so far. The
  paper's runs used `--results-dir results/celeba/benchmark/results` (see "Local notes").
- `scripts/realworld_*.py`: the application runs behind the paper's numbers.
  `realworld_final_runs.py` → `results/realworld_final/report.json` (Uganda and Ghana
  tables), `realworld_uganda_groupcluster.py` (Uganda multilevel limitation); both build on
  `realworld_clustered_nexis.py`.
- `animations/`: Manim scenes for the website videos (`docs/assets/`).
- `docs/`: the GitHub Pages website only (`index.html`, `assets/`).

## Commits

Conventional commits, scoped by app: `feat(ghana):`, `fix(uganda):`, `docs:`,
`refactor:`, `revert(...)`. Commit messages are public: no venue or review wording.

## Local notes

Machine-specific notes for this checkout on the ISTA cluster (formerly the untracked
`LOCAL.md`, merged here on 2026-09-27; `LOCAL.md` stays in `.gitignore`).

### Backup of untracked outputs

Path: `/fs3/group/locatgrp/rcadei/nexis-archived_exp/` (its `README.md` describes the
contents; `MANIFEST.sha256` has a per-file SHA-256 manifest).

Made 2026-09-26 at tag `pre-cleanup-2026-09` (then named `neurips-rebuttal-final`,
commit `2da8785`), before the repo cleanup. Copy-only: nothing in the repo was moved or
deleted when it was made. The 2026-09-27 cleanup then deleted exploratory and rebuttal
output folders from the working tree (e.g. `results/celeba/{experiment_rep,
experiment_type1, experiment_v2_r0, appendix*, agreement_b1, terminal_filter_rebuttal,
experiment_tmp, experiment_resample_a0, sae_dinov2_*}`, `figures_v2/comparison.md`); their
copies are in the backup.

What is there:

- `data/`, `results/`, `logs/`: copies of the untracked repo folders (CelebA, CelebA
  replica, Uganda, Ghana inputs, embeddings, SAE codes, all experiment outputs, SLURM logs).
- `cache/`: `~/.cache/nexis_cert`, `nexis_cap`, `nexis_pcm_probe` (NeurIPS rebuttal
  terminal-filter / opt2-certify code and ablation CSVs, e.g. `nexis_cert/run_all6.py`,
  cited in `src/apps/celeba/ablation_rho_filter.py`; its CSVs were
  `results/celeba/terminal_filter_rebuttal/all6_*.csv`; `nexis_cap` is the link layout
  used to run the benchmark package, then `iclr/`, on the original CelebA artefacts).
- `untracked_code/`: `src/apps/celeba/alignment_screening.py`,
  `scripts/celeba/submit_alignment_screening.sh`.
- `notebooks/old/`, `notebooks/results/`, `tweet/`, and a pre-existing `paper/NeurIPS'26/`.
- Not copied: `models/` (13 GB GeoChat weights) and `animations/media/` (Manim cache).

### Why it matters

GPU embedding and SAE training are not bit-reproducible: a retrained SAE gives other
coordinate indices than the ones the paper quotes (5348, 5537, 533, 3821, ...). To
reproduce the paper exactly, restore these files instead of retraining:

- CelebA: `results/celeba/sae_siglip_k{5,20}.pt`, `data/celeba/embeddings/sae*_k*.npy`;
  replica `results/celeba/resample_b1/`, `data/celeba_resample_b1/`.
- Uganda: `results/uganda/prithvi_l5_1024/` (`sae_model.pt`, `individual_features.npz`,
  `site_features.npz`, the per-outcome `nexis_result.json` published sets).
- Ghana: `data/ghana/satellite/{sae_model.pt, sae_activations.npy, spectral_indices.csv}`,
  `results/ghana/codes/nexis_fwer_crve/result.json`.
- Paper-number outputs: `results/realworld_final/`, `results/realworld_uganda_groupcluster/`,
  `results/celeba/experiment_v2*/`, `results/celeba/figures_v2/`,
  `results/celeba/benchmark/`, and the `ground_truth.json` files under
  `results/celeba/experiment{,_r3,_resample_b1,_ushape}/`.

`results/celeba/benchmark/` is the benchmark package's run of the paper (`data/`,
`results/`, `logs/`); it was `results/celeba/iclr_local_runs/` until 2026-09-27 and keeps
that old name in the backup. `src/apps/celeba/run_statistics.py` reads its
`results/runs/main/`. Its `data/celeba/*` symlinks are relative and dangle (they were made
inside `iclr/data/`); to rerun the package on the original artefacts use
`--data-dir ~/.cache/nexis_cap/data`, e.g. (from `src/apps/celeba/benchmark/`)
`python run.py violation --data-dir ~/.cache/nexis_cap/data --results-dir
../../../../results/celeba/benchmark/results`. Checked 2026-09-27: `main --smoke` and
`violation --smoke --jobs 1` run in ~2 min each on the login node with that data dir.

`data/ghana/survey/` (LEAP 1000) is restricted data: do not redistribute.

Symlinks under `cache/nexis_cap/data/` (and `~/.cache/nexis_cap/data/`) are absolute and
point back into this repo; they dangle if the repo's `data/` or `results/` are cleaned
(the targets are in the backup's `data/` and `results/`).

### Other local paths

- Python: `/nfs/scistore19/locatgrp/rcadei/.conda/envs/crl/bin/python3` (Python 3.11.8,
  the versions pinned in `pyproject.toml`).
- CelebA Hugging Face cache: `~/.cache/huggingface/datasets/flwrlabs___celeba` (revision
  `2d738f5…`, the one pinned in `src/apps/celeba/embed.py` and
  `src/apps/celeba/benchmark/config.py`).
