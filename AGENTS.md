# AGENTS.md

Guidance for coding agents (Claude Code, Codex) working in this repository.

## Where we are

Research code for the NEXIS paper (*From Tokens to Policy: Causal and Interpretable
Heterogeneous Treatment Effects Identification*): the method plus three applications
(CelebA semi-synthetic, Uganda YOP, Ghana LEAP 1000). There is no test suite, linter or CI.
Correctness is checked by re-running pipelines and comparing against the numbers in
`docs/*_experiment_brief.md`, which document the ICLR'27 paper's numbers and their sources.

The NeurIPS rebuttal branch is merged into `main` and deleted (pre-cleanup state: tag
`neurips-rebuttal-final`). `paper/ICLR'27` is the final version of the paper and the
ground truth for every number. The method (`src/method/nexis.py`) and the CelebA/Uganda
applications are **frozen**: touch them only to clean or reproduce, and flag any change
that moves a published number. Machine-specific notes (backups, local paths) are in the
untracked `LOCAL.md`.

Plan, in order:

1. Update the project website (`docs/index.html`, `docs/assets/`, `animations/`).
2. Merge the `ghana` branch (the LEAP 1000 deep dive, ~50 commits ahead). That becomes
   the active project, and its detailed conventions get added back here.
3. Fold `iclr/` into `src/apps/celeba` under a neutral name, running on the single
   `src/method/nexis.py`, as the canonical CelebA pipeline.

## The paper

`paper/` is a separate git clone of the Overleaf project. It is ignored by this repo and
has its own history. Follow `paper/AGENTS.md` there. Sync it with `git overleaf pull` and
`git overleaf push`, and only when the user asks. From the code side, the only thing to
touch there is `paper/ICLR'27/figures/` (the ICLR 2027 version, the only one left), and
only when the user asks: regenerate a figure with its plotting script and commit it in the
paper repo. `README.md` and the briefs in `docs/` map every figure to its script.

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
and codes behind published coordinate indices (restore them from the backup, see `LOCAL.md`).

## Code map

- `src/method/nexis.py`: the method, and the only file that carries the paper's
  statistical claims. It is app-agnostic. Frozen. Its defaults are the NeurIPS ones; the
  paper's algorithm is `nexis(backward=False, terminal_filter=True)` (see `README.md`).
- `src/causality/estimation.py`: HC1-robust OLS, ATE and GATE/CATE reporting.
- `src/train/`: the TopK SAE (`overcomplete`) used by the CelebA pipeline.
- `src/apps/{celeba,uganda,ghana,synthetic}/`: one pipeline per application
  (download → embed → SAE → NEXIS → VLM interpretation → figures). When editing shared
  code, check every call site: Uganda and CelebA back published numbers.
- `scripts/realworld_*.py`: the application runs behind the paper's numbers.
  `realworld_final_runs.py` → `results/realworld_final/report.json` (Uganda and Ghana
  tables), `realworld_uganda_groupcluster.py` (Uganda multilevel limitation); both build on
  `realworld_clustered_nexis.py`.
- `iclr/`: self-contained reproduction package for every CelebA experiment (its own
  `nexis/` copy of the method, one command per block). Only `main` and `violation` are
  checked against the paper so far; to be folded into `src/apps/celeba` (plan step 3).
- `animations/`: Manim scenes for the website videos (`docs/assets/`).
- `docs/*_experiment_brief.md`: the ICLR'27 numbers, their source files and the design
  choices behind them. If a change moves one, update the brief in the same commit.

## Commits

Conventional commits, scoped by app: `feat(ghana):`, `fix(uganda):`, `docs:`,
`refactor:`, `revert(...)`.
