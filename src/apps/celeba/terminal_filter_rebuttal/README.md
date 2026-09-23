# Terminal filter: NeurIPS rebuttal experiments (archive)

These scripts are verbatim copies of the rebuttal experiments run on 26–27 July 2026
from `~/.cache/nexis_cert/`, outside the repo. They are kept here for provenance.
They are not wired into the pipelines: each script writes its CSVs next to itself
(`HERE`). Copies of the outputs are in `results/celeba/terminal_filter_rebuttal/`.

- `certify.py`: the two post-hoc certifications that were compared. Option 1 is a
  global pathwise Bonferroni correction. Option 2 is the subset-robust filter, which
  keeps j only if p(j | A) <= alpha/m for every A in S~\{j}. Option 2 is what the
  rebuttal proposed; it now lives in `nexis(..., terminal_filter=True)`.
- `run_all6.py` (launched by `job4.sh` / `job4b.sh`): the full CelebA grid with the
  variants base, opt2, pathwise, scr, scr_opt2 and scr_pathwise. Output:
  `all6_*.csv`. Macro-averaging over k20/sae, k20/sae_precode and k5/sae (126
  configs x 50 seeds) gives the rebuttal table: precision/recall/IoU of
  0.731/0.732/0.684 for base and 0.765/0.722/0.707 for opt2, with 538 runs
  improved, 5,657 unchanged and 105 degraded.
- `run_all4.py`, `run_certified_all.py`, `opt2_*.py`, `rand_gate.py`: earlier
  exploratory variants (pooled variance, screening, randomization-calibrated gate).

The follow-up ablation that crosses the filter with rho, and with or without the
backward step, is `src/apps/celeba/ablation_rho_filter.py`.
