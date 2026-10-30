# NEXIS: Neural EXposure Interaction Search

**From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification**
· Riccardo Cadei, Frank Otchere, Nyasha Tirivayi, Gustavo Angeles Tagliaferro, Falco J. Bargagli-Stoffi, Francesco Locatello
· [arXiv](https://arxiv.org/abs/2606.17010) · [Website](https://riccardocadei.github.io/NEXIS/)

NEXIS finds *which* pre-treatment features drive the heterogeneity of a treatment effect.
Given an outcome, a binary treatment and a large pool of candidate features (survey
covariates, and for example the atoms of a sparse autoencoder (SAE) trained on
foundation-model embeddings of satellite images), it returns a small set of features that
characterizes the conditional average treatment effect (CATE), with family-wise error
control: a forward search with conditional interaction tests and Bonferroni gates, then a
terminal step that keeps a feature only if it stays significant given every subset of the
others. Learned features are then interpreted from their most and least active raw items
(for example by a vision-language model), and the result can guide how the treatment is
targeted next time.

![NEXIS pipeline](animations/assets/pipeline.gif)

This repository holds the `nexis` Python library (`src/nexis`) and, separately, the code
that reproduces the paper's three case studies (`src/apps`).

## Install

The package is not on PyPI. For library use (Python >= 3.11; needs numpy, scipy, pandas,
scikit-learn):

```bash
pip install "git+https://github.com/riccardocadei/NEXIS.git"
```

Only the `nexis` package is installed. The `lgbm` extra
(`pip install "nexis[lgbm] @ git+https://github.com/riccardocadei/NEXIS.git"`) adds
lightgbm for the nonparametric tests `test="GCM: lgbm"` and `"PCM: lgbm"`; `nexis.sae`
(TopK SAE training) needs the `gpu` extra.

To reproduce the paper, clone the repository and install the pipelines' extras, pinned to
the environment the paper's runs used (Python 3.11.8):

```bash
git clone https://github.com/riccardocadei/NEXIS && cd NEXIS
pip install -r requirements-paper.txt -e ".[apps,gpu,geo]"
# apps: CPU experiments and figures; gpu: embeddings, SAE training and VLM
# interpretation; geo: Google Earth Engine downloads and maps
```

`pyproject.toml` lists those versions as lower bounds. The application code in `src/apps`
is not installed: it runs from the checkout (`python src/apps/<app>/<script>.py` from the
repo root; each script puts `src/` on `sys.path`).

## Quickstart

Households in villages, a grant randomized by village within districts, candidates from a
household survey and from sparse features learned on village images. The runnable version,
with the synthetic data, is [`examples/quickstart.py`](examples/quickstart.py).

```python
import nexis

data = nexis.Dataset(
    units=units,          # one row per household (index household_id): village_id, district_id, grant
    sources=[
        nexis.Source(survey, name="baseline survey", level="household_id", time="pre",
                     support={"head_female": "binary", "hh_size": "count"}, origin="raw",
                     domain="demographics", access="restricted", modality="survey"),
        nexis.Source(atoms, name="satellite", level="village_id", time="pre",
                     support="sparse", origin="learned", domain="environment",
                     access="public", modality="image", raw=tiles,
                     interpreter=MyInterpreter()),
        nexis.Source(rain, name="rainfall", level="district_id", time="during",
                     support="continuous", origin="raw", domain="environment",
                     access="public", modality="raster"),
        nexis.Source(endline, name="endline survey", level="household_id", time="post",
                     support="continuous", origin="hand_crafted", domain="economy",
                     access="restricted", modality="survey"),
    ],
    design=nexis.Randomized(treatment="grant", assigned_at="village_id",
                            stratified_by="district_id"),
)
data.info()                  # levels, sources and their attributes, candidates, missing
                             # values, design checks

result = nexis.discover(data, outcome="consumption")
result.modifiers             # ['atom_3']
result.table()               # every candidate: its source and attributes, declared and
                             # detected level, clustering, support, p-value, status
interpretation = result.interpret(k=4)
interpretation               # label, most and least active ids of each learned modifier
interpretation.show()        # image grid per modifier (matplotlib)
```

An interpreter is any object with `describe(name, top, bottom) -> str`, called with the
covariate's name and the raw items (here image arrays; paths work too) of its most and
least active ids.

## Concepts

**Sources.** A `Source` is one data source (a survey, a satellite product, a climate or
conflict database): a table of numeric columns indexed by the ids of one level, measured
at one time relative to treatment. Keep one `Source` per data source; a list of them is
the single declaration of a study's data, from which the `Dataset` is built. Every
attribute is a keyword; all are required except `modality` (and `raw`, `interpreter`,
`render`, for learned columns only):

| Attribute | Values | Per | Role |
|---|---|---|---|
| `level` | the ids the rows are indexed by, e.g. `village_id` | source | **search**: joins the source onto the units; clustering |
| `time` | `historic`, `pre`, `during`, `post` | source | **search**: which columns are candidates |
| `support` | `binary`, `count`, `nonnegative`, `sparse`, `continuous` | column (or one for all) | **search**: support rule |
| `name` | the data source, e.g. `"CHIRPS"`, `"UNICEF household survey"` | source (unique) | interpretation |
| `domain` | a lowercase identifier: `demographics`, `housing`, `economy`, `accessibility`, `urbanization`, `environment`, `security`, `health`, ... | column (or one for all) | interpretation |
| `origin` | `raw`, `hand_crafted`, `learned` | column (or one for all) | interpretation |
| `access` | `public`, `restricted`, `proprietary` | source | bookkeeping |
| `modality` | `survey`, `image`, `raster`, `events`, `time_series`, `text`, or `None` | source | bookkeeping |

The three attributes that change the search:

- `level`: the table's index must be named so (or unnamed), and `level` must be the
  index name or a column of `units`: the source is broadcast onto the units through it.
  The level where each candidate actually varies is detected from the data and shown
  next to the declared one (`varies_at` in `data.info()` and `result.table()`); the
  clustering uses it (see the clustering rule). A covariate shared by groups that are not
  an existing level (a market catchment, a raster cell) gets its own level: an id column
  in `units` and a `Source` indexed by it.
- `time`: `historic` (long-run character predating the baseline, e.g. climate normals)
  and `pre` columns are candidates; `during` columns (measured while treatment ran) are
  excluded unless `discover(..., include_during=True)` (include them only if the
  treatment cannot have changed them); `post` columns are outcomes, never searched.
- `support`: `binary` ({0, 1}), `count` (integers >= 0), `nonnegative` (reals >= 0),
  `sparse` (>= 0, zero the most frequent value, e.g. SAE activations) or `continuous`
  (any finite real). Binary and sparse columns always get the support rule (below); the
  others get it when the data show a mass point.

The three that guide the interpretation of what is found, never the search: the source
`name`, the `domain` (what the variable is about; an open vocabulary) and the `origin`:
`raw` (as recorded by the source), `hand_crafted` (derived by the analyst from raw data,
e.g. an index or a distance) or `learned` (learned by a model, e.g. an SAE atom of
satellite images). Every candidate competes in one pool whatever its origin, and the
Bonferroni gates count all of them. Learned columns need `raw` (a mapping covering every
id, or a callable id -> raw item, read lazily) and an `interpreter`; `render` (optional)
turns a raw item matplotlib cannot draw (e.g. a multi-band GeoTIFF tile) into an
H x W x 3 image for `interpretation.show()`.

The two that are bookkeeping, shown in the tables only: `access` (`public`: downloadable
without an account; `restricted`: needs a registration, an account or a licence;
`proprietary`: obtained directly from the data owner, not downloadable) and `modality`,
the kind of raw data the columns are extracted from (`survey`: questionnaire or census
answers; `image`: satellite or aerial tiles, photographs; `raster`: gridded geospatial
layers such as climate, population or land cover, sampled at the units' locations;
`events`: dated, geolocated event records such as conflicts, aggregated per id;
`time_series`: regular series per id or station, such as prices; `text`: documents or
free text).

*Checked against the data*, with an error naming the columns: the support (against the
values), the level (against the index name, and against the units' id column in the
`Dataset`, where every unit must have a row), `raw` covering every id, the
interpreter's signature, unique source names and column names, and, over the units, that
no candidate is constant, identical to another candidate or identical to the treatment,
and none is an exact linear combination of other candidates and an intercept (the error
names a minimal dependent set, e.g. a household size next to its age-group counts: such
a pool violates faithfulness, so keep one level of abstraction per concept; columns with
missing values are out of the pool and of this check). Two candidates with
|correlation| > 0.999 give a warning. `discover()` standardises the tested candidates over
the analysed units, so the test's tolerances do not depend on their units. Every vocabulary is
validated, and every per-column dict must cover the columns exactly. *Declared only*:
time, domain, origin, access and modality. The data cannot tell when a variable was
measured, so declare `time` with care: a `pre` column that the treatment caused biases
the search.

**Designs.** `Randomized(treatment, assigned_at, stratified_by=None, sampled_by=None)`
and `Discontinuity(treatment, running_variable, assigned_at, sampled_by=None,
cutoff=None)` (sharp). The checks run when the `Dataset` is built and fail with a clear
message: treatment 0/1 with both arms present, constant within the assignment unit,
assignment units nested in strata and with the sampling clusters; for a discontinuity, a
sharp separation (the violating units are named), the inferred cutoff interval and
direction, and a given cutoff inside that interval. They warn on fewer than 20 clusters
and on strata holding a single arm. Design variables never enter the candidate pool.

**Clustering rule.** Each candidate's interaction test is clustered at the coarsest of
the assignment level, the sampling level and the level where the candidate varies; HC1
when that is the unit itself. Strata enter as fixed effects. The level where a candidate
varies is detected as the coarsest level within whose cells it is constant, among the
levels nested with its declared level: coarser levels whose cells are unions of its
cells, and finer ones. It is therefore never finer than the declared level. For example,
with an extra level of 161 groups over 162 communities (two communities merged, the
others alone), a community column that happens to take the same value in the two merged
communities is detected at the group level and clustered by the 161 groups, which is
coarser than (or as coarse as) the communities; a column that differs between them stays
at the community. A level that cuts across the declared one (some community split
between two groups) is never detected for its columns, even when a column happens to be
constant within its cells, because clustering by it would split the declared clusters.

**Support rule.** A clustered test is identified by the clusters behind each (side of the
candidate) x (arm) cell. For a candidate declared `binary` or `sparse` (cells: zero vs
nonzero), or with a mass point in the data (its most frequent value holds at least 10% of
the units; cells: at the mode vs off it), the support is the smallest number of distinct
clusters over the four cells x (treated, control); a column declared `continuous` or `nonnegative` that
repeats a value over 10% of its ids gets the rule from the data, with a warning.
Candidates with support below `min_support` (default 4) are dropped before the search. The
rule uses only the candidates, the treatment and the clusters, never the outcome, so the
Bonferroni count over the remaining pool stays valid.

**Missing values** never change the analysis silently. A missing id, or a unit without a
row in a source, is an error (add a row of NaN to declare a missing value). Units with a
missing treatment or running variable are dropped by `Dataset`, units with a missing
outcome by `discover`, each with a warning that counts them by arm. A candidate with a
missing value among the units analysed is excluded from the pool, with a warning, and
listed with its count in `data.info()` and `result.table()`. Infinite values are an error.

**The search on arrays.** `discover` runs `nexis.nexis` on the candidate matrix with the
level-aware test `nexis.design.LevelAwareTest`. The search can also be called directly:

```python
from nexis import nexis

res = nexis(y, t, w, backward=False, terminal_filter=True)   # the paper's algorithm
res.selected, res.feature_names, res.metadata["terminal_candidates"]
```

`w` is an (n, m) array or DataFrame of every candidate (a DataFrame's column names become
`feature_names`, and `w.attrs["cluster"]` supplies cluster labels when `cluster=` is not
passed). The defaults of `nexis()` are those of an earlier version of the method
(interleaved backward step, no terminal step), kept so that older results reproduce:
always pass `backward=False, terminal_filter=True` for the paper's algorithm.

| Parameter | Default | Paper | Meaning |
|---|---|---|---|
| `alpha` | `0.05` | `0.05` | Level α: forward gate α/\|remaining\|, terminal gate α/m (m = columns of `w`) |
| `backward` | `True` | `False` | Interleaved backward step after every forward addition |
| `terminal_filter` | `False` | `True` | Keep j ∈ S̃ only if p_j(A) ≤ α/m for every A ⊆ S̃ \ {j} |
| `rho` | `0.5` | `0.5` | Spectral-gap stop: new \|t\| < ρ · min \|t\| of the selected; `0`/`None` disables |
| `adjust` | `"FWER"` | `"FWER"` | Forward gate: Bonferroni, `"FDR"` (Benjamini–Hochberg) or `None` |
| `max_rounds` | `20` | no cap | Cap on forward rounds (CelebA sweeps: 10; applications: 20, never binding) |
| `test` | `"linear"` | `"linear"` | T × Z_j interaction t-test; also `"GCM: ..."` and `"PCM: ..."` (nonparametric) |
| `cluster`, `hc1`, `pvalue_fn` | `None` | | Cluster-robust or HC1 errors, or a custom conditional test |

See the docstrings (`help(nexis.discover)`, `help(nexis.nexis)`) and the architecture
overview in [`src/nexis/__init__.py`](src/nexis/__init__.py).

**Tests.** `python tests/test_regression.py` (the engine's outputs are frozen in
`tests/expected.json`), `python tests/test_api.py`, `python tests/test_examples.py`; all
run on synthetic data, with or without pytest.

## Reproducing the paper

| Case study | Data | Code and numbers |
|---|---|---|
| CelebA (semi-synthetic benchmark) | public | [`src/apps/celeba/README.md`](src/apps/celeba/README.md) |
| Uganda YOP (Youth Opportunities Program) | public replication data, imagery re-downloaded | [`src/apps/uganda/README.md`](src/apps/uganda/README.md) |
| Ghana LEAP 1000 (Livelihood Empowerment Against Poverty) | restricted survey | [`src/apps/ghana/README.md`](src/apps/ghana/README.md) |

Each application README lists every number the paper quotes, its source file and script,
and the design choices behind it. Scripts in `scripts/<app>/` target SLURM but also run
with `bash` from the repo root; logs go to `logs/`, outputs to the untracked `data/` and
`results/`. GPUs are needed for embeddings, SAE training and VLM interpretation; NEXIS
itself runs on CPU in minutes. GPU training is not bit-reproducible, so a retrained SAE
gives other atom indices than the ones quoted in the paper. The application scripts call
the engine directly (`nexis.nexis`, and `nexis.design.build_rule` with its own default
`min_support=5`), so their numbers do not depend on the defaults of `discover`.

### CelebA

Data: the 19,867 validation images of CelebA, streamed from the Hugging Face mirror
`flwrlabs/celeba` at a pinned revision. Compute: embeddings and SAEs ~1 h on one GPU,
sweeps ~400 CPU-hours.

```bash
# GPU: SigLIP 2 embeddings and TopK SAEs (m = 9,216, k = 5 and 20), plus the replica SAE
bash scripts/celeba/submit_embed.sh
bash scripts/celeba/submit_sae.sh
sbatch scripts/celeba/submit_resample_embed.sh b1 1
sbatch scripts/celeba/submit_resample_sae.sh 20 b1
# CPU sweeps, 50 seeds (jobs whose output exists are skipped)
bash scripts/celeba/submit_experiment.sh          # main, k=5, Z_pre, method ablations
bash scripts/celeba/submit_experiment_ushape.sh   # U-shape DGP
bash scripts/celeba/submit_dgp_extra.sh           # replica, r = 3, 1, 0
bash scripts/celeba/submit_per_modifier_recall.sh
bash scripts/celeba/submit_alignment_appendix.sh  # Principal Alignment check
# Figures and tables -> results/celeba/figures_v2/
python src/apps/celeba/figure_main.py --data-dir results/celeba/experiment_v2/k20/sae \
    --out-path results/celeba/figures_v2/figure_main.pdf --nexis-key NEXIS-v2
python src/apps/celeba/figure_appendix.py --experiment-dir results/celeba/experiment_v2 \
    --out-dir results/celeba/figures_v2 --variant v2
python src/apps/celeba/figure_dgp_extra.py
python src/apps/celeba/figure_test_story_v2.py
python src/apps/celeba/interleaved_backward_align.py sae_precode_k20 && \
    python src/apps/celeba/third_modifier_candidates.py
python src/apps/celeba/run_statistics.py
(cd src/apps/celeba/benchmark && python run.py violation)   # the controlled-violation figure
```

[`src/apps/celeba/benchmark/`](src/apps/celeba/benchmark/README.md) is a self-contained
package for every CelebA experiment (`python run.py <block>` from that folder); only its
`main` and `violation` blocks are checked against the paper so far.

### Uganda YOP

Compute: SAE ~1 h on one GPU, VLM ~30 min on one H100.

```bash
python src/apps/uganda/download_tiles.py --mode rct        # 331 trial sites
python src/apps/uganda/download_tiles.py --mode national   # national grid (SAE corpus)
python src/apps/uganda/extract_satellite_features.py --tif-dir data/uganda/satellite/tif_rct --out-dir data/uganda/satellite/rct
python src/apps/uganda/extract_satellite_features.py --tif-dir data/uganda/satellite/tif_national --out-dir data/uganda/satellite/national
sbatch scripts/uganda/submit_train_sae.sh                  # -> results/uganda/prithvi_l5_1024/
python src/apps/realworld_final_runs.py --only uganda      # paper numbers
python src/apps/uganda/multilevel_groupcluster.py          # multilevel limitation
python src/apps/uganda/district_sensitivity.py             # district-dummy sensitivity
python src/apps/uganda/table_gate.py                       # GATE columns
sbatch scripts/uganda/submit_interpret.sh                  # VLM labels (Qwen2.5-VL-72B)
python src/apps/uganda/figure_neural.py
python src/apps/uganda/figure_maps.py
python src/apps/figure1_tiles.py                           # Figure 1 tiles
```

### Ghana LEAP 1000

Compute: SAE ~2 h and VLM ~45 min on one H100. Run notes: Section 11 of
[`src/apps/ghana/README.md`](src/apps/ghana/README.md).

```bash
python src/apps/ghana/download_satellite_images.py --year 2015   # 162 communities
python src/apps/ghana/download_national_grid.py                  # national grid (SAE corpus)
python src/apps/ghana/extract_satellite_features.py
python src/apps/ghana/train_sae.py --d-hidden 4096 --k 25 --epochs 2000 --batch-size 256 \
    --train-embeddings data/ghana/satellite/national/prithvi_embeddings.npy \
    --eval-embeddings data/ghana/satellite/prithvi_embeddings.npy \
    --eval-ids data/ghana/satellite/prithvi_comm_ids.npy --out-dir data/ghana/satellite
python src/apps/realworld_final_runs.py --only ghana             # paper numbers
python src/apps/ghana/table_gate.py                              # GATEs
bash scripts/ghana/run_figure_neural.sh                          # VLM temporal step (GPU) and figure
python src/apps/ghana/figure_neural_1777.py                      # exploratory figure
python src/apps/ghana/figure_maps.py
```

## Data

The real-world data are not distributed with this repository, and it contains no
unit-level records; the tests and the example use synthetic data.

- **CelebA**: public; the scripts stream it from the Hugging Face mirror `flwrlabs/celeba`.
- **Uganda YOP**: survey microdata from the public replication release of Jerzak et al.
  (2023) of the Blattman, Fiala & Martinez (2014) trial
  ([Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/O8XOSF),
  [Hugging Face](https://huggingface.co/datasets/cjerzak/ImageHeterogeneity)), saved as
  `data/uganda/UgandaDataProcessed.csv`. Landsat 7 imagery is re-extracted from Google
  Earth Engine (`earthengine authenticate`), since it cannot be redistributed.
- **Ghana LEAP 1000**: the household panel (2015–2017) is restricted: request access from
  UNICEF Ghana; it must not be redistributed. Landsat 8 imagery from Google Earth Engine.

## Citation

```bibtex
@article{cadei2026nexis,
  title   = {From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification},
  author  = {Cadei, Riccardo and Otchere, Frank and Tirivayi, Nyasha and Angeles Tagliaferro, Gustavo and Bargagli-Stoffi, Falco J. and Locatello, Francesco},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.17010}
}
```

## License

MIT, see [`LICENSE`](LICENSE).
