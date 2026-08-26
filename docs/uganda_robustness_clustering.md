# Uganda — clustered and multilevel inference for the NEXIS discoveries

Reproduce with:

```bash
# variance-estimator sweep + nested mixed model
python src/apps/uganda/robustness_clustering.py \
    --embed-model prithvi_l5 --sae-dim 1024 \
    --outcomes skilled_employed,log_biz_assets

# level-aware clustering, wild cluster bootstrap, randomization inference
python src/apps/uganda/multilevel_inference.py \
    --embed-model prithvi_l5 --sae-dim 1024 \
    --outcomes skilled_employed,log_biz_assets \
    --n-boot 99999 --n-perm 99999
```

Outputs in `results/uganda/prithvi_l5_1024/{robustness_clustering,multilevel_inference}/`.
Nothing is retrained; the frozen SAE artifacts are read as-is.

---

## 1. The published SEs are homoskedastic OLS with no clustering

`nexis()` defaults to `cluster=None, hc1=False` (`src/method/nexis.py`) and
`analyze.py` never passes either. Brief §6 and appendix `\paragraph{Standard errors}`
are accurate as written. The CR1S path already existed in the core but was never
wired into the Uganda app; it is validated here against `statsmodels`
`cov_type='cluster'`/`'HC1'` (exact agreement on the t-statistic).

## 2. Level of each candidate — corrected

**The `lang_*` dummies are region-level, not community-level.** Appendix §*Data
hierarchy* and brief §3 both place them at community level; that is wrong, and it
matters more than any other single point in this document. `lang_group` *defines* a
partition of the sample into **7 regions**; each dummy is a one-hot encoding of that
partition. A variable constant within region is trivially constant within every site
inside it, so a level check that only scans group and community mislabels them. The
corrected scan runs coarsest-first:

| Level | Clusters | Count | Members |
|---|---|---|---|
| **Region** (`lang_group`) | **7** | **7** | **all `lang_*` dummies** |
| District / block | 14 | 0 | — |
| Community (site) | 327 | 158 | 146 SAE neurons + 12 spectral indices |
| Group (randomisation unit) | 439 | 1 | `group_female` |
| Individual | 2082 | 4 | `age`, `female`, `father_educ`, `mother_educ` |

Region ⊃ district ⊃ community is a strict chain here (no community spans a district,
no district spans a language group). Groups are the exception: 30 of 439 draw members
from more than one site, so group is *not* nested in community.

## 3. Design facts

| Partition | Clusters | With within-cluster T variation | Used as a cluster level? |
|---|---|---|---|
| Region | 7 | 7 | diagnostic only (see §4b) |
| Block / district | 14 | 14 | **no — it is a blocking variable** |
| Community | 327 | 49 | yes |
| Group | 439 | 0 (T assigned at group level) | yes |

Treatment was randomised over groups **within 14 district blocks** (recovered as
`ceil(strata/2)`; each block has both arms and sits inside one district and one
region). T is constant within 278 of 327 communities, so for a community-level
modifier the T×Z contrast is almost entirely between-cluster.

**District is deliberately not a clustering level.** Clustering is called for by
clustered *sampling* or clustered *assignment* (Abadie, Athey, Imbens & Wooldridge
2023); a stratification variable is neither. Blocking is handled by *conditioning* —
block fixed effects and block-stratified re-randomisation — not by clustering. With
G = 14 its CRVE is also anti-conservative, and empirically it was the single estimator
that disagreed with all others, "rescuing" `Z_339` (6.8e-05) and `W_lang_7` (7.0e-05)
after every other estimator rejected them. With it removed, the three remaining CR
variants agree unanimously (3/5 and 0/2 features clearing their gate).

**Block treatment propensity varies from 0.20 to 0.62 across the 14 blocks.** With a
blocked design and propensity that unequal, block fixed effects are needed for
*unbiasedness*, not merely efficiency — without them the pooled estimate absorbs the
correlation between a block's treatment propensity and its outcome level. The
published specification includes none. This, rather than clustering, is the correct
way to respect the design, and §6 reports every test with and without them.

One structural consequence: with block FE the `lang_*` **main effects are exactly
collinear** with the district dummies — a language group *is* a set of districts. The
T × lang_j interactions remain identified, but the design is rank-deficient and must
be rank-filtered before any variance is computed (`rank_filter`), or `inv(X'X)`
silently returns garbage.

## 4. Two distinct degeneracies, in opposite directions

Neither is a bug — both reproduce in `statsmodels`.

**(a) Sparse neurons at community level → anti-conservative.** The T×Z_j interaction
is identified off clusters that are *both* active and treated. With one or two such
cells the sandwich meat collapses while the t-test still uses df = G−1:

| Feature | active communities | of which treated | p (OLS) | p (HC1) | p (CR-community) |
|---|---|---|---|---|---|
| `Z_909` | 7 | 1 | 3.3e-04 | 5.6e-47 | 8.7e-20 |
| `Z_177` | 6 | 1 | **0.51** | 0.53 | **2.3e-07** |

`Z_177` is null under OLS and 7-sigma under clustering. Classic few-treated-clusters
failure (MacKinnon & Webb 2017). The published `--active-threshold 5` (≥5 active
*sites*) is far too weak for sandwich inference.

**(b) Language dummies at region level → no power at all.** Each `lang_*` dummy is
active in **exactly 1 of 7 regions**, by construction. This is the one-treated-cluster
case, where the restricted wild bootstrap is provably degenerate. The exhaustive
enumeration makes it visible: `W_lang_4` and `W_lang_2` return *identical* p-values of
0.2969 = 38/128, because with G=7 the bootstrap distribution is driven by the sign
flip of the single active region and carries almost no information from the data.
p = 0.30 here is not evidence of absence — it is the test having nothing to work with.

## 5. The method

There is a standard answer, and it does not need new methodology. Cameron & Miller
(2015, §II.C): **cluster at the coarsest level at which the regressor of interest
varies.** For a pool spanning levels, that makes the cluster level a property of the
*candidate*, not of the analysis. Concretely, four tiers:

1. **Level-aware CR1S** — each candidate clustered at its own level (region for
   `lang_*`, community for neurons/spectral, group for `group_female`, HC1 for
   individual-level). Cheap enough to run inside the selection loop.
2. **Support gate at the candidate's own level** — require ≥10 active-treated and ≥10
   active-control clusters, or the sandwich is not usable (§4a). 98 of 170 pass;
   **all 7 region-level candidates are dropped**, along with 65 community-level ones.
3. **Restricted wild cluster bootstrap-t** (Rademacher, null-imposed; Cameron, Gelbach
   & Miller 2008) where clusters are few, enumerated exhaustively when 2^G ≤ 2^14 so
   the p-value is exact and the resolution floor is explicit.
4. **Design-based randomization inference** — re-randomise T over groups within the 14
   real blocks under a constant-effect null. Uses only the randomness the
   experimenters created; immune to arbitrary correlation in Y and to the
   few-clusters problem. **This is the only valid test for the `lang_*` modifiers.**

NEXIS needed no redesign to accept this. It consumes only p-values and t-statistics,
so the level-aware test drops in through a new `nexis(pvalue_fn=...)` hook
(`src/method/nexis.py`) — which simply realises the claim the appendix already makes
("any valid p-value-returning test can be plugged in").

## 6. Results

`--n-boot 99999 --n-perm 99999`. Gate = the Bonferroni threshold NEXIS actually
applies on the realised path (≈3.0e-04).

| Feature | Level | active reg/blk/comm | OLS | CR (own level) | WCB | RI | WCB **+block FE** | **RI +block FE** |
|---|---|---|---|---|---|---|---|---|
| `W_lang_4` | region | 1/3/36 | 3.7e-10 | *degenerate* | 0.2969 *(no power)* | 1.0e-05 | 0.359 *(no power)* | **1.0e-05** ✓ |
| `W_lang_2` | region | 1/3/48 | 2.8e-07 | *degenerate* | 0.2969 *(no power)* | 2.8e-04 | 0.344 *(no power)* | **2.0e-05** ✓ |
| `W_lang_7` | region | 1/1/29 | 1.0e-04 | *degenerate* | 0.6094 *(no power)* | 0.170 | 0.719 *(no power)* | 0.225 ✗ |
| `Z_339` | community | 6/7/21 | 2.8e-06 | 4.1e-03 ✗ | 0.104 ✗ | 0.019 | 0.091 ✗ | 5.4e-03 ✗ |
| `Z_533` | community | 7/14/99 | 8.7e-06 | 1.8e-04 ✓ | 1.6e-04 ✓ | 8.7e-04 | 5.6e-04 ✗ | 1.8e-03 ✗ |
| `W_ndvi_mean` | community | 7/14/327 | 5.9e-08 | 4.6e-04 ✗ | 6.0e-05 ✓ | 3.5e-04 | 1.6e-02 ✗ | 1.3e-02 ✗ |
| `Z_820` | community | 4/7/40 | 9.5e-05 | 4.2e-04 ✗ | 4.0e-03 ✗ | 4.5e-04 | 1.3e-02 ✗ | 3.6e-03 ✗ |

The rightmost column is the correct specification: block fixed effects for the blocked
design, design-based inference for the variance. Under it exactly **two** of the seven
published modifiers survive — `W_lang_4` and `W_lang_2`.

Adding block FE moves results in both directions, which is the signature of a real
omitted-variable effect rather than noise. It *strengthens* `W_lang_2` (2.8e-04 →
2.0e-05, from borderline to clear) and `Z_339` (0.019 → 5.4e-03). It *destroys*
Panel B: `W_ndvi_mean` degrades 37× (3.5e-04 → 1.3e-02) and `Z_820` 8× (4.5e-04 →
3.6e-03). NDVI is strongly geographic, so without block FE a large part of the
apparent "NDVI modifies the treatment effect" signal was the varying block propensity
being read as environmental heterogeneity. **Panel B fails for a reason that has
nothing to do with clustering.**

Monte Carlo resolution: B = 99999, so p̂ = 1e-05 and 2e-05 rest on 0 and 1 exceedances
(at the floor); the failures at 1e-02 and 1e-03 are resolved to within a few percent.

**Level-aware NEXIS re-selection** (support-gated pool, so no `lang_*` candidates
exist): `skilled_employed` → `Z_323`, `Z_859`; `log_biz_assets` → `Z_323`,
`W_female`. None of the published modifiers is recovered, and `Z_323` fails its own
Bonferroni gate conditional on the published set (3.6e-03), so it is not a replacement
discovery either.

**Nested mixed model** (community + group random intercepts; REML, converged) — note
it omits the region level, so it is not a substitute for the region-level question:

| Outcome | Term | coef | SE | p |
|---|---|---|---|---|
| `skilled_employed` | T × `W_lang_4` | −0.378 | 0.078 | 1.2e-06 |
| | T × `W_lang_2` | −0.313 | 0.071 | 9.7e-06 |
| | T × `Z_339` | −0.091 | 0.024 | 1.8e-04 |
| | T × `Z_533` | −0.106 | 0.032 | 8.8e-04 |
| | T × `W_lang_7` | +0.234 | 0.091 | 1.0e-02 |
| `log_biz_assets` | T × `W_ndvi_mean` | +2.100 | 0.572 | 2.4e-04 |
| | T × `Z_820` | −0.276 | 0.102 | 6.6e-03 |

ICCs: 0.068 community / 0.066 group (`skilled_employed`); 0.058 / **0.181**
(`log_biz_assets`). Signs and magnitudes are stable across every estimator — this is
an inference question, not an estimation question.

### The RI-vs-clustering gap is substantive, not an artifact

`W_lang_4` gives RI p = 1e-05 and region-clustered p = 0.30. Both are correct answers
to different questions:

- **RI (internal validity):** given the randomisation actually performed, is the
  Karamojong-vs-rest difference in treatment effect larger than chance assignment
  produces? **Decisively yes.**
- **Region clustering (external validity):** would this replicate over a fresh draw of
  regions? **Unanswerable from 7 regions, 1 of them Karamojong.**

NEXIS's claim is about *this* trial — for whom did YOP work — so the design-based
answer is the appropriate primary one. But the generalisability caveat is real and
currently unstated: a language-group modifier estimated from one region cannot be
projected to regions outside the trial.

## 7. Is this in agreement with the appendix?

The appendix limitations paragraph makes four claims. One holds, three do not.

| Appendix claim | Verdict |
|---|---|
| "clustering at the group level ignores the community-level dependence induced by site-constant satellite features" | **Correct.** |
| language dummies are "community-level" | **Wrong.** They are region-level (7 clusters) — §2. Appendix §*Data hierarchy* needs the same fix. |
| "a community-level cluster bootstrap is degenerate for those same features (all within-community observations share the same regressor value)" | **Wrong.** Constancy within cluster is the *canonical* case for a cluster bootstrap, not a degeneracy; between-cluster variation identifies the coefficient. The community-level WCB runs fine and returns p = 1.6e-04 (`Z_533`) and 6.0e-05 (`W_ndvi_mean`). The real degeneracy is at *region* level for the language dummies, for a different reason (one active cluster) — §4b. |
| "Neither standard clustered approach is well-suited… We leave the design of an appropriate multilevel test to future work." | **Overstated.** The standard approach is per-candidate clustering plus WCB plus design-based RI; it is ~250 lines and needs no extension to NEXIS beyond a test hook the appendix already claims exists. |

Also affected: the power argument at appendix §*Representations* ("mean ≈47
communities per cluster vs ≈24 per individual district") counts the wrong unit. For
inference on a T×language interaction, the relevant count is the number of language
clusters (**7**), not communities within them. Aggregating districts into language
groups *reduces* the number of independent units for the modifier from 14 to 7; it
buys precision only under the assumption that treatment-effect shocks do not vary at
the region level, which is exactly what is untestable here.

## 8. Are the results robust?

| Feature | RI + block FE (primary) | Cluster-robust at own level | Verdict |
|---|---|---|---|
| `W_lang_4` (Karamojong) | **1.0e-05** ✓ | not testable (1 of 7 regions) | **Robust internally**; not generalisable beyond the trial's regions |
| `W_lang_2` (Lugbara) | **2.0e-05** ✓ | not testable | **Robust internally**; same caveat |
| `W_lang_7` (Pallisa) | 0.225 ✗ | not testable | **Not robust** |
| `Z_533` (vegetation heterogeneity) | 1.8e-03 ✗ | 5.6e-04 ✗ | **Not robust** (borderline pre-FE) |
| `Z_339` (perennial river) | 5.4e-03 ✗ | 0.091 ✗ | **Not robust** |
| `W_ndvi_mean` | 1.3e-02 ✗ | 1.6e-02 ✗ | **Not robust** — driven by omitted block FE |
| `Z_820` (structured agriculture) | 3.6e-03 ✗ | 1.3e-02 ✗ | **Not robust** |

At the corrected multiplicity threshold, **two of seven** published modifiers survive:
`W_lang_4` and `W_lang_2`, both by design-based inference, both region-level with one
active region out of seven. The published OLS p-values of 1e-10 to 1e-04 overstate the
evidence by treating 2082 individuals as independent when the modifiers vary across 7
regions or 327 sites; Panel B additionally rests on an unblocked specification.

**Recommendation.** Report `W_lang_4` and `W_lang_2` as the heterogeneity result, with
block fixed effects and design-based p-values, and state the 7-region generalisability
limit explicitly. Demote `W_lang_7`, `Z_339`, `Z_533` and all of Panel B to
exploratory. Replace the appendix limitations paragraph: the multilevel test is not
future work, and the stated reason for avoiding a community-level bootstrap is
incorrect.

Note this leaves **no surviving SAE-neuron modifier** — the two survivors are
hand-crafted covariates inherited from Blattman et al. That is a material change to
what the Uganda case study demonstrates and should be stated plainly rather than
worked around.

**Caveats.** The mixed model omits the region level (7 units will not identify a
region random effect; this is why the design-based route is used instead). RI imposes
a constant-effect null with a single pooled τ̂; a block-specific τ̂ is the natural
sensitivity check. Region-level WCB is reported to show the test has no power, not as
evidence of absence.
