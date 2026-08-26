# Rebuttal draft — multilevel / clustered inference on the Uganda case study

Backing analysis: `docs/uganda_robustness_clustering.md`.
Code: `src/apps/uganda/{robustness_clustering,multilevel_inference}.py`.

---

## Version 0 — very concise (one point within a longer rebuttal) ← use this

> **Uganda standard errors.**
>
> We agree. Appendix D.6 flagged this and deferred a multilevel test to future work;
> since these p-values drive the Bonferroni gate, deferral was not adequate, and we
> have now run it.
>
> We cluster each candidate at the coarsest level at which it varies, using a wild
> cluster bootstrap where clusters are few and design-based randomization inference
> over the trial's 14 randomization blocks. This required no extension to NEXIS, which
> already accepts any p-value-returning test. Two corrections surfaced: the language
> dummies are *region*-level (7 clusters), not community-level, and our specification
> omitted block fixed effects despite block treatment propensity ranging 0.20–0.62.
>
> | Modifier | Level | OLS (published) | Design-based |
> |---|---|---|---|
> | Karamojong | region (7) | 3.7e-10 | **1e-05** |
> | Lugbara | region (7) | 2.8e-07 | **2e-05** |
> | Pallisa | region (7) | 1.0e-04 | 0.23 |
> | Neuron 339 | community (327) | 2.8e-06 | 5.4e-03 |
> | Neuron 533 | community (327) | 8.7e-06 | 1.8e-03 |
> | NDVI | community (327) | 5.9e-08 | 1.3e-02 |
> | Neuron 820 | community (327) | 9.5e-05 | 3.6e-03 |
>
> Against the gate (≈3e-04) two of seven modifiers survive; point estimates and signs
> are unchanged throughout, so this is an inference rather than an estimation issue.
> No learned satellite feature survives correction.
>
> We label all Uganda findings as suggestive in the main text as requested, adopt block
> fixed effects with design-based inference as the primary specification, and correct
> the hierarchy statement and the limitations paragraph.

---

## Version A — concise (≈300 words, for a tight rebuttal box)

We thank the reviewer for pressing on this. The concern is correct, and acting on it
changed our results.

**We were wrong about the data hierarchy.** The language-group dummies are not
community-level, as stated in Appendix D.1 — they are *region*-level. `lang_group`
partitions the sample into 7 regions and each dummy is a one-hot of that partition.
Our published p-values treated 2,082 individuals as independent for a modifier that
takes 7 distinct values.

**A standard multilevel test exists; we implemented it.** We cluster each candidate at
the coarsest level at which it varies (Cameron & Miller 2015): region for language
dummies, community for SAE atoms and spectral indices, group for group composition.
NEXIS required no extension — it consumes only p-values, so the level-aware test plugs
into a `pvalue_fn` hook. We add a restricted wild cluster bootstrap-t where clusters
are few (exhaustively enumerated at G=7, so exact), and design-based randomization
inference re-randomising T over groups within the 14 district blocks actually used.
We also add block fixed effects: block treatment propensity ranges 0.20–0.62, so they
are needed for unbiasedness, and the published specification omitted them.

**Results.** Under block FE + randomization inference, two of seven modifiers survive
the Bonferroni gate: Karamojong (p=1e-05) and Lugbara (p=2e-05). Pallisa (0.23),
neuron 339 (5.4e-03), neuron 533 (1.8e-03), NDVI (1.3e-02) and neuron 820 (3.6e-03) do
not. Panel B's failure is driven by the omitted block fixed effects, not by clustering.
Language-dummy interactions are *not testable* by any cluster-robust method: each is
active in exactly 1 of 7 regions, the one-treated-cluster case.

**Revision.** We correct the hierarchy, adopt block FE + randomization inference as the
primary specification, retain only the two language modifiers as confirmatory, demote
the remainder to exploratory, and replace the limitations paragraph. We state
explicitly that a modifier estimated from one region does not generalise beyond the
trial's regions.

---

## Version B — full (for a longer response or the revised appendix)

### 1. The reviewer is right, and one of our stated facts was wrong

Appendix D.1 places the language-group dummies at community level. They are
**region**-level. `lang_group` *defines* a partition of the sample into 7 regions, and
each dummy is a one-hot encoding of that partition; a variable constant within region
is trivially constant within every site inside it, which is how the mislabel survived.
The corrected hierarchy on the n = 2,082 analysis sample:

| Level | Clusters | Candidates |
|---|---|---|
| Region (`lang_group`) | 7 | all 7 language dummies |
| District (= randomisation block) | 14 | — |
| Community (site) | 327 | 146 SAE atoms + 12 spectral indices |
| Group (randomisation unit) | 439 | group composition |
| Individual | 2,082 | 4 demographics |

Two further structural facts we had not reported: groups are **not** nested in
communities (30 of 439 draw members from more than one site), and treatment is
constant within 278 of 327 communities, so for a community-level modifier the T×Z
contrast is almost entirely between-cluster.

### 2. Our stated reason for avoiding clustering was incorrect

We wrote that "a community-level cluster bootstrap is degenerate … all within-community
observations share the same regressor value". This is wrong: constancy within cluster
is the *canonical* case for a cluster bootstrap, not a degeneracy — between-cluster
variation identifies the coefficient. Our community-level wild bootstrap runs without
issue.

There *is* a degeneracy, but at a different level and for a different reason. Each
language dummy is active in exactly **1 of 7 regions**. This is the one-treated-cluster
case, where the restricted wild bootstrap provably has no power; exhaustive enumeration
makes it visible, since Karamojong and Lugbara return *identical* p-values of
0.2969 = 38/128 — the statistic is reading the design, not the data. So p = 0.30 is not
evidence of absence; region-level cluster-robust inference on these modifiers is simply
unavailable, and only a design-based test can speak to them.

We also over-claimed that "neither standard clustered approach is well-suited" and left
a multilevel test to future work. The standard approach exists and took ~250 lines.

### 3. What we did

Following Cameron & Miller (2015, §II.C) — cluster at the coarsest level at which the
regressor of interest varies — the cluster level becomes a property of the *candidate*,
not of the analysis. Four tiers:

1. **Level-aware CR1S**, each candidate clustered at its own level. NEXIS needed no
   redesign: it consumes only p-values and t-statistics, so this enters through a
   `pvalue_fn` hook — realising the extensibility the paper already claims.
2. **A support gate at the candidate's own level** (≥10 active-treated and ≥10
   active-control clusters). Without it the sandwich is anti-conservative to the point
   of absurdity: neuron 177 is null under OLS (p = 0.51) and 7-sigma under community
   clustering (p = 2.3e-07), the classic few-treated-clusters failure (MacKinnon & Webb
   2017). 98 of 170 candidates pass; all 7 region-level candidates are excluded.
3. **Restricted wild cluster bootstrap-t** (Rademacher, null-imposed), enumerated
   exhaustively when 2^G is small so the p-value is exact.
4. **Design-based randomization inference**: re-randomise T over groups within the 14
   district blocks Blattman et al. actually used, under a constant-effect null. This
   uses only the randomness the experimenters created and is immune both to arbitrary
   correlation in Y and to the few-clusters problem.

We do **not** cluster on district. District is the randomisation *block*; clustering is
called for by clustered sampling or clustered assignment (Abadie, Athey, Imbens &
Wooldridge 2023), and a stratification variable is neither. Blocking is handled by
conditioning. Accordingly we add **block fixed effects**, which the published
specification omitted — block treatment propensity ranges from 0.20 to 0.62, so they
are required for unbiasedness, not merely efficiency. (With block FE the language main
effects are exactly collinear with the district dummies — a language group *is* a set
of districts — so the design must be rank-filtered before any variance is computed.)

### 4. Results

Gate = the Bonferroni threshold NEXIS applies on the realised path (≈3.0e-04).
B = 99,999 for both bootstrap and permutation.

| Modifier | Level | active reg/blk/comm | Published OLS | CR (own level) | **RI + block FE** |
|---|---|---|---|---|---|
| Karamojong | region | 1/3/36 | 3.7e-10 | not testable | **1.0e-05** ✓ |
| Lugbara | region | 1/3/48 | 2.8e-07 | not testable | **2.0e-05** ✓ |
| Pallisa | region | 1/1/29 | 1.0e-04 | not testable | 0.225 ✗ |
| Neuron 339 (perennial river) | community | 6/7/21 | 2.8e-06 | 0.091 ✗ | 5.4e-03 ✗ |
| Neuron 533 (vegetation heterog.) | community | 7/14/99 | 8.7e-06 | 5.6e-04 ✗ | 1.8e-03 ✗ |
| NDVI | community | 7/14/327 | 5.9e-08 | 1.6e-02 ✗ | 1.3e-02 ✗ |
| Neuron 820 (structured agri.) | community | 4/7/40 | 9.5e-05 | 1.3e-02 ✗ | 3.6e-03 ✗ |

Block fixed effects move results in *both* directions, which is the signature of a real
omitted-variable effect rather than noise: they strengthen Lugbara (2.8e-04 → 2.0e-05)
and neuron 339 (0.019 → 5.4e-03), and they destroy Panel B — NDVI degrades 37×
(3.5e-04 → 1.3e-02). NDVI is strongly geographic, so much of the apparent
"NDVI modifies the treatment effect" signal was varying block propensity read as
environmental heterogeneity. **Panel B fails for a reason unrelated to clustering.**

Signs and magnitudes are stable under every estimator; this is an inference problem,
not an estimation problem. A nested mixed model (community + group random intercepts,
REML) agrees on the ordering. Re-running full NEXIS selection with the level-aware test
on the support-gated pool recovers none of the published modifiers.

### 5. Internal vs external validity

Karamojong gives p = 1e-05 by randomization inference and 0.30 under region clustering.
Both are correct answers to different questions. Randomization inference asks whether
the observed difference exceeds what the actual assignment mechanism produces — yes,
decisively. Region clustering asks whether it would replicate over a fresh draw of
regions — unanswerable from 7 regions, 1 of them Karamojong. NEXIS's claim is about
*this* trial, so the design-based answer is the appropriate primary one, but we now
state the generalisability limit explicitly rather than leaving it implicit.

### 6. Changes to the paper

1. Correct the data hierarchy: language dummies are region-level (7 clusters).
2. Adopt block fixed effects + randomization inference as the primary specification.
3. Retain Karamojong and Lugbara as confirmatory findings; demote Pallisa, neurons 339,
   533, 820 and NDVI to exploratory, with the revised p-values reported.
4. Replace the limitations paragraph: remove the incorrect claim about community-level
   bootstrap degeneracy, remove "future work", and state the real obstacle (language
   modifiers are active in one region, so only design-based inference applies).
5. Correct the power argument in Appendix D.3: pooling 14 districts into 7 language
   groups *halves* the independent units for the modifier — it buys precision only
   under the assumption that treatment-effect shocks do not vary by region, which is
   exactly what cannot be tested here.
6. State that no SAE-neuron modifier survives, and adjust the case study's claims
   accordingly.

### 7. What we are not claiming

Point 6 is a material reduction in what the Uganda study demonstrates: the two
surviving modifiers are hand-crafted covariates already available to Blattman et al.,
not learned satellite features. We think the case study still shows that NEXIS
*recovers* known heterogeneity under correct inference, and that its conditional
selection is far more parsimonious than marginal screening — though we note the
marginal-baseline counts reported in Appendix D.5 (71 and 45) were computed under the
original OLS specification and would need recomputing under the corrected one before
we restate that comparison. We no longer claim the method surfaces novel environmental
modifiers on this dataset. We would rather state this than defend the original numbers.
