# Reviewer question: a (modified) conditional mean independence test for Eq. (7)

> *"Could one use a (modified) conditional mean independence test (see work by Anton
> Lundborg et al) for (7)?"*

**Short answer: yes, it fits exactly, and the "modified" qualifier is the right one.**
We implemented it, re-ran the statistical-test ablation with it, and it turns out to
close a real gap in the current test suite rather than merely duplicate it.

---

## 0. Reviewer answer (drop-in text)

> **Yes — and it turned out to be more than a cosmetic addition, so we have implemented
> it and added it to the test ablation.**
>
> The suggestion fits the setting exactly. Equation (7) is the $S=\emptyset$ case of the
> null NEXIS actually tests at every step, Equation (8),
> $H_0(j\mid S): \E[\tau \mid \bm Z^{S\cup\{j\}}] = \E[\tau \mid \bm Z^{S}]$,
> and both are conditional *mean* independence statements. The natural instrument is
> therefore the Projected Covariance Measure (PCM) of Lundborg, Kim, Shah and Samworth
> (2024), which targets $\E[V \mid X, Z] = \E[V \mid Z]$ directly.
>
> The reviewer's parenthetical "(modified)" identifies precisely the obstacle: $\tau$ is
> never observed, so the PCM cannot be applied off the shelf. The modification is short.
> Under randomisation with $e=\Prob(T=1)$ known by design, the R-learner pseudo-outcome
> we already use for our GCM variants (Equation~\ref{eq:r_learner}),
> $\widehat\varphi = (Y - \widehat m(\bm Z^S))(T-e)/\{e(1-e)\}$,
> satisfies $\E[\widehat\varphi \mid \bm Z] = \E[\tau \mid \bm Z]$ for *any* function
> $\widehat m$ of $\bm Z$ — $\widehat m$ affects variance but never bias, because
> $T$ is independent of the potential outcomes. Hence $H_0(j\mid S)$ is equivalent to
> $\E[\varphi \mid \bm Z^{S\cup\{j\}}] = \E[\varphi \mid \bm Z^{S}]$, which is the PCM
> null with $V=\varphi$, $X=Z^j$, $Z=\bm Z^S$. Substituting the pseudo-outcome is the
> only modification required; the rest of the PCM machinery — fitting a projection
> $\widehat f_j$ of the conditional-mean contrast on one half of the sample and forming a
> one-sided residual-product statistic on the other — carries over unchanged.
>
> This is *not* redundant with the GCM variants already in the paper. The GCM tests
> $\E[\mathrm{cov}(\varphi, Z^j \mid \bm Z^S)] = 0$, which is implied by, but not
> equivalent to, $H_0(j \mid S)$: any alternative whose conditional-mean contrast is
> uncorrelated with $Z^j$ is invisible to it at every sample size. We had flagged this
> blind spot for the linear working model; the GCM inherits it because it residualises
> $Z^j$ linearly. The PCM is the only member of the family that removes it.
>
> We added two PCM instantiations (quadratic and LightGBM projection) to
> Appendix~\ref{sec:method-details:test} and re-ran the full test ablation
> (Table~\ref{tab:celeba:test_ablation}). Two findings:
>
> 1. **On the benchmark as published (Panel A) nothing changes.** The linear test still
>    reaches recall $0.95$ at roughly half the sample size and effect magnitude of every
>    assumption-lean alternative, and the PCM matches the GCM ($n^\star=2000$ at $\eta=5$
>    for both), so our recommended default is unaffected. Notably the PCM does not pay
>    the expected sample-splitting penalty here — its one-sidedness offsets the split.
> 2. **On a DGP where the GCM is blind (Panel B) the PCM is the only test that works.**
>    We construct $\tau$ as an orthogonalised quadratic of the two principal coordinates,
>    calibrated on the full CelebA population so that $\E[g(Z^j)]=0$ and
>    $\mathrm{Cov}(g(Z^j), Z^j)=0$ exactly, while $\E[\tau \mid Z^j]$ genuinely varies.
>    The linear test and *both* GCM variants recover zero coordinates at every $(n,\eta)$
>    on the grid and do not improve with more data — they are inconsistent there, not
>    merely underpowered. The PCM with a LightGBM projection recovers
>    $\mathcal{S}^\star$ exactly (precision and recall $1.00$) from $n=750$.
>
> All numbers are over 50 Monte-Carlo seeds and we verified them on a disjoint seed block
> (seeds 50--99 against 0--49): every entry agrees within $1.4$ Monte-Carlo standard
> errors.
>
> We also report one honest limitation. On *dense* pre-activations the PCM's empirical
> family-wise error rate drifts above nominal as $n$ grows ($0.10$--$0.12$ at $n=5000$
> against $0.05$ nominal, versus $0.00$--$0.02$ for the GCM); on the sparse codes used
> throughout the paper it is $0.00$ everywhere. The excess is confined to the extreme
> tail that the Bonferroni gate $\alpha/m \approx 5\times10^{-6}$ reads; the bulk of the
> $p$-value distribution is calibrated. We traced it to a small number of high-leverage
> observations in the fitted projection, and we report it explicitly rather than tune it
> away — a multiplier bootstrap of the null would be the principled remedy.
>
> The practical guidance in Appendix~\ref{sec:method-details:test} is updated
> accordingly: keep the linear test as the default; when non-monotone heterogeneity in
> the dictionary cannot be ruled out, escalate to the **PCM rather than the GCM**, since
> the GCM buys robustness to nuisance misspecification but not to this class of
> alternative.

---

## 1. Why it fits

Equation (7) defines the marginal effect-modifier set
$\mathcal{M}^\star = \{ j : \mathbb{P}(\mathbb{E}[\tau \mid Z^j] \neq \mathbb{E}[\tau]) > 0 \}$,
and Equation (8) its conditional counterpart, the null NEXIS actually tests at every
forward and backward step,

$$H_0(j \mid S):\quad \mathbb{E}[\tau \mid \bm Z^{S \cup \{j\}}] = \mathbb{E}[\tau \mid \bm Z^{S}] \quad \text{a.s.}$$

(7) is (8) at $S = \emptyset$. Both are **conditional mean independence** statements, so
the family the reviewer points to is the natural one:

* Lundborg, Shah & Peters, *Conditional independence testing in Hilbert spaces*,
  JRSS-B **84**(5), 1821–1850 (2022) — the paper at the URL, which generalises the GCM;
* Lundborg, Kim, Shah & Samworth, *The projected covariance measure for assumption-lean
  variable significance testing*, Ann. Statist. **52**(6) (2024) — the **PCM**, which
  targets $H_0: \mathbb{E}[V \mid X, Z] = \mathbb{E}[V \mid Z]$ directly. This is the
  one that matches (8) term for term.

**The modification.** $\tau = Y(1) - Y(0)$ is never observed, so the PCM cannot be
applied off the shelf. Under randomisation with $e = \mathbb{P}(T=1)$ known by design,
the R-learner pseudo-outcome already used by our GCM variants,

$$\widehat\varphi = \frac{(Y - \widehat m(\bm Z^S))(T - e)}{e(1-e)},$$

satisfies $\mathbb{E}[\widehat\varphi \mid \bm Z] = \mathbb{E}[\tau \mid \bm Z]$ for
**any** function $\widehat m$ of $\bm Z$ ($\widehat m$ moves variance, never bias).
Hence

$$H_0(j\mid S) \iff \mathbb{E}[\varphi \mid \bm Z^{S\cup\{j\}}] = \mathbb{E}[\varphi \mid \bm Z^{S}],$$

which is precisely PCM's null with $V = \varphi$, $X = Z^j$, $Z = \bm Z^S$. That
substitution *is* the modification, and it is the only one needed.

**Why it is not redundant with what we already have.** The GCM tests
$\mathbb{E}[\mathrm{cov}(\varphi, Z^j \mid \bm Z^S)] = 0$, which is *implied by* but not
*equivalent to* $H_0(j\mid S)$. Any alternative whose conditional-mean contrast is
uncorrelated with $Z^j$ — a U-shape, a symmetric threshold — is invisible to it at
every sample size. Our appendix already flags this blind spot for the linear test
(§ "CATE-equivalence test"); the GCM as instantiated inherits it, because it
residualises $Z^j$ linearly. The PCM removes it: it learns a projection
$\widehat f(Z^j, \bm Z^S)$ of the conditional-mean contrast on one half of the sample
and forms the residual-product statistic on the other, so it is consistent against
essentially any conditional-mean alternative.

---

## 2. Implementation

`conditional_interaction_pvalues_pcm` in [nexis.py](../src/method/nexis.py), selected via
`nexis(..., test="PCM: quadratic")` or `test="PCM: lgbm"`.

1. $\varphi$ built with $\widehat m$ **cross-fitted within each half separately**.
2. Split $[n]$ into halves $A, B$. For each direction $(\text{train}, \text{test})$:
   * fit a projection $\widehat f_j$ per candidate on `train` — a polynomial in $Z^j$
     partialled out against $[1, \bm Z^S]$, vectorised over all $m$ candidates at once
     (batched $2\times2$ normal equations, so the cost stays $O(n\,m)$), and, for
     `PCM: lgbm`, a LightGBM refit of $\mathbb{E}[\varphi \mid Z^j, \bm Z^S]$ for the 32
     candidates with the largest training-half explained sum of squares;
   * on `test`, form $R_i = (\varphi_i - \widehat{\mathbb{E}}[\varphi\mid \bm Z^S_i])\cdot(\widehat f_j - \widehat{\mathbb{E}}[\widehat f_j \mid \bm Z^S_i])$
     and studentise. **One-sided**: under the alternative the projection is aligned
     with the contrast by construction, so the covariance is positive.
3. Combine the two directions by $2\cdot\min(p_1, p_2)$.

Coefficients are rescaled to unit fitted variance. Without this the projection
collapses under $H_0$ (mean *and* variance of $R$ both vanish) and the statistic is
ill-defined; rescaling fixes the direction without touching the train-measurable
magnitude.

### Three implementation choices that were forced by measurement

Measured size at $\alpha=0.05$ on the synthetic null of
[check_pcm_calibration.py](../src/method/check_pcm_calibration.py) ($n=600$, $|S|=2$;
400–600 reps for the intermediate variants, 1000 for the shipped ones):

| Choice | Naive option | size | Kept | size |
|---|---|---|---|---|
| Split combination (poly projection) | pool both halves and studentise over all $n$ (DML-style cross-fitting) | 0.090 | $2\cdot\min(p_1,p_2)$ | **0.048** |
| $\widehat m$ for $\varphi$ (lgbm projection) | cross-fit over the full sample | 0.095 | cross-fit within each half | 0.080 |
| Nuisances for `PCM: lgbm` | LightGBM | 0.080 | poly2 (Ridge, degree 2) — same power | **0.049** |

Each row holds the other two at their final setting; the three fixes compose to
0.095 → 0.049 for `PCM: lgbm`.

The first is the interesting one: cross-fitting is valid in DML because the nuisance
converges. Here the projection is *normalised noise* under $H_0$ and never converges,
so the two half-blocks stay strongly dependent and the pooled variance is understated.
`combine="crossfit"` is kept in the code but documented as invalid.

We also tried a one-term Cornish–Fisher correction for the far tail and a winsorised
basis; **both failed** and were removed (see §4).

---

## 3. Results

50 seeds per grid point, $\alpha = 0.05$, $|\mathcal{S}^\star| = 2$, SigLIP SAE
$m = 9{,}216$. $n^\star$ / $\eta^\star$ = smallest grid value reaching mean recall
$\geq 0.95$; "—" = never on the grid. Precision/Recall/IoU at $n=2000$, $\eta=5$.

### 3a. Synthetic sanity check (size and power, isolated single test)

$n=600$, $m=40$, $|S|=2$, one candidate tested; 1000 reps for size, 200 for power.
Rejection rate at $\alpha=0.05$:

| test | size ($\eta=0$) | power, linear alt. | power, U-shaped alt. | s/test |
|---|---|---|---|---|
| linear | 0.048 | **1.000** | 0.125 | 0.001 |
| GCM: quadratic | 0.029 | 0.980 | 0.035 | 0.011 |
| GCM: lgbm | 0.042 | 0.970 | 0.045 | 0.043 |
| PCM: quadratic | 0.048 | 0.790 | **0.980** | 0.022 |
| PCM: lgbm | 0.049 | 0.445 | 0.875 | 0.071 |

All five control size. On a linear alternative the ordering is the paper's; on a
U-shaped alternative the linear test and *both* GCM variants sit at their nominal
level — no power at all — while the PCM rejects almost always.

### 3b. Published benchmark, sparse codes $z$ ($\tau$ linear in the binary attributes)

| test | n*(η=2) | n*(η=5) | η*(n=500) | η*(n=2000) | Prec | Rec | IoU | s/run |
|---|---|---|---|---|---|---|---|---|
| marginal (FWER) | 2000 | 750 | 6 | 2 | 0.21±0.01 | 1.00±0.00 | 0.21±0.01 | 1.3 |
| **linear (default)** | 2000 | **500** | **5** | **2** | 1.00±0.00 | 1.00±0.00 | **1.00±0.00** | 5.3 |
| GCM: quadratic | 3500 | 2000 | — | 4 | 1.00±0.00 | 0.99±0.01 | 0.99±0.01 | 5.3 |
| GCM: lgbm | 3500 | 2000 | — | 4 | 1.00±0.00 | 0.99±0.01 | 0.99±0.01 | 4.9 |
| PCM: quadratic | 3500 | 2000 | — | 3 | 0.99±0.01 | 1.00±0.00 | 0.99±0.01 | 24.6 |
| PCM: lgbm | 3500 | 2000 | — | 4 | 1.00±0.00 | 1.00±0.00 | 1.00±0.00 | 134.1 |

The published conclusion is unchanged: **linear still dominates** on this DGP, reaching
recall 0.95 at roughly half the sample size / effect magnitude of every assumption-lean
alternative. The PCM costs nothing relative to the GCM here despite splitting the
sample — one-sidedness offsets the split — and is marginally better in $\eta^\star$ at
$n=2000$ (3 vs 4). So adding it does not disturb the paper's recommended default.

### 3c. Same DGP, dense pre-activations $z_{\mathrm{pre}}$

| test | n*(η=2) | n*(η=5) | η*(n=500) | η*(n=2000) | Prec | Rec | IoU | s/run |
|---|---|---|---|---|---|---|---|---|
| marginal (FWER) | 2000 | 500 | 5 | 2 | 0.00±0.00 | 1.00±0.00 | 0.00±0.00 | 1.5 |
| **linear (default)** | 2000 | **750** | **6** | **2** | **0.99±0.01** | **1.00±0.00** | **0.99±0.01** | 5.1 |
| GCM: quadratic | 10000 | — | — | — | 0.32±0.05 | 0.46±0.06 | 0.31±0.05 | 3.4 |
| GCM: lgbm | 10000 | — | — | — | 0.33±0.06 | 0.37±0.06 | 0.31±0.06 | 5.9 |
| PCM: quadratic | 10000 | — | — | — | 0.42±0.05 | 0.60±0.06 | 0.41±0.05 | 17.2 |
| PCM: lgbm | 10000 | 5000 | — | — | 0.51±0.06 | 0.61±0.06 | 0.47±0.06 | 108.7 |

On correlated dense features the PCM is clearly the better assumption-lean option
(recall 0.61 vs 0.37, precision 0.51 vs 0.33), though still behind linear.

### 3d. The case that motivates the reviewer's suggestion: a GCM-blind alternative

New DGP (`--effect-form ortho_quadratic`, [scm.py](../src/apps/celeba/scm.py)):
$\tau$ is driven by an orthogonalised quadratic of the two ground-truth coordinates,
constructed from the full CelebA column so that in the population
$\mathbb{E}[g(Z^j)] = 0$ **and** $\mathrm{Cov}(g(Z^j), Z^j) = 0$ exactly
(verified to $10^{-16}$). $\mathbb{E}[\tau \mid Z^j]$ genuinely varies, but no test
reading a linear functional of $Z^j$ can see it. Continuous pre-activations are
required — on sparse post-topk codes any function of $Z^j$ is effectively two-valued,
hence monotone, and the U-shape degenerates.

| test | n*(η=5) | η*(n=2000) | Prec | Rec | IoU | s/run |
|---|---|---|---|---|---|---|
| marginal (FWER) | — | — | 0.00±0.00 | 0.11±0.03 | 0.00±0.00 | 0.8 |
| linear (default) | — | — | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 11.8 |
| GCM: quadratic | — | — | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 16.6 |
| GCM: lgbm | — | — | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 | 46.5 |
| **PCM: quadratic** | 5000 | — | 0.80±0.05 | 0.87±0.04 | 0.80±0.05 | 56.9 |
| **PCM: lgbm** | **750** | **1** | **1.00±0.00** | **1.00±0.00** | **1.00±0.00** | 144.4 |

**This is the answer to the reviewer.** Every test currently in the paper —
including the "fully nonparametric" GCM with LightGBM nuisances — has *exactly zero*
recall at every $(n, \eta)$ on the grid, and does not improve with more data: they are
inconsistent against this alternative, not merely underpowered. The PCM recovers
$\mathcal{S}^\star$ perfectly from $n = 750$. It is the only instantiation in the
suite that makes Assumption (test validity + consistency) hold on this DGP.


### 3e. Replication on a disjoint seed block

Seeds 50--99 rerun independently of the primary block 0--49 (the seed *is* the draw,
so this is a second Monte-Carlo sample, not a re-run), written to
`results/celeba/experiment_rep/` and `results/celeba/experiment_ushape_rep/`.
Cells are primary / replication; $\sigma$ is the pooled Monte-Carlo standard error.

Published DGP, sparse codes:

| test | n*(η=2) | n*(η=5) | Prec @ (2000, 5) | Rec @ (2000, 5) |
|---|---|---|---|---|
| marginal (FWER) | 2000 / 2000 | 750 / 500 | 0.21 / 0.19 (−1.4σ) | 1.00 / 1.00 (+0.0σ) |
| linear (default) | 2000 / 2000 | 500 / 500 | 1.00 / 0.99 (−1.0σ) | 1.00 / 1.00 (+0.0σ) |
| GCM: quadratic | 3500 / 3500 | 2000 / 2000 | 1.00 / 1.00 (+0.0σ) | 0.99 / 1.00 (+1.0σ) |
| GCM: lgbm | 3500 / 3500 | 2000 / 2000 | 1.00 / 1.00 (+0.0σ) | 0.99 / 1.00 (+1.0σ) |
| PCM: quadratic | 3500 / 3500 | 2000 / 2000 | 0.99 / 1.00 (+1.0σ) | 1.00 / 1.00 (+0.0σ) |
| PCM: lgbm | 3500 / 5000 | 2000 / 2000 | 1.00 / 1.00 (+0.0σ) | 1.00 / 1.00 (+0.0σ) |

GCM-blind DGP, pre-activations:

| test | n*(η=5) | Prec @ (2000, 5) | Rec @ (2000, 5) |
|---|---|---|---|
| marginal (FWER) | — / — | 0.00 / 0.00 (+0.5σ) | 0.11 / 0.10 (−0.2σ) |
| linear (default) | — / — | 0.00 / 0.00 (+0.0σ) | 0.00 / 0.00 (+0.0σ) |
| GCM: quadratic | — / — | 0.00 / 0.00 (+0.0σ) | 0.00 / 0.00 (+0.0σ) |
| GCM: lgbm | — / — | 0.00 / 0.00 (+0.0σ) | 0.00 / 0.00 (+0.0σ) |
| PCM: quadratic | 5000 / 5000 | 0.80 / 0.83 (+0.4σ) | 0.87 / 0.93 (+1.3σ) |
| PCM: lgbm | 750 / 750 | 1.00 / 1.00 (+0.0σ) | 1.00 / 1.00 (+0.0σ) |

Every entry agrees within 1.4σ. The two discrepancies are both threshold crossings of
the recall-0.95 rule on a coarse grid (marginal 750→500, `PCM: lgbm` 3500→5000), not
differences in the underlying curves.

### 3f. Type-I error control ($\eta = 0$, so any selection is a false discovery)

Empirical FWER $\widehat{\mathbb{P}}(\widehat{\mathcal{S}}_n \neq \emptyset)$, 50 seeds,
nominal $\alpha = 0.05$:

| features | test | n=500 | n=2000 | n=5000 |
|---|---|---|---|---|
| z (codes) | all six | 0.00±0.00 | 0.00±0.00 | 0.00±0.00 |
| z_pre | marginal (FWER) | 0.04±0.03 | 0.02±0.02 | 0.00±0.00 |
| z_pre | linear (default) | 0.04±0.03 | 0.02±0.02 | 0.00±0.00 |
| z_pre | GCM: quadratic | 0.06±0.03 | 0.02±0.02 | 0.00±0.00 |
| z_pre | GCM: lgbm | 0.06±0.03 | 0.02±0.02 | 0.00±0.00 |
| z_pre | PCM: quadratic | 0.04±0.03 | 0.06±0.03 | **0.10±0.04** |
| z_pre | PCM: lgbm | **0.10±0.04** | 0.06±0.03 | **0.12±0.05** |

---

## 4. Honest limitation to report with the PCM

On **dense pre-activations** the PCM's empirical FWER drifts above nominal as $n$ grows
(0.10–0.12 at $n = 5000$ vs 0.00–0.02 for the GCM). On sparse codes — the paper's
default representation — it is 0.00 everywhere.

Diagnosis (8 seeds $\times$ 9,216 null tests per cell, $S = \emptyset$, $z_\mathrm{pre}$):

| test | rejections at α/m = 5.4e-6 | at 1e-4 | at 1e-3 |
|---|---|---|---|
| expected | 0.40 | 7.4 | 74 |
| GCM: quadratic, n=5000 | 0 | 2 | 62 |
| PCM: quadratic, n=2000 | 0 | 2 | 37 |
| PCM: quadratic, n=5000 | 2 | 11 | 100 |

So the bulk of the p-value distribution is fine and the excess is confined to the far
tail that the Bonferroni gate $\alpha/m \approx 5\times10^{-6}$ reads. Two candidate
fixes were implemented and **both rejected on measurement**:

* **winsorising the polynomial basis at $\pm4$ SD** (hypothesis: an unbounded $x^2$ on
  skewed pre-activations) — no effect (100 → 100 at $10^{-3}$);
* **one-term Cornish–Fisher correction** $T + \widehat\gamma(2T^2+1)/(6\sqrt n)$
  (hypothesis: third-order skewness) — made it *worse*, 100 → 141.

The residual excess is therefore driven by a handful of high-leverage observations
in the fitted projection, where neither bounding the basis nor an Edgeworth expansion
applies. A multiplier bootstrap of the null would be the principled remedy and is the
obvious next step if this matters; we did not implement it.

The practical recommendation is unchanged and now has a sharper boundary:

* **linear stays the default** — it dominates on the benchmark as published (3b, 3c);
* **switch to the PCM, not the GCM,** when nonlinear heterogeneity in the dictionary
  cannot be ruled out: the GCM buys robustness to nuisance misspecification but is
  still blind to non-monotone conditional-mean alternatives (3a, 3d);
* on dense/unaudited representations at large $n$, read PCM selections with the
  tail caveat of §4 in mind.

---

## 5. Reproducing

```bash
# test ablation with the PCM merged into the published sweeps
sbatch scripts/celeba/run_experiment_pcm.sh    sae         20 effect siglip
sbatch scripts/celeba/run_experiment_pcm.sh    sae         20 n      siglip
sbatch scripts/celeba/run_experiment_pcm.sh    sae_precode 20 effect siglip
sbatch scripts/celeba/run_experiment_pcm.sh    sae_precode 20 n      siglip
# GCM-blind DGP
sbatch scripts/celeba/run_experiment_ushape.sh effect
sbatch scripts/celeba/run_experiment_ushape.sh n
# type-I control
sbatch scripts/celeba/run_experiment_type1.sh  sae
sbatch scripts/celeba/run_experiment_type1.sh  sae_precode

# replication on a disjoint seed block (50-99)
sbatch scripts/celeba/run_experiment_replicate.sh attr   effect 50
sbatch scripts/celeba/run_experiment_replicate.sh attr   n      50
sbatch scripts/celeba/run_experiment_replicate.sh ushape effect 50
sbatch scripts/celeba/run_experiment_replicate.sh ushape n      50

# THE table for the paper (two panels, Table~\ref{tab:celeba:test_ablation})
python src/apps/celeba/table_test_ablation.py --dgp paper

# supporting tables (Markdown + LaTeX → results/celeba/appendix/)
python src/apps/celeba/table_test_ablation.py --dgp attr            --feat sae
python src/apps/celeba/table_test_ablation.py --dgp attr            --feat sae_precode
python src/apps/celeba/table_test_ablation.py --dgp ortho_quadratic
python src/apps/celeba/table_test_ablation.py --dgp type1

# calibration / power micro-benchmarks used in §2 and §4
python src/method/check_pcm_calibration.py   # size + power vs linear/GCM, synthetic
python src/apps/celeba/compare_replication.py --dgp attr
python src/apps/celeba/compare_replication.py --dgp ushape
```

Figure `results/celeba/appendix/method_test.pdf` has been regenerated with both PCM
variants added to the `test` ablation group.
