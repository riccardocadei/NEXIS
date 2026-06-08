# Generalised Heterogeneous Treatment Effect (HTE) Identification

**TL;DR:** *Generalised identification of what drives heterogeneous treatment effects — combining complex pre-treatment measurements with interpretable domain priors.*

Understanding *why* treatment effects vary across individuals is a fundamental challenge in causal inference. Standard methods predict effect variation but cannot reliably identify which specific features actually drive the heterogeneity, nor do they offer statistical guarantees. To address this, we introduce **NEXIS** (Neural Exposure Interaction Search): a framework providing a powerful **hypothesis generation** component. It sequentially tests and selects genuine effect modifiers from a vast pool of candidates while strictly controlling the family-wise error rate (FWER).

Our approach is generalised in the sense that the input feature space is unrestricted and can freely combine (i) complex, high-dimensional pre-treatment measurements such as representations from foundation models (satellite imagery, medical imaging), and (ii) interpretable prior variables (demographics, administrative records). By linking mechanistic interpretability with causal inference, NEXIS bridges the gap between opaque deep learning features and statistically rigorous hypothesis generation.

### Problem setup

<table>
<tr>
<td valign="middle" width="62%">

Consider a randomised experiment with treatment **T**, outcome **Y**, and pre-treatment observations **X**. We posit a set of effect-modification factors **W** (some latent and some partially observed). Our framework is **generalised** as it aims to identify effects directly by combining complex measurements with domain priors, both of which serve as observable manifestations of **W** acting on **X** and driving the heterogeneous response to treatment (see figure).

The pre-treatment input **X** is the union of two complementary sources:

- **Complex measurements** — high-dimensional representations extracted from raw data (satellite imagery, sensor readings, omics), typically via a foundation model + Sparse Autoencoder, yielding thousands of candidate neurons
- **Interpretable priors** — measured baseline variables the researcher already has (demographics, survey items, administrative records), entered directly as additional candidates

NEXIS screens the combined candidate set for treatment effect modification, conditioning each new test on the features already selected and applying a Bonferroni gate, so that FWER is controlled throughout regardless of the total number of candidates.

*Shaded nodes are observed; the node **W** can be latent or partially observed.*

</td>
<td valign="middle" align="center" width="38%">
<img src="assets/causal_model.png" width="260" alt="Causal model: observed nodes T, Y, X in grey; modifier W in white"/>
<br/>
<sub><b>Causal model:</b> Some pre-treatment variables <b>W</b>, latent (entangled in a complex measurement <b>X</b>) or partially observed, drive the heterogeneous response to treatment <b>T</b> on outcome <b>Y</b>.</sub>
</td>
</tr>
</table>

### Motivating example — Uganda Youth Opportunities Programme

A concrete instantiation pairs randomised experiments with satellite imagery: given an RCT with treatment `T` and outcomes `Y`, and pre-treatment satellite imagery from unit locations, modern vision models (e.g. Prithvi, DINOv2, DINOv3) extract rich spatial features. Sparse Autoencoders then map these dense embeddings to interpretable individual neurons. Given the resulting high-dimensional representation `Z`, the question becomes: *which learned features meaningfully interact with treatment to drive outcome differences?* NEXIS provides a principled selection, while simultaneously screening explicit covariates alongside the learned deep features.

---

## Method

The pipeline proceeds in three distinct stages.

### Step 1 — Representation learning

Raw pre-treatment observations (e.g. satellite imagery) are first passed through a **foundation model** — such as Prithvi (geospatial) or DINOv2/DINOv3 (vision) — to obtain dense, high-dimensional patch embeddings. These embeddings are then fed to a **Sparse Autoencoder (SAE)**, which decomposes the dense representation into a large number of sparse, near-monosemantic neurons. Each neuron captures a specific, human-interpretable visual concept (e.g. *presence of water*, *road density*, *vegetation type*). The result is a high-dimensional but structured feature matrix $Z \in \mathbb{R}^{n \times p}$, with $p$ potentially reaching thousands of candidates, that forms the input to the selection stage.

### Step 2 — Neural Exposure Interaction Search

Given the combined candidate matrix $Z$ (SAE neurons + any additional measured covariates) and an RCT or observational dataset $(Y, T)$, NEXIS iteratively selects a feature or neuron $j$ by testing the conditional interaction hypothesis:

$$
\mathcal{H}_0(j \mid S) : \quad \gamma_j = 0 \quad \text{in} \quad Y \sim 1 + T + Z_S + T \cdot Z_S + Z_j + T \cdot Z_j
$$

The null hypothesis $\mathcal{H}_0$ conditions on the already-selected set $S$. At each step, a Bonferroni gate is applied uniformly over all remaining candidates, sequentially narrowing the search while guaranteeing tight control over the family-wise error rate (FWER) throughout. Selection automatically halts when no remaining candidate clears the gated significance threshold.

```python
from src.method import nexis

result = nexis(y=Y, t=T, z=Z, alpha=0.05)
print(result.selected)   # list of selected neuron indices
```

### Step 3 — Interpretation

Once NEXIS selects a set of neurons, each selected neuron is interpreted using a **Vision-Language Model (VLM)** pipeline. For every selected neuron $j$, the top-activating and bottom-activating satellite patches are retrieved and passed to a VLM (e.g. Qwen-VL, GeoChat) with a structured prompt asking what visual concept the neuron responds to. The per-patch captions are then aggregated by an LLM into a concise, human-readable description (e.g. *"areas without perennial water sources"*). This yields a fully interpretable summary of each effect modifier, grounding statistically selected neurons in domain-meaningful language.

---

## Experiments

### Uganda Youth Opportunities Programme

![Uganda study sites](results/uganda/map.png)

We apply NEXIS to the Uganda YOP, a cash-and-training RCT in northern Uganda ([Blattman, Fiala & Martinez, 2014](https://chrisblattman.com/documents/research/2014.GeneratingSkilledEmployment.QJE.pdf)). We pair each participant with pre-treatment satellite imagery (year 2000) and extract learned features using **Prithvi** (geospatial foundation model), **DINOv2**, and **DINOv3**. A Sparse Autoencoder trained on top maps the dense embedding to sparse, interpretable neurons; NEXIS then screens these neurons — together with any additional measured covariates — for treatment effect modification.

For the primary outcome **log skilled-trade hours** (n = 2,372, ATE = +0.020, p < 0.001), NEXIS selects 2 effect modifiers:

| Rank | Feature | Interpretation | GATE (inactive) | GATE (active) | Δ CATE | p-value |
|------|---------|----------------|----------------|--------------|--------|---------|
| 1 | SAE\_659 | No perennial water source | −0.001 | +0.038 | **+0.039\*** | 3.0 × 10⁻⁶ |
| 2 | lang\_6  | Lugbara language region    | +0.030 | −0.003 | **−0.033\*** | 1.8 × 10⁻³ |

> \* 95% CI of Δ CATE excludes zero.

The programme is substantially more effective in drier areas without perennial water — a finding invisible to average-effect analysis and not surfaced by prior work on this trial. Below: GATE estimates, geographic distribution, treatment balance, and the satellite patches most/least activating each selected neuron (Prithvi encoder).

![NEXIS results — skilled employment (Prithvi)](results/uganda/figures/figure_neural_skilled_employed.png)

**To reproduce**, note that notebooks are primarily for visualisation: [notebooks/uganda.ipynb](notebooks/uganda.ipynb). The actual reproducible end-to-end experiments (supporting eight outcomes like labour, earnings, assets, wellbeing) run via the pipeline script:

```bash
bash scripts/uganda/run.sh --models=prithvi,dinov2,dinov3 --all-outcomes
```

### Ghana LEAP 1000

![Ghana study sites](results/ghana/map/map_paper.png)

We apply NEXIS to the Ghana Livelihood Empowerment Against Poverty 1000 (LEAP 1000) programme, a cluster-randomised cash-transfer trial targeting extremely poor households in Northern and Upper East Ghana ([ISSER, 2018](https://www.unicef.org/ghana/reports/ghana-leap-1000-evaluation)). The outcome is adult-equivalent household consumption expenditure per month (n = 2,331, 162 communities). Pre-treatment satellite imagery is extracted using **DINOv2**; a Sparse Autoencoder maps the embeddings to 4,096 sparse neurons.

NEXIS selects 1 effect modifier:

| Rank | Feature | Interpretation | GATE (inactive) | GATE (active) | Δ CATE | p-value |
|------|---------|----------------|----------------|--------------|--------|---------|
| 1 | SAE\_1777 | Presence of water infrastructure | — | — | **significant** | < 0.05 |

The selected neuron highlights areas with surface water infrastructure as a key moderator of the cash-transfer effect on household welfare.

![NEXIS results — Ghana consumption (DINOv2)](results/ghana/figures/figure_neural_ghana_combined.png)

**To reproduce**: [notebooks/ghana.ipynb](notebooks/ghana.ipynb).

### CelebA — semi-synthetic benchmark

To validate FWER control and power, we construct a semi-synthetic RCT benchmark on CelebA face images with **2 known direct effect modifiers** (*wearing a hat*, *wearing eyeglasses*). A Sparse Autoencoder (13,824 codes) trained on SigLIP representations provides the candidate dictionary. We sweep over effect size and sample size and compare NEXIS against marginal interaction screening (unadjusted and Bonferroni-adjusted).

Marginal screening exhibits a **precision collapse** as power grows — accumulating indirect modifiers correlated with the true ones. NEXIS consistently recovers the true modifier set by iterative conditioning.

**To reproduce**: [notebooks/celeba.ipynb](notebooks/celeba.ipynb).

---

## Repository structure

```
src/
  method/
    nexis.py        # core NEXIS algorithm, CATE-equivalence tests, evaluation utilities
  causality/
    estimation.py   # ATE/GATE estimation utilities
  apps/
    celeba/         # CelebA semi-synthetic benchmark (embed, train SAE, run experiments)
    ghana/          # Ghana LEAP 1000 pipeline (data, embed, interpret, visualize)
    synthetic/      # synthetic DGP and sweep scripts
    uganda/         # Uganda YOP pipeline (data, embed, interpret, summarize, visualize)

notebooks/
  synthetic.ipynb   # synthetic benchmark (effect size & sample size sweeps)
  celeba.ipynb      # CelebA semi-synthetic benchmark
  uganda.ipynb      # Uganda YOP real-data analysis and visualisation
  ghana.ipynb       # Ghana LEAP 1000 real-data analysis and visualisation

scripts/
  uganda/
    run.sh          # full Uganda pipeline (embedding → SAE → NEXIS → interpret → summarize → plot)
    reanalyze.sh    # re-run Uganda analysis steps only (skips embedding/SAE training)
  ghana/            # analogous scripts for Ghana
  celeba/           # analogous scripts for CelebA benchmark

assets/
  NEXIS.pdf               # full paper (NeurIPS 2026 submission)
  aistats26-workshop.pdf  # AISTATS 2026 workshop version
  causal_model.png        # causal DAG figure used in README

results/
  uganda/
    map/                         # study site maps
    figures/                     # summary figures for paper
    {model}_{dim}/{outcome}/     # NEXIS results per (model, outcome)
  ghana/
    map/  gate/  figures/        # maps, GATE estimates, selected-neuron figures
  celeba/
    experiment/{k}/{encoder}/    # power/FWER sweeps per (k, encoder) configuration

data/              # real-world datasets (not tracked by git)
```

---

## Citation

If you use NEXIS, please cite our paper (preprint coming soon; see [assets/aistats26-workshop.pdf](assets/aistats26-workshop.pdf) for the workshop version):

```bibtex
@inproceedings{cadei2026nexis,
  title     = {From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification},
  author    = {Cadei, Riccardo and Otchere, Frank and Tirivayi, Nyasha and
               Angeles Tagliaferro, Gustavo and Bargagli-Stoffi, Falco J. and Locatello, Francesco},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  note      = {Preprint available at TODO}
}
```


