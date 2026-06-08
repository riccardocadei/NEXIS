# From Tokens to Policy: Causal and Interpretable Heterogeneous Treatment Effects Identification

**TL;DR:** *NEXIS identifies which features causally drive treatment effect heterogeneity — combining foundation-model representations of complex observations with a principled sequential selection procedure that controls false discoveries.*

Real-world interventions rarely work the same way for everyone. Understanding *why* and *how* a treatment effect varies is essential to optimise policies accordingly. Existing HTE methods trade expressivity for interpretability, but as long as some active heterogeneity drivers are unmeasured, both ends of this spectrum allow for spurious characterisations with no causal reading.

We argue that **causal HTE identification** is now within reach, thanks to (i) more extensive pre-treatment measurements (multi-modal, satellite imagery, sensor streams) and (ii) representation learning pipelines that scale their analysis. We re-frame HTE identification as a **Markov-blanket discovery problem** on a sufficient and aligned pre-treatment representation, and introduce **NEXIS** (Neural EXposure Interaction Search): an iterative forward-backward procedure with provable consistent selection.

We deploy NEXIS on two anti-poverty programs in Africa, augmenting each with satellite imagery capturing previously unmeasured environmental effect modifiers, and producing novel interpretable and prescriptive guidelines for program optimisation.

### Problem setting and effect modification taxonomy

<table>
<tr>
<td valign="middle" width="60%">

Consider a controlled experiment with treatment **T**, outcome **Y**, and pre-treatment observations **X**. The treatment effect heterogeneity is causally explained by **direct effect modifiers** W<sup>dir</sup> — latent factors that interact with treatment to drive variation in Y. However, W<sup>dir</sup> is rarely measured directly, and other spurious correlates arise:

- **W<sup>dir</sup>** — *direct* modifiers: have a causal pathway to τ; the only ones licensing policy intervention
- **W<sup>ind</sup>** — *indirect* modifiers: ancestors of W<sup>dir</sup>; operate only via mediation
- **W<sup>prx</sup>** — *proxies*: descendants of W<sup>dir</sup>; no causal pathway to τ
- **W<sup>cc</sup>** — *common-cause* modifiers: share a common ancestor with W<sup>dir</sup>

W<sup>dir</sup> is entangled in complex pre-treatment observations **X** (e.g. satellite imagery). A representation map ψ: X → Z (foundation model + Sparse Autoencoder) yields thousands of candidate neurons. NEXIS screens Z for the principal proxies of W<sup>dir</sup>, conditioning each new test on already-selected features and applying a Bonferroni gate for FWER control.

*Gray nodes: observed. White nodes: latent.*

</td>
<td valign="middle" align="center" width="40%">
<img src="assets/causal_model.png" width="280" alt="Causal model showing T, Y, X, Z (observed) and W^dir, W^ind, W^prx, W^cc (latent) with effect modification taxonomy"/>
<br/>
<sub><b>Effect modification taxonomy</b> (Van der Weele 2007). Only <b>W<sup>dir</sup></b> carries a causal interpretation and licenses policy intervention.</sub>
</td>
</tr>
</table>

---

## Method

The pipeline proceeds in three distinct stages.

### Step 1 — Representation learning

Raw pre-treatment observations (e.g. satellite imagery) are first passed through a **foundation model** — such as Prithvi (geospatial) or DINOv2/DINOv3 (vision) — to obtain dense, high-dimensional patch embeddings. These embeddings are then fed to a **Sparse Autoencoder (SAE)**, which decomposes the dense representation into a large number of sparse, near-monosemantic neurons. Each neuron captures a specific, human-interpretable visual concept (e.g. *presence of water*, *road density*, *vegetation type*). The result is a high-dimensional but structured feature matrix $Z \in \mathbb{R}^{n \times p}$, with $p$ potentially reaching thousands of candidates, that forms the input to the selection stage.

### Step 2 — Neural Exposure Interaction Search

Given Z (SAE neurons + any measured covariates) and an experiment (Y, T), NEXIS runs a forward-backward Markov-blanket discovery loop. At each step it tests the **CATE-equivalence null**:

$$
H_0(j \mid S) : \quad \mathbb{E}[\tau \mid \bm{Z}^{S \cup \{j\}}] = \mathbb{E}[\tau \mid \bm{Z}^{S}] \quad \text{a.s.}
$$

- **Forward**: add the candidate j\* with smallest p-value if it clears the Bonferroni gate α/|S̄|
- **Backward**: re-test each selected coordinate and drop any that became redundant given the rest
- Iterate until convergence; FWER ≤ α throughout

Under Measurement and Representation Sufficiency, the output S\* identifies the direct modifier CATE: τ(W<sup>dir</sup>) = E[τ | Z<sup>S\*</sup>] a.s.

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


