"""Check that the real-world NEXIS selections survive the new default.

The paper's Uganda YOP and Ghana LEAP 1000 selections were produced with the
interleaved backward step and no terminal step (nexis defaults at the time:
backward=True, terminal_filter=False).  The new default is the forward step
followed by the terminal backward step, nexis(..., backward=False,
terminal_filter=True).  For every outcome the paper reports, this script

  0. re-runs the published configuration and asserts it returns the published set,
  (a) re-runs the published search with the terminal step appended,
  (b) runs the new default (forward step + terminal step) at the application's rho,

and, for every coordinate of each candidate set S~, reports the worst subset
p-value max_{A ⊆ S~\\{j}} p(j | A) against the terminal threshold alpha/m.

CPU only, a few seconds.  Reads data/ and results/ (both local, not tracked).

    python scripts/verify_new_default_realworld.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.method.nexis import nexis, conditional_interaction_pvalues  # noqa: E402

ALPHA = 0.05
OUT = ROOT / "results" / "new_default_realworld"


# ── Data: Uganda (as src/apps/uganda/analyze.py, prithvi_l5 / 1024) ───────────

def uganda_data(outcome: str):
    from apps.uganda.analyze import build_covariates
    from apps.uganda.data import resolve_outcome

    model_dir = ROOT / "results" / "uganda" / "prithvi_l5_1024"
    data_dir = ROOT / "data" / "uganda"
    df = pd.read_csv(data_dir / "UgandaDataProcessed.csv", low_memory=False)
    df = df.rename(columns={"Wobs": "T", resolve_outcome(outcome): "Y"})

    Z_all = np.load(model_dir / "individual_features.npz")["features"]
    site = np.load(model_dir / "site_features.npz")["site_features"]
    active = (site > 0).sum(axis=0) >= 5
    sae_idx = np.where(active)[0]
    Z_all = Z_all[:, active]

    mask = df["Y"].notna() & np.isfinite(Z_all[:, 0])
    df = df[mask].reset_index(drop=True)
    Z = Z_all[mask]

    W_df = build_covariates(df)
    spec = pd.read_csv(data_dir / "satellite" / "rct" / "spectral_indices.csv").set_index("site_key")
    spec_mat = np.full((len(df), spec.shape[1]), np.nan)
    for i, key in enumerate(df["geo_long_lat_key"].values):
        if pd.notna(key) and int(key) in spec.index:
            spec_mat[i] = spec.loc[int(key)].values
    W_df = pd.concat([W_df, pd.DataFrame(spec_mat, columns=spec.columns, index=df.index)], axis=1)

    Z_full = np.hstack([Z, W_df.values.astype(float)])
    names = [f"Z_{k}" for k in sae_idx] + [f"W_{c}" for c in W_df.columns]
    y = df["Y"].values.astype(float)
    t = df["T"].values.astype(float)
    published = json.loads((model_dir / outcome / "nexis_result.json").read_text())
    pub = [s["label"] for s in published["nexis_fwer"]["selected"]]
    run_kw = dict(alpha=ALPHA, max_rounds=20, adjust="FWER", rho=0.5)
    pv_kw: dict = {}
    return dict(y=y, t=t, z=Z_full, names=names, published=pub,
                run_kw=run_kw, pv_kw=pv_kw, rho_new=0.5)


# ── Data: Ghana (as notebooks/ghana.ipynb, the run behind results/ghana/codes/nexis_fwer_crve) ─

def ghana_data():
    from src.apps.ghana.data import load_data, W_ALL, W_LABELS

    data_dir = ROOT / "data" / "ghana"
    df = load_data(data_dir)
    both = df.groupby("hhid")["wave"].nunique()
    df = df[df["hhid"].isin(both[both == 2].index)]
    df0, df1 = df[df["wave"] == 0], df[df["wave"] == 1]
    merged = (df0.set_index("hhid")[["T", "comm", "Y"] + W_ALL]
              .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"}))
              .dropna(subset=["Y1"]))
    y = (merged["Y1"] - merged["Y"]).values.astype(float)
    t = merged["T"].values.astype(float)
    comms = merged["comm"].values

    act = np.load(data_dir / "satellite" / "sae_activations.npy")
    ids = np.load(data_dir / "satellite" / "sae_comm_ids.npy")
    live = (act > 0).sum(axis=0) >= 5
    live_idx = np.where(live)[0]
    row = merged["comm"].map(dict(zip(ids, range(len(ids))))).values
    Z = act[:, live][row]                       # 131 SAE neurons, community-level
    W = merged[W_ALL].values.astype(float)      # 24 survey covariates (preliminary phase)

    published = json.loads((ROOT / "results" / "ghana" / "codes" / "nexis_fwer_crve"
                            / "result.json").read_text())
    pub = [f"Z_{e['neuron_idx']}" for e in published["selected_z"]] + \
          [f"W_{e['label']}" for e in published["selected_w"]]
    run_kw = dict(alpha=ALPHA, adjust="FWER", cluster=comms, rho=0.5,
                  w=W, w_names=[W_LABELS.get(c, c) for c in W_ALL])
    return dict(y=y, t=t, z=Z, names=[f"Z_{k}" for k in live_idx], published=pub,
                run_kw=run_kw, pv_kw=dict(cluster=comms), rho_new=None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def worst_subset_p(y, t, z, S, j, pv_kw):
    """max over A ⊆ S\\{j} of p(j | A), and the maximising A."""
    others = [s for s in S if s != j]
    worst, arg = -1.0, None
    for r in range(len(others) + 1):
        for A in combinations(others, r):
            p = float(conditional_interaction_pvalues(y=y, t=t, z=z, S=list(A),
                                                      candidates=[j], **pv_kw)[j])
            if p > worst:
                worst, arg = p, list(A)
    return worst, arg


def labels(res, names):
    return [names[i] for i in res.selected]


def check(app, outcome, d):
    y, t, z, names = d["y"], d["t"], d["z"], d["names"]
    kw = d["run_kw"]
    m = z.shape[1]

    pub_run = nexis(y, t, z, backward=True, terminal_filter=False, **kw)
    got = labels(pub_run, names)
    if sorted(got) != sorted(d["published"]):
        raise SystemExit(f"[{app}/{outcome}] published run NOT reproduced: "
                         f"got {got}, expected {d['published']}")
    assert pub_run.metadata["m"] == m, "W-phase selected something; m would change"

    a = nexis(y, t, z, backward=True, terminal_filter=True, **kw)
    kw_b = dict(kw, rho=d["rho_new"])
    b = nexis(y, t, z, backward=False, terminal_filter=True, **kw_b)
    rows = {"published": got, "a_published+terminal": labels(a, names),
            "b_forward+terminal": labels(b, names),
            "b_forward_only_S~": [names[i] for i in b.metadata["terminal_candidates"]]}
    extra = {}
    if d["rho_new"] != kw["rho"]:
        b2 = nexis(y, t, z, backward=False, terminal_filter=True, **kw)
        rows[f"b_forward+terminal_rho{kw['rho']}"] = labels(b2, names)
        rows[f"b_forward_only_S~_rho{kw['rho']}"] = [names[i] for i in b2.metadata["terminal_candidates"]]
        extra["b_rho_published"] = b2

    gate = ALPHA / m
    worst = {}
    for tag, S in [("published S~", pub_run.selected),
                   ("forward-only S~", b.metadata["terminal_candidates"])]:
        for j in S:
            p, A = worst_subset_p(y, t, z, S, j, d["pv_kw"])
            worst[f"{tag}: {names[j]}"] = {"worst_p": p, "at_A": [names[k] for k in A],
                                           "kept": bool(p <= gate)}

    print(f"\n=== {app} / {outcome}   m = {m}   alpha/m = {gate:.3e}")
    for k, v in rows.items():
        print(f"  {k:36s} {v}")
    for k, v in worst.items():
        print(f"  worst p  {k:36s} {v['worst_p']:.3e}  at A={v['at_A']}  "
              f"{'kept' if v['kept'] else 'DROPPED'}")
    return {"m": m, "gate": gate, **rows, "worst": worst}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {}
    for outcome in ["skilled_employed", "log_biz_assets"]:
        report[f"uganda/{outcome}"] = check("uganda", outcome, uganda_data(outcome))
    report["ghana/consumption"] = check("ghana", "consumption", ghana_data())
    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved → {OUT / 'report.json'}")


if __name__ == "__main__":
    main()
