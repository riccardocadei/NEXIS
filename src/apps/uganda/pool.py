"""The Uganda YOP candidate pool NEXIS runs on (m = 170), with its data hierarchy.

146 SAE atoms of results/uganda/prithvi_l5_1024 (active at >= 5 of the 331 sites) and
24 covariates (age, female, father's and mother's education, group_female, the 7
language-group dummies, the 12 spectral indices), in one matrix.  T = grant received
(Wobs).  The published sets are read from
results/uganda/prithvi_l5_1024/<outcome>/nexis_result.json (`nexis_fwer`).

Imported by src/apps/uganda/{final_runs,multilevel_groupcluster,district_sensitivity,
table_gate}.py; needs src/ on sys.path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from apps.uganda.analyze import build_covariates
from apps.uganda.data import resolve_outcome
from nexis.multilevel import codes

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "uganda"
MODEL_DIR = ROOT / "results" / "uganda" / "prithvi_l5_1024"


def uganda(outcome: str) -> dict:
    """The pool for `outcome` (skilled_employed or log_biz_assets) as a multilevel dict
    (see src/nexis/multilevel.py): y, t, z, names, the lottery assignment, the
    published set and the levels region > district > community > group > individual."""
    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    df = df.rename(columns={"Wobs": "T", resolve_outcome(outcome): "Y"})

    Z_all = np.load(MODEL_DIR / "individual_features.npz")["features"]
    site = np.load(MODEL_DIR / "site_features.npz")["site_features"]
    active = (site > 0).sum(axis=0) >= 5
    sae_idx = np.where(active)[0]
    Z_all = Z_all[:, active]

    mask = df["Y"].notna() & np.isfinite(Z_all[:, 0])
    df = df[mask].reset_index(drop=True)
    Z = Z_all[mask]

    W_df = build_covariates(df)
    spec = pd.read_csv(DATA_DIR / "satellite" / "rct" / "spectral_indices.csv").set_index("site_key")
    spec_mat = np.full((len(df), spec.shape[1]), np.nan)
    for i, key in enumerate(df["geo_long_lat_key"].values):
        if pd.notna(key) and int(key) in spec.index:
            spec_mat[i] = spec.loc[int(key)].values
    W_df = pd.concat([W_df, pd.DataFrame(spec_mat, columns=spec.columns, index=df.index)], axis=1)

    published = json.loads((MODEL_DIR / outcome / "nexis_result.json").read_text())
    levels = {"region": codes(df["lang_group"]), "district": codes(df["district"]),
              "community": codes(df["geo_long_lat_key"]), "group": codes(df["groupid"]),
              "individual": np.arange(len(df))}
    return dict(app="uganda", outcome=outcome,
                y=df["Y"].values.astype(float), t=df["T"].values.astype(float),
                z=np.hstack([Z, W_df.values.astype(float)]),
                names=[f"Z_{k}" for k in sae_idx] + [f"W_{c}" for c in W_df.columns],
                assigned=df["assigned"].values.astype(float),
                published=[s["label"] for s in published["nexis_fwer"]["selected"]],
                levels=levels,
                order=["region", "district", "community", "group", "individual"],
                assign="group", block="district", community="community")
