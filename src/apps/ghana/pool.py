"""The Ghana LEAP 1000 candidate pools NEXIS runs on, with their data hierarchy.

Balanced panel (households seen in both waves), outcome = first-differenced monthly
consumption (2017 - 2015), T = LEAP household.  Two pools, searched in one pass each:

  ghana()                  155 = 131 SAE neurons (active in >= 5 of the 162
                           communities, data/ghana/satellite/sae_activations.npy) + 24
                           household survey covariates, the pool of the June runs;
  ghana_with_spectral(d)   167 = those 155 + 12 community-level spectral indices
                           (data/ghana/satellite/spectral_indices.csv), the pool the
                           paper's main text describes.

The published set on the 155 pool is {Z_3821, Z_2095}.  The June run in
results/ghana/codes/nexis_fwer_crve/result.json passed the survey covariates through a
former two-phase form of nexis() whose W-only phase selected nothing, so it searched the
131 neurons alone (m = 131); its looser gates also admitted Z_3318.  nexis() no longer
has that form, so that file is not the reference here.

Imported by src/apps/ghana/{final_runs,table_gate}.py; needs src/ on sys.path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from apps.ghana.data import load_data, W_ALL
from nexis.multilevel import codes

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data" / "ghana"


def _panel() -> pd.DataFrame:
    """Baseline rows of the households seen in both waves, with the endline outcome Y1."""
    df = load_data(DATA_DIR)
    both = df.groupby("hhid")["wave"].nunique()
    df = df[df["hhid"].isin(both[both == 2].index)]
    df0, df1 = df[df["wave"] == 0], df[df["wave"] == 1]
    return (df0.set_index("hhid")[["T", "comm", "district", "region", "Y"] + W_ALL]
            .join(df1.set_index("hhid")[["Y"]].rename(columns={"Y": "Y1"}))
            .dropna(subset=["Y1"]))


def ghana() -> dict:
    """The 155 pool as a multilevel dict (see src/nexis/multilevel.py): levels
    region > district > community > household; T assigned by household."""
    merged = _panel()
    comm = merged["comm"].values
    act = np.load(DATA_DIR / "satellite" / "sae_activations.npy")
    ids = np.load(DATA_DIR / "satellite" / "sae_comm_ids.npy")
    live = (act > 0).sum(axis=0) >= 5
    live_idx = np.where(live)[0]
    row = merged["comm"].map(dict(zip(ids, range(len(ids))))).values
    Z = act[:, live][row]                       # 131 SAE neurons, community-level
    W = merged[W_ALL].values.astype(float)      # 24 survey covariates, household-level
    levels = {"region": codes(merged["region"]), "district": codes(merged["district"]),
              "community": codes(comm), "household": np.arange(len(merged))}
    return dict(app="ghana", outcome="consumption",
                y=(merged["Y1"] - merged["Y"]).values.astype(float),
                t=merged["T"].values.astype(float), z=np.hstack([Z, W]),
                names=[f"Z_{k}" for k in live_idx] + [f"W_{c}" for c in W_ALL],
                comm_id=comm, published=["Z_3821", "Z_2095"], levels=levels,
                order=["region", "district", "community", "household"],
                assign="household", block=None, community="community")


def ghana_with_spectral(d: dict) -> dict:
    """The 167 pool: the 155 of ghana() plus the 12 community-level spectral indices."""
    spec = pd.read_csv(DATA_DIR / "satellite" / "spectral_indices.csv").set_index("comm_id")
    S = spec.loc[d["comm_id"]].values.astype(float)
    assert np.isfinite(S).all()
    return dict(d, z=np.hstack([d["z"], S]), names=d["names"] + [f"S_{c}" for c in spec.columns])
