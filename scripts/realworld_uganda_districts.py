"""Uganda YOP district-dummy sensitivity analysis, with the paper's NEXIS configuration.

The Uganda pool of the paper (scripts/realworld_clustered_nexis.py::uganda, m = 170)
holds 7 language-group dummies W_lang_1..7.  This sensitivity analysis replaces them,
in place, with the 14 district dummies W_district_<NAME> (m = 177), everything else
unchanged: 146 SAE atoms, age, female, father's and mother's education, group_female
and the 12 spectral indices; T = grant received (Wobs); linear T x Z_j test with
homoskedastic OLS SEs, no clustering, no support gate (the paper's
`Wobs | published test` run of scripts/realworld_final_runs.py).

It runs both NEXIS variants, rho = 0.5, alpha = 0.05, FWER forward gate:

  published   nexis(backward=True,  terminal_filter=False)   (NeurIPS, June run)
  new default nexis(backward=False, terminal_filter=True)    (the paper's algorithm)

The published variant must reproduce the June run behind the brief's p ~ 7.7e-5,
src/apps/uganda/analyze.py --district-dummies --out-suffix _districts
(results/uganda/prithvi_l5_1024/<outcome>_districts/nexis_result.json, `nexis_fwer`),
whose p-values are p(j | S \\ {j}) on the final set S.  The script checks the pool
names, the selected set and those p-values against that file when it exists.

For every district dummy it also reports the marginal p, the p given the final set
and the p given the candidate set S~, against the forward gate alpha/m and the
terminal gate alpha/m.  Language group -> districts: Karamojong = KOTIDO, MOROTO,
NAKAPIRIPIRIT; Lugbara = ARUA, YUMBE; Pallisa = PALLISA.

CPU only, under a minute.  Reads data/ and results/ (local, not tracked); writes
results/realworld_final/uganda_districts.json (refuses to overwrite without
--overwrite).

    python scripts/realworld_uganda_districts.py [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import realworld_clustered_nexis as R  # noqa: E402

OUT = ROOT / "results" / "realworld_final" / "uganda_districts.json"
MODEL_DIR = ROOT / "results" / "uganda" / "prithvi_l5_1024"
OUTCOMES = ["skilled_employed", "log_biz_assets"]
VARIANTS = {"published NEXIS": R.PUBLISHED_CFG, "new default": R.NEW_DEFAULT}
LANG_DISTRICTS = {"Karamojong": ["KOTIDO", "MOROTO", "NAKAPIRIPIRIT"],
                  "Lugbara": ["ARUA", "YUMBE"], "Pallisa": ["PALLISA"]}


def uganda_districts(outcome):
    """The paper's Uganda pool with W_lang_* replaced in place by W_district_*."""
    from apps.uganda.data import resolve_outcome
    d = R.uganda(outcome)
    df = pd.read_csv(ROOT / "data" / "uganda" / "UgandaDataProcessed.csv", low_memory=False)
    Zall = np.load(MODEL_DIR / "individual_features.npz")["features"]
    df = df[df[resolve_outcome(outcome)].notna() & np.isfinite(Zall[:, 0])].reset_index(drop=True)
    assert len(df) == len(d["y"]) and np.array_equal(df["Wobs"].values, d["t"])
    dist = pd.get_dummies(df["district"], prefix="district", dtype=float)
    lang = [j for j, n in enumerate(d["names"]) if n.startswith("W_lang_")]
    assert lang == list(range(lang[0], lang[-1] + 1)) and len(lang) == 7
    a, b = lang[0], lang[-1] + 1
    z = np.hstack([d["z"][:, :a], dist.values, d["z"][:, b:]])
    names = d["names"][:a] + [f"W_{c}" for c in dist.columns] + d["names"][b:]
    return dict(d, z=z, names=names)


def june_run(outcome):
    f = MODEL_DIR / f"{outcome}_districts" / "nexis_result.json"
    return json.loads(f.read_text()) if f.exists() else None


def district_table(d, fn, S_final, S_tilde, m):
    y, t, z, names = d["y"], d["t"], d["z"], d["names"]
    idx = {n: j for j, n in enumerate(names)}
    cols = [j for j, n in enumerate(names) if n.startswith("W_district_")]
    fin = [idx[n] for n in S_final]
    til = [idx[n] for n in S_tilde]
    pm = fn(y, t, z, [], cols)
    rows = {}
    for j in cols:
        pf = float(fn(y, t, z, [k for k in fin if k != j], [j])[j])
        ps = float(fn(y, t, z, [k for k in til if k != j], [j])[j])
        rows[names[j]] = dict(p_marginal=float(pm[j]), p_given_final=pf, p_given_Stilde=ps,
                              in_Stilde=names[j] in S_tilde, selected=names[j] in S_final,
                              below_alpha_over_m=bool(min(pf, ps) <= R.ALPHA / m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if OUT.exists() and not args.overwrite:
        raise SystemExit(f"{OUT} exists; pass --overwrite to replace it")
    report = dict(alpha=R.ALPHA, rho=R.RHO, test="linear, homoskedastic OLS, no clustering",
                  treatment="Wobs (grant received)", lang_districts=LANG_DISTRICTS)
    fn = R.plain_test()
    for outcome in OUTCOMES:
        d = uganda_districts(outcome)
        m = d["z"].shape[1]
        print(f"\n{'=' * 100}\nuganda / {outcome}   n={len(d['y'])}   pool={m}   alpha/m={R.ALPHA / m:.2e}")
        june = june_run(outcome)
        block = dict(n=len(d["y"]), pool=m, names=d["names"], runs={})
        if june is not None:
            meta = june["feature_meta"]
            june_names = [f"Z_{k}" for k in meta["sae_active_idx"]] + [f"W_{w}" for w in meta["w_names"]]
            assert june_names == d["names"], "pool differs from the June run"
            print("  pool: identical names and order to the June run")
        for vname, cfg in VARIANTS.items():
            print(f"  -- {vname}")
            r = R.run(d, vname, fn, cfg)
            r["districts"] = district_table(d, fn, r["selected"], r["S_tilde"], m)
            if cfg is R.PUBLISHED_CFG and june is not None:
                jsel = {s["label"]: s["pvalue"] for s in june["nexis_fwer"]["selected"]}
                ours = {c["name"]: c["p_cond_final"] for c in r["coords"] if c["selected"]}
                same = sorted(jsel) == sorted(ours)
                rel = max(abs(ours[k] / jsel[k] - 1) for k in jsel) if same else None
                r["june_check"] = dict(june_selected=jsel, same_set=same, max_rel_diff_p=rel)
                print(f"        June run (nexis_fwer): {jsel}\n        same set: {same}"
                      + (f"   max relative p difference: {rel:.1e}" if same else ""))
            block["runs"][vname] = r
            print(f"        district dummies (p marginal | p given final set | p given S~):")
            for n, row in r["districts"].items():
                flag = "selected" if row["selected"] else ("in S~" if row["in_Stilde"] else "")
                print(f"          {n:26s} {row['p_marginal']:.1e} | {row['p_given_final']:.1e} | "
                      f"{row['p_given_Stilde']:.1e}  {flag}")
        report[f"uganda/{outcome}"] = block

    print(f"\n{'=' * 100}\nSUMMARY (skilled employment, the outcome of the paper's sentence)")
    b = report["uganda/skilled_employed"]
    for vname, r in b["runs"].items():
        print(f"\n  {vname}: S~ = {r['S_tilde']}  selected = {r['selected']}")
        for c in r["coords"]:
            if c["name"] == "W_district_PALLISA":
                pf = f"{c['p_cond_final']:.2e}" if c["p_cond_final"] is not None else "-"
                print(f"    PALLISA: p | rest of final set {pf}   marginal {c['p_marginal']:.2e}   "
                      f"certification (worst subset) {c['worst_subset_p']:.2e} "
                      f"(alpha/m = {r['gate']:.2e})")
        for lang, dists in LANG_DISTRICTS.items():
            for dn in dists:
                row = r["districts"][f"W_district_{dn}"]
                print(f"    {lang:10s} {dn:14s} marginal {row['p_marginal']:.1e}  "
                      f"| final {row['p_given_final']:.1e}  "
                      f"{'selected' if row['selected'] else ('in S~' if row['in_Stilde'] else 'not in S~')}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float))
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
