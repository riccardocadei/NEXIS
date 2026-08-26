"""
Robustness of the Uganda NEXIS results to the variance estimator.

The published results (docs/uganda_experiment_brief.md §6) use homoskedastic
OLS standard errors with no clustering.  The Uganda design is hierarchical:

    individual  ⊂  group (randomisation unit)  ⊂  community (satellite site)

so residuals are plausibly correlated within group and within community, and
the candidate modifiers themselves live at different levels of that hierarchy.
This script

  1. labels every one of the 170 NEXIS candidates with the level at which it
     actually varies (verified empirically from the data, not assumed),
  2. re-runs the full NEXIS selection under several variance estimators
     (homoskedastic, HC1, CR1S clustered at group / community / component /
     district) and reports whether the selected set changes,
  3. re-estimates the final published model with nested random effects
     (community + group-within-community) as a multilevel cross-check.

Nothing is retrained: the frozen SAE artifacts in
results/uganda/{model}_{dim}/ are read as-is.

Usage
-----
    python src/apps/uganda/robustness_clustering.py \
        --embed-model prithvi_l5 --sae-dim 1024 \
        --outcomes skilled_employed,log_biz_assets
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = ROOT / "data" / "uganda"
sys.path.insert(0, str(ROOT / "src"))

from method.nexis import nexis, conditional_interaction_pvalues  # noqa: E402
from apps.uganda.data import resolve_outcome                     # noqa: E402
from apps.uganda.analyze import build_covariates                 # noqa: E402

# Cluster identifiers, coarsest last.  'component' is derived (see below).
#
# District is deliberately NOT a clustering level.  It is the randomisation *block*:
# treatment was assigned to groups within districts.  Clustering is called for by
# clustered sampling or clustered assignment (Abadie, Athey, Imbens & Wooldridge
# 2023); a stratification variable is neither, and is handled by conditioning (block
# fixed effects, block-stratified re-randomisation) rather than by clustering.  With
# G=14 its CRVE is also anti-conservative, which is what we observed: it was the only
# estimator that "rescued" features every other estimator rejected.
CLUSTER_COLS = {
    "group":     "groupid",
    "community": "geo_long_lat_key",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embed-model", default="prithvi_l5")
    p.add_argument("--sae-dim", type=int, default=1024)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--active-threshold", type=int, default=5)
    p.add_argument("--outcomes", default="skilled_employed,log_biz_assets")
    p.add_argument("--min-support", type=int, default=10,
                   help="Effective-support gate for the sandwich runs: a candidate "
                        "is admissible only if at least this many clusters are "
                        "active-and-treated AND at least this many are "
                        "active-and-control.  The T*Z_j interaction is identified "
                        "off exactly those cells; below ~10 the sandwich meat is a "
                        "sum of a handful of terms and collapses toward zero, "
                        "producing spuriously tiny SEs.")
    p.add_argument("--out-dir", default="")
    p.add_argument("--no-mixed", dest="mixed", action="store_false",
                   help="Skip the (slow) mixed-effects cross-check.")
    p.set_defaults(mixed=True)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset construction — mirrors analyze.py exactly (individual level, W as
# candidates, spectral indices on).  Verified against the stored nexis_result
# by reproducing the homoskedastic p-values.
# ──────────────────────────────────────────────────────────────────────────────
def build_dataset(args, outcome):
    csv_col = resolve_outcome(outcome)
    model_dir = ROOT / "results" / "uganda" / f"{args.embed_model}_{args.sae_dim}"

    df = pd.read_csv(DATA_DIR / "UgandaDataProcessed.csv", low_memory=False)
    if csv_col not in df.columns or df[csv_col].isna().all():
        return None
    df = df.rename(columns={"Wobs": "T", csv_col: "Y"})

    Z_all = np.load(model_dir / "individual_features.npz")["features"]
    site_full = np.load(model_dir / "site_features.npz")["site_features"]
    active_mask = (site_full > 0).sum(axis=0) >= args.active_threshold
    sae_orig_idx = np.where(active_mask)[0]
    Z_all = Z_all[:, active_mask]
    n_sae = int(active_mask.sum())

    mask = df["Y"].notna() & np.isfinite(Z_all[:, 0])
    df_sub = df[mask].reset_index(drop=True)
    Z_sub = Z_all[mask]

    W_df = build_covariates(df_sub, district_dummies=False)
    spec_path = DATA_DIR / "satellite" / "rct" / "spectral_indices.csv"
    if spec_path.exists():
        spec_df = pd.read_csv(spec_path).set_index("site_key")
        spec_cols = list(spec_df.columns)
        spec_mat = np.full((len(df_sub), len(spec_cols)), np.nan)
        for i, key in enumerate(df_sub["geo_long_lat_key"].values):
            if pd.notna(key) and int(key) in spec_df.index:
                spec_mat[i] = spec_df.loc[int(key)].values
        W_df = pd.concat(
            [W_df, pd.DataFrame(spec_mat, columns=spec_cols, index=df_sub.index)],
            axis=1)

    w_names = list(W_df.columns)
    Z_full = np.hstack([Z_sub, W_df.values.astype(float)])
    labels = ([f"Z_{int(j)}" for j in sae_orig_idx] + [f"W_{w}" for w in w_names])

    return dict(
        Y=df_sub["Y"].values.astype(float),
        T=df_sub["T"].values.astype(float),
        Z_full=Z_full, labels=labels, n_sae=n_sae, df=df_sub,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Cluster construction
# ──────────────────────────────────────────────────────────────────────────────
def component_clusters(df):
    """Connected components of the group–community bipartite graph.

    Groups are *not* strictly nested in communities (a minority of groups draw
    members from more than one site), so neither `groupid` nor
    `geo_long_lat_key` alone is a valid partition for CRVE if dependence runs
    through both.  The coarsening that respects both is the set of connected
    components — the standard fix for non-nested multiway clustering.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for g, c in zip(df["groupid"].values, df["geo_long_lat_key"].values):
        union(f"g{g}", f"c{c}")
    return np.array([find(f"g{g}") for g in df["groupid"].values], dtype=str)


def build_clusters(df):
    out = {}
    for name, col in CLUSTER_COLS.items():
        out[name] = df[col].astype(str).values
    out["component"] = component_clusters(df)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Level labelling — empirical, from the data
# ──────────────────────────────────────────────────────────────────────────────
def label_levels(data, clusters):
    """Classify each candidate by the coarsest partition it is constant within.

    Scanned coarsest-first — region (lang_group, 7) is a strict coarsening of
    district (14), which is a strict coarsening of community (327).  Scanning only
    group/community would mislabel the region-level variables as community-level,
    since a variable constant within region is trivially constant within every site
    inside it.  The `lang_*` dummies are region-level, not community-level: they are
    one-hot encodings of the very partition that defines a region.
    """
    df = data["df"]
    Z = data["Z_full"]
    order = ["region", "district", "community", "group"]
    keys = {"region": pd.Series(df["lang_group"].astype(str).values),
            "district": pd.Series(df["district"].astype(str).values),
            "community": pd.Series(clusters["community"]),
            "group": pd.Series(clusters["group"])}
    rows = []
    for j, lab in enumerate(data["labels"]):
        col = pd.Series(Z[:, j])
        tot = np.nanstd(col.values)
        scale = tot if tot > 0 else 1.0
        sds = {k: float(col.groupby(v).transform("std").fillna(0.0).max())
               for k, v in keys.items()}
        level = "individual"
        for k in order:
            if sds[k] / scale < 1e-10:
                level = k
                break
        rows.append(dict(idx=j, label=lab,
                         kind="SAE" if j < data["n_sae"] else "W",
                         level=level,
                         within_region_sd=sds["region"],
                         within_district_sd=sds["district"],
                         within_group_sd=sds["group"],
                         within_community_sd=sds["community"],
                         total_sd=float(tot)))
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Variants
# ──────────────────────────────────────────────────────────────────────────────
def support_diagnostics(data, clusters, level="community"):
    """Per-candidate effective support for the T*Z_j interaction.

    The interaction coefficient is identified by contrasting treated vs control
    units *within* the active and inactive strata of Z_j.  Under a cluster-robust
    (or heteroskedasticity-robust) sandwich the variance is estimated from the
    per-cluster score sums; if only a couple of clusters are simultaneously
    active and treated, the meat matrix is effectively a sum of one or two terms
    and shrinks toward zero, while the t-test still uses df = G-1.  The result
    is an SE biased catastrophically downward — the classic 'few treated
    clusters' failure (MacKinnon & Webb 2017; Conley & Taber 2011).

    Also returns max leverage of the 4-column design [1, T, Z_j, T*Z_j]; HC0/HC1
    are not leverage-corrected, so max h_ii near 1 is the heteroskedastic analog
    of the same pathology.
    """
    Z, T = data["Z_full"], data["T"]
    g = pd.Series(clusters[level])
    n = len(T)
    rows = []
    for j, lab in enumerate(data["labels"]):
        z = Z[:, j]
        act = z != 0
        n_at = g[act & (T == 1)].nunique()
        n_ac = g[act & (T == 0)].nunique()
        X = np.column_stack([np.ones(n), T, z, T * z])
        try:
            lev = float(np.max(np.sum((X @ np.linalg.pinv(X.T @ X)) * X, axis=1)))
        except np.linalg.LinAlgError:                        # pragma: no cover
            lev = np.nan
        rows.append(dict(idx=j, label=lab,
                         n_active_obs=int(act.sum()),
                         n_active_clusters=int(g[act].nunique()),
                         n_active_treated_clusters=int(n_at),
                         n_active_control_clusters=int(n_ac),
                         max_leverage=lev))
    return pd.DataFrame(rows)


def se_variants(clusters):
    v = [("homoskedastic (published)", dict()),
         ("HC1", dict(hc1=True))]
    for name in ["group", "community", "component"]:
        v.append((f"CR1S cluster={name}", dict(cluster=clusters[name])))
    return v


def run_variant(data, kwargs, args, adjust, cols=None):
    """Run NEXIS on the (optionally restricted) candidate pool.

    cols: array of column indices of Z_full to expose as candidates.  Selected
    indices are mapped back to the full-pool numbering before returning.
    """
    Z = data["Z_full"] if cols is None else data["Z_full"][:, cols]
    res = nexis(data["Y"], data["T"], Z,
                alpha=args.alpha, max_rounds=args.max_steps,
                adjust=adjust, verbose=False, **kwargs)
    if cols is None:
        return [(int(j), float(res.pvalues[j])) for j in res.selected]
    return [(int(cols[j]), float(res.pvalues[j])) for j in res.selected]


def gate_table(data, S, variants, m_total):
    """Marginal and conditional p for each published feature, vs its Bonferroni gate.

    The relevant question is not 'is p still small' but 'does p still clear the
    gate NEXIS actually applied' — α/|remaining| at the step where the feature
    entered.  We evaluate the leave-one-out conditional p(j | S\\{j}) against
    α/(m - |S| + 1), the gate the feature faces on the realised path.
    """
    rows = []
    for j in S:
        S_minus = [k for k in S if k != j]
        gate = 0.05 / (m_total - len(S_minus))
        for vname, kw in variants:
            pm = float(conditional_interaction_pvalues(
                y=data["Y"], t=data["T"], z=data["Z_full"],
                S=[], candidates=[j], **kw)[j])
            pc = float(conditional_interaction_pvalues(
                y=data["Y"], t=data["T"], z=data["Z_full"],
                S=S_minus, candidates=[j], **kw)[j])
            rows.append(dict(feature=data["labels"][j], estimator=vname,
                             p_marginal=pm, p_conditional=pc,
                             bonferroni_gate=gate, passes=bool(pc <= gate)))
    return pd.DataFrame(rows)


def conditional_p_at_S(data, S, kwargs):
    """p-value of each j in S given S\\{j}, under a given variance estimator."""
    out = {}
    for j in S:
        S_minus = [k for k in S if k != j]
        p = conditional_interaction_pvalues(
            y=data["Y"], t=data["T"], z=data["Z_full"],
            S=S_minus, candidates=[j], **kwargs)
        out[j] = float(p[j])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Multilevel cross-check
# ──────────────────────────────────────────────────────────────────────────────
def mixed_check(data, S, outcome):
    """Refit the final selected model with nested random effects.

    Y ~ T + Z_S + T:Z_S,  random intercepts for community and group-within-
    community.  Reports the T:Z_j Wald p-values.
    """
    import statsmodels.formula.api as smf

    df = data["df"]
    Z = data["Z_full"]
    labs = [data["labels"][j] for j in S]
    d = pd.DataFrame({"Y": data["Y"], "T": data["T"]})
    safe = []
    for j, lab in zip(S, labs):
        name = lab.replace("-", "_")
        d[name] = Z[:, j]
        safe.append(name)
    d["community"] = df["geo_long_lat_key"].astype(str).values
    d["group"] = df["groupid"].astype(str).values

    rhs = " + ".join(["T"] + safe + [f"T:{s}" for s in safe])
    try:
        m = smf.mixedlm(f"Y ~ {rhs}", d, groups=d["community"],
                        re_formula="1", vc_formula={"grp": "0 + C(group)"})
        fit = m.fit(reml=True, method="lbfgs", maxiter=500)
    except Exception as e:                                   # pragma: no cover
        return {"error": f"{type(e).__name__}: {e}"}

    out = {"converged": bool(fit.converged), "outcome": outcome, "terms": {}}
    for s in safe:
        for key in (f"T:{s}", f"{s}:T"):
            if key in fit.params.index:
                out["terms"][s] = {"coef": float(fit.params[key]),
                                   "se": float(fit.bse[key]),
                                   "pvalue": float(fit.pvalues[key])}
                break
    try:
        out["var_community"] = float(fit.cov_re.iloc[0, 0])
        out["var_group"] = float(fit.vcomp[0])
        out["var_resid"] = float(fit.scale)
    except Exception:
        pass
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "results" / "uganda" / f"{args.embed_model}_{args.sae_dim}"
        / "robustness_clustering")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"embed_model": args.embed_model, "sae_dim": args.sae_dim,
              "alpha": args.alpha, "outcomes": {}}

    for outcome in args.outcomes.split(","):
        outcome = outcome.strip()
        data = build_dataset(args, outcome)
        if data is None:
            print(f"SKIP {outcome}: outcome column missing")
            continue

        clusters = build_clusters(data["df"])
        n = len(data["Y"])
        counts = {k: int(len(np.unique(v))) for k, v in clusters.items()}
        print("=" * 78)
        print(f"OUTCOME: {outcome}   n={n}   candidates={data['Z_full'].shape[1]}")
        print("  cluster counts:", counts)

        lv = label_levels(data, clusters)
        sup = support_diagnostics(data, clusters, level="community")
        lv = lv.merge(sup.drop(columns=["label"]), on="idx")
        lv.to_csv(out_dir / f"{outcome}_candidate_levels.csv", index=False)
        lvl_counts = {f"{k[0]}/{k[1]}": int(v) for k, v in
                      lv.groupby(["kind", "level"]).size().items()}
        print("  candidate levels:", lvl_counts)

        # how much within-cluster T variation survives at each level
        tvar = {}
        for k, v in clusters.items():
            s = pd.Series(data["T"]).groupby(pd.Series(v)).nunique()
            tvar[k] = {"clusters": int(len(s)),
                       "with_T_variation": int((s > 1).sum())}
        print("  T variation within cluster:", tvar)

        # Effective-support gate for the sandwich estimators
        ok = ((lv["n_active_treated_clusters"] >= args.min_support) &
              (lv["n_active_control_clusters"] >= args.min_support))
        cols = lv.loc[ok, "idx"].values.astype(int)
        print(f"  candidates passing support gate (>={args.min_support} active-treated "
              f"AND >={args.min_support} active-control communities): "
              f"{len(cols)}/{len(lv)}")

        variants = se_variants(clusters)
        res_o = {"n": n, "cluster_counts": counts, "T_variation": tvar,
                 "levels": lvl_counts,
                 "min_support": args.min_support,
                 "n_candidates_full": int(len(lv)),
                 "n_candidates_supported": int(len(cols)),
                 "supported_candidates": [data["labels"][j] for j in cols],
                 "variants_full_pool": {}, "variants_supported_pool": {}}

        selected_sets = {}
        for pool_name, pool_cols in [("variants_full_pool", None),
                                     ("variants_supported_pool", cols)]:
            print(f"  ── {pool_name.replace('variants_','').replace('_',' ')} ──")
            for vname, kwargs in variants:
                entry = {}
                for adjust in ["FWER", "FDR"]:
                    sel = run_variant(data, kwargs, args, adjust, cols=pool_cols)
                    entry[adjust] = [{"idx": j, "label": data["labels"][j],
                                      "pvalue": p} for j, p in sel]
                    if adjust == "FWER" and pool_cols is None:
                        selected_sets[vname] = [data["labels"][j] for j, _ in sel]
                res_o[pool_name][vname] = entry
                print(f"    {vname:28s} FWER -> "
                      f"{[s['label'] for s in entry['FWER']]}")

        # Published selection, re-tested under every variance estimator
        pub_labels = selected_sets["homoskedastic (published)"]
        lab2idx = {lab: j for j, lab in enumerate(data["labels"])}
        pub = [lab2idx[l] for l in pub_labels]
        res_o["published_S"] = pub_labels
        res_o["published_S_support"] = {
            data["labels"][j]: {
                "n_active_treated_clusters":
                    int(lv.loc[lv["idx"] == j, "n_active_treated_clusters"].iloc[0]),
                "n_active_control_clusters":
                    int(lv.loc[lv["idx"] == j, "n_active_control_clusters"].iloc[0]),
                "max_leverage":
                    float(lv.loc[lv["idx"] == j, "max_leverage"].iloc[0]),
            } for j in pub}
        res_o["published_S_pvalues"] = {}
        for vname, kwargs in variants:
            pv = conditional_p_at_S(data, pub, kwargs)
            res_o["published_S_pvalues"][vname] = {
                data["labels"][j]: v for j, v in pv.items()}
        print("  published S, conditional p under each estimator:")
        for vname in res_o["published_S_pvalues"]:
            vals = res_o["published_S_pvalues"][vname]
            print(f"    {vname:28s} " +
                  "  ".join(f"{k}={v:.2e}" for k, v in vals.items()))

        gt = gate_table(data, pub, variants, data["Z_full"].shape[1])
        gt.to_csv(out_dir / f"{outcome}_gate_table.csv", index=False)
        res_o["gate_table"] = gt.to_dict("records")
        surv = gt.groupby("estimator")["passes"].sum().to_dict()
        print(f"  published features clearing their Bonferroni gate "
              f"(of {len(pub)}):", surv)

        if args.mixed and pub:
            print("  fitting nested mixed model ...")
            res_o["mixed_model"] = mixed_check(data, pub, outcome)
            mm = res_o["mixed_model"]
            if "terms" in mm:
                for k, v in mm["terms"].items():
                    print(f"    T x {k:14s} coef={v['coef']:+.4f} "
                          f"se={v['se']:.4f} p={v['pvalue']:.3e}")

        report["outcomes"][outcome] = res_o
        print()

    path = out_dir / "robustness_clustering.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved → {path}")


if __name__ == "__main__":
    main()
