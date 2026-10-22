"""The NEXIS search: forward selection with Bonferroni gates, the optional backward
sweep and the terminal filter, and the marginal baseline (marginal_select).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import lgamma, log10
from typing import Callable, List, Optional, Sequence, Dict, Tuple, Union
import numpy as np

from .pvalues import (conditional_interaction_pvalues, conditional_interaction_pvalues_gcm,
                      conditional_interaction_pvalues_pcm)


@dataclass
class SelectionResult:
    selected: List[int]            # indices into the feature space nexis ran on
    pvalues: np.ndarray            # one entry per feature in that space
    method: str
    alpha: float
    metadata: Dict[str, float]
    feature_names: List[str] = field(default_factory=list)  # one name per column of w


def _candidate_matrix(w, cluster=None):
    """Unpack the candidate matrix `w` of nexis() / marginal_select().

    Returns (W, names, cluster): W the (n, M) float ndarray, one name per column and
    the cluster labels.  Duck-typed so this module stays independent of any
    application code:

      - `w` with a `.columns` attribute (a pandas DataFrame): names are the column
        labels verbatim; when `cluster` is None it is read from w.attrs['cluster']
        (absent key or None value: no clustering).  Any other attrs are ignored.
      - a plain ndarray: names are the column positions as strings, "0" … "M-1".

    A 1-D `w` is treated as a single column.
    """
    names = list(w.columns) if hasattr(w, "columns") else None
    if cluster is None:
        cluster = (getattr(w, "attrs", None) or {}).get("cluster")
    W = np.asarray(w, dtype=float)
    if W.ndim == 1:
        W = W[:, None]
    if names is None:
        names = [str(j) for j in range(W.shape[1])]
    return W, names, cluster


# ── Pathwise Bonferroni backward gate ─────────────────────────────────────────
#
# The backward sweep tests H0(j | S\{j}) for j in S.  Correcting only over the
# |S| coordinates on the realised path ignores that S itself was chosen by the
# data.  The family of backward hypotheses reachable on ANY data-dependent path
# partitions by selected-set size s = |S|: at depth s there are
#
#     N_s = m * C(m-1, s-1) = s * C(m, s)
#
# distinct (coordinate, conditioning-set) pairs.  Spending alpha across depths
# with deterministic weights w_s >= 0, sum_s w_s <= 1, and Bonferroni within each
# depth gives the gate
#
#     g_s = alpha * w_s / N_s
#
# so the total Type-I budget over every reachable backward hypothesis is <= alpha.
# Default w_s = 1/(s(s+1)) sums to 1 over all positive integers, so no maximum
# search depth has to be pre-specified.

def default_alpha_spending(s: int) -> float:
    """w_s = 1 / (s (s+1));  sum_{s>=1} w_s = 1."""
    return 1.0 / (s * (s + 1))


AlphaSpending = Union[Callable[[int], float], Sequence[float], None]


def _log_comb(n: int, k: int) -> float:
    """Natural log of C(n, k)."""
    if k < 0 or k > n:
        return -np.inf
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def log10_n_backward_hypotheses(m: int, s: int) -> float:
    """log10 N_s = log10( m * C(m-1, s-1) )."""
    if s < 1 or m < 1:
        return -np.inf
    return (np.log(m) + _log_comb(m - 1, s - 1)) / np.log(10)


def resolve_alpha_spending(spending: AlphaSpending,
                           max_depth: Optional[int] = None) -> Callable[[int], float]:
    """Validate and normalise an alpha-spending specification.

    spending may be None (use the default 1/(s(s+1))), a callable s -> w_s, or a
    sequence [w_1, w_2, ...].  Weights must be non-negative and their total mass
    over the supported depths must not exceed 1 (checked exactly for sequences,
    and over 1..max_depth for callables when max_depth is given).
    """
    if spending is None:
        return default_alpha_spending

    if callable(spending):
        fn = spending
        if max_depth is not None:
            ws = [float(fn(s)) for s in range(1, int(max_depth) + 1)]
            if any(w < 0 for w in ws):
                raise ValueError("alpha_spending weights must be non-negative")
            if sum(ws) > 1.0 + 1e-9:
                raise ValueError(
                    f"alpha_spending mass over depths 1..{max_depth} is "
                    f"{sum(ws):.6f} > 1")

        def _checked(s: int) -> float:
            w = float(fn(s))
            if w < 0:
                raise ValueError(f"alpha_spending weight w_{s}={w} is negative")
            return w

        return _checked

    ws = [float(w) for w in spending]
    if any(w < 0 for w in ws):
        raise ValueError("alpha_spending weights must be non-negative")
    total = sum(ws)
    if total > 1.0 + 1e-9:
        raise ValueError(f"alpha_spending weights sum to {total:.6f} > 1")

    def _from_seq(s: int) -> float:
        return ws[s - 1] if 1 <= s <= len(ws) else 0.0

    return _from_seq


def pathwise_backward_gate(alpha: float, m: int, s: int,
                           weight_fn: Callable[[int], float]) -> Tuple[float, float, float]:
    """Return (g_s, log10 g_s, log10 N_s) for the pathwise backward gate at depth s.

    g_s underflows to 0.0 in float64 for very large s; log10 g_s is exact, so
    comparisons are made in log space (see _passes_log10).
    """
    w_s = weight_fn(s)
    log10_ns = log10_n_backward_hypotheses(m, s)
    if w_s <= 0:
        return 0.0, -np.inf, log10_ns
    log10_g = log10(alpha) + log10(w_s) - log10_ns
    g = 10.0 ** log10_g if log10_g > -300 else 0.0
    return g, log10_g, log10_ns


def _passes_log10(p: float, log10_thr: float) -> bool:
    """p <= 10**log10_thr, evaluated in log space so tiny gates don't underflow."""
    if p <= 0.0:
        return True
    if not np.isfinite(log10_thr):
        return False
    return log10(p) <= log10_thr


def marginal_select(
    y: np.ndarray,
    t: np.ndarray,
    w,
    alpha: float = 0.05,
    adjust: Optional[str] = None,  # None | "FWER" | "FDR"
    groups: Optional[Dict[str, List[int]]] = None,
    cluster: Optional[np.ndarray] = None,
) -> SelectionResult:
    """Marginal interaction test with optional multiple-testing adjustment.

    adjust=None  : raw threshold at level alpha (no correction).
    adjust="FWER": Bonferroni correction (α/M).  groups splits the budget
                   per group instead of globally.
    adjust="FDR" : Benjamini-Hochberg step-up procedure at level alpha.

    w: the (n, M) candidate matrix, an ndarray or a pandas DataFrame, read exactly
      as in nexis() (column names, and w.attrs['cluster'] when `cluster` is None).
    """
    W, names, cluster = _candidate_matrix(w, cluster)
    pvals = conditional_interaction_pvalues(y=y, t=t, z=W, S=[], cluster=cluster)
    m = len(pvals)
    _adj = adjust.upper() if adjust is not None else None

    if _adj is None:
        selected = np.where(pvals <= alpha)[0].tolist()
        method = "marginal_raw"
        metadata: Dict[str, float] = {"threshold": float(alpha), "m": float(m)}
    elif _adj == "FWER":
        if groups is not None:
            # Per-group Bonferroni: each group corrects for its own size only.
            mask = np.zeros(m, dtype=bool)
            for gname, gidxs in groups.items():
                thr_g = alpha / max(len(gidxs), 1)
                for j in gidxs:
                    if pvals[j] <= thr_g:
                        mask[j] = True
            selected = np.where(mask)[0].tolist()
            method = "marginal_fwer_grouped"
            metadata = {"m": float(m), **{
                f"thr_{g}": alpha / max(len(idxs), 1)
                for g, idxs in groups.items()
            }}
        else:
            thr = alpha / max(m, 1)
            selected = np.where(pvals <= thr)[0].tolist()
            method = "marginal_fwer"
            metadata = {"threshold": float(thr), "m": float(m)}
    elif _adj == "FDR":
        order = np.argsort(pvals)
        thresholds = (np.arange(1, m + 1) / m) * alpha
        below = pvals[order] <= thresholds
        if below.any():
            kstar = int(np.where(below)[0].max())
            selected = order[:kstar + 1].tolist()
        else:
            selected = []
        method = "marginal_fdr"
        metadata = {"alpha": float(alpha), "m": float(m)}
    else:
        raise ValueError("adjust must be None, 'FWER', or 'FDR'")

    return SelectionResult(
        selected=selected,
        pvalues=pvals,
        method=method,
        alpha=alpha,
        metadata=metadata,
        feature_names=names,
    )


# ── NEXIS ──────────────────────────────────────────────────────────────────────

def nexis(
    y: np.ndarray,
    t: np.ndarray,
    w,
    alpha: float = 0.05,
    max_rounds: Optional[int] = 20,
    rho: Optional[float] = 0.5,
    backward: bool = True,
    adjust: Optional[str] = "FWER",  # None | "FWER" (Bonferroni) | "FDR" (BH)
    test: str = "linear",            # "linear" | "quadratic" | "GCM" | "PCM" variants
    nuisance: str = "poly2",         # gcm/pcm only: "poly2" | "lgbm" | "rf"
    n_splits: int = 5,             # gcm/pcm only
    n_estimators: int = 100,       # gcm/pcm only
    max_depth: Optional[int] = None,  # gcm/pcm only
    pcm_projection: str = "poly",    # pcm only: "poly" | "lgbm"
    pcm_order: int = 2,              # pcm only: polynomial-projection degree
    pcm_screen_top: int = 32,        # pcm only: candidates refitted with the ML projection
    pcm_combine: str = "bonferroni", # pcm only: "bonferroni" (2·min) | "single" | "crossfit" (invalid)
    cluster: Optional[np.ndarray] = None,  # CR1S cluster-robust SEs (linear test)
    hc1: bool = False,                     # HC1 robust SEs (linear test, when cluster is None)
    backward_gate: str = "standard",       # "standard" (alpha/s) | "pathwise" (g_s)
    alpha_spending: AlphaSpending = None,  # pathwise only; default w_s = 1/(s(s+1))
    pvalue_fn=None,                        # custom conditional test (see below)
    terminal_filter: bool = False,         # subset-robust terminal filter at alpha/m
    terminal_max_size: Optional[int] = None,  # refuse to enumerate beyond this |S~|
    verbose: bool = False,
) -> SelectionResult:
    """Forward(-backward) selection (NEXIS — Neural Exposure Interaction Search).

    Each round:
      1. Forward: among candidates passing Gate 1, pick the best.
         Gate 1 depends on adjust:
           None    : p ≤ α
           "FWER"  : p ≤ α/|remaining|  (Bonferroni, default)
           "FDR"   : BH applied to all remaining p-values at level α
         Spectral gap (ρ): stop if the chosen candidate's
         |t| < rho * min(|t| of already-selected features).
      2. Backward (skipped when backward=False): remove j ∈ S if it no longer
         passes the gate given S\\{j}.
         None/"FWER" remove sequentially; "FDR" batches all backward p-values
         with the current S and applies BH before removing.
    Repeats until S is unchanged (fixed point).

    backward=False runs a pure greedy forward pass — useful for ablation.

    w: the (n, M) matrix of ALL pre-treatment candidates — learned coordinates (e.g.
      SAE codes) and hand-crafted covariates alike — as an ndarray or a pandas
      DataFrame.  Every column competes in one search, and m = M is the pool that
      every Bonferroni gate (forward α/|remaining|, pathwise backward, terminal α/m)
      counts; there is no preliminary phase that screens a subset of columns first.
      A DataFrame carries its own metadata, read by duck typing (no pandas import):
        - SelectionResult.feature_names are its column labels, verbatim;
        - w.attrs['cluster'] supplies the cluster labels when `cluster` is not
          passed (an explicit `cluster=` wins).  Other attrs are ignored.
      For a plain ndarray, feature_names are the column positions as strings,
      "0", "1", …, "M-1".  selected and pvalues always index the columns of w.

    test:
      "linear"  — parametric interaction t-test; fast, assumes linear effects.
                  Supports controls / main_controls / interaction_only.
      "gcm"     — GCM-hybrid test (nonparametric φ̂, linear Z^j residualisation).
                  Speed: ~1.2× (poly2), ~3× (lgbm), ~27× (rf) vs linear.
                  controls / main_controls / interaction_only are ignored.
      "PCM: quadratic" / "PCM: lgbm"
                — modified Projected Covariance Measure (Lundborg, Kim, Shah &
                  Samworth, 2024): a one-sided conditional-MEAN-independence test
                  built on a half-sample-learned projection of the conditional-mean
                  contrast.  Unlike the GCM it stays consistent against alternatives
                  whose conditional covariance with Z^j vanishes (U-shapes, symmetric
                  thresholds), at the cost of splitting the sample.  Tuned via
                  pcm_projection / pcm_order / pcm_screen_top / pcm_combine.

    backward_gate:
      "standard" — the published rule: remove j if p(j|S\{j}) > α/|S| (or > α when
                   adjust=None).  Corrects only over coordinates on the realised path.
      "pathwise" — Pathwise Bonferroni: remove j if p(j|S\{j}) > g_s, where
                   g_s = α·w_s / (m·C(m-1, s-1)), s = |S| and m is the number of
                   candidate coordinates entering NEXIS.  Corrects simultaneously over
                   every backward hypothesis reachable on ANY data-dependent path, so
                   the total Type-I budget over that family is ≤ α.  Only the backward
                   threshold changes; the forward gate is untouched and no separate
                   certification stage is added.

    alpha_spending (pathwise only):
      Deterministic depth weights w_s ≥ 0 with Σ_s w_s ≤ 1, fixed before running.
      None (default) uses w_s = 1/(s(s+1)), which sums to 1 over all positive integers
      and so needs no pre-specified maximum depth.  May also be a callable s → w_s or
      a sequence [w_1, w_2, …].  Validated for non-negativity and total mass ≤ 1.

    pvalue_fn:
      Optional drop-in replacement for the conditional interaction test.  Called as
      pvalue_fn(y=, t=, z=, S=, candidates=, return_tstats=) and must return the same
      shapes as conditional_interaction_pvalues: a length-m vector of p-values (ones
      off `candidates`), or (pvalues, tstats) when return_tstats=True.  `z` is the
      candidate matrix w as an (n, M) float ndarray.  Overrides
      `test`, `cluster` and `hc1`.  NEXIS only ever consumes p-values and t-statistics
      from the test, so any valid test plugs in here — e.g. a level-aware clustered
      test on a candidate pool that spans several levels of nesting
      (LevelAwareTest in nexis.multilevel).

    rho (ρ):
      Relative stopping threshold in (0, 1].  At each forward step the new
      candidate is admitted only if |t_new| ≥ ρ × min_{j∈S}|t_j|.
      Equivalently, ρ = 1/K where K is the maximum plausible ratio between
      the strongest and weakest true direct-modifier CATE contrasts.
      Recommended range: 0.2 (effects may vary 5×) to 0.5 (effects within 2×).

    terminal_filter:
      Subset-robust terminal filter (fix for post-selection bias).
      Once the forward-backward search stops at S~, keep j ∈ S~ only if
      p(j | A) ≤ α/m for EVERY A ⊆ S~ \\ {j} (the empty set included), where m is the
      number of candidate coordinates entering NEXIS.  Removals are simultaneous.
      On the recall event S* ⊆ S~ a spurious j must reject the fixed true null
      H0(j | S*), so precision no longer relies on test validity at a data-dependent
      conditioning set.  Cost: at most |S~|·2^(|S~|-1) tests; a coordinate is dropped
      at its first failing subset, largest subsets first.  terminal_max_size raises
      ValueError instead of enumerating when |S~| exceeds it.
      Off by default, so the published selections are unchanged.
    """
    # Normalise test aliases and set nuisance accordingly.
    # "quadratic" → gcm + poly2 nuisance
    # "GCM"       → gcm + lgbm nuisance
    # "linear"    → linear (nuisance unused)
    # The canonical lowercase keys "gcm" and "pcm" are pass-through: they keep the
    # caller's `nuisance` / `pcm_projection` rather than re-deriving them.  Only the
    # user-facing aliases set those.
    _test_key = test.lower().strip()
    if _test_key in {"gcm: quadratic", "quadratic"}:
        test, nuisance = "gcm", "poly2"
    elif _test_key in {"gcm: lgbm", "lgbm"}:
        test, nuisance = "gcm", "lgbm"
    elif _test_key == "gcm":
        test = "gcm"
    elif _test_key == "pcm":
        test = "pcm"
    elif _test_key in {"pcm: quadratic", "pcm: poly"}:
        test, nuisance, pcm_projection = "pcm", "poly2", "poly"
    elif _test_key == "pcm: lgbm":
        # LightGBM *projection*, but poly2 nuisances: the PCM half-split leaves n/2
        # for Ê[φ|Z^S] and Ê[f̂|Z^S], and fully nonparametric nuisances at that size
        # are not calibrated (measured size 0.080 at α=0.05 on the check design vs
        # 0.049 with poly2, at identical power).  Pass nuisance="lgbm" explicitly
        # to override.
        test, nuisance, pcm_projection = "pcm", "poly2", "lgbm"
    elif _test_key != "linear":
        raise ValueError("test must be 'linear', 'GCM: quadratic', 'GCM: lgbm', "
                         "'PCM: quadratic', or 'PCM: lgbm'")

    # rho=0 is treated as rho=None (gate 2 disabled).
    if rho is not None and rho == 0:
        rho = None

    # Backward gate selection.  "standard" keeps the published alpha/s rule;
    # "pathwise" corrects over every backward hypothesis reachable on any path.
    _bwd = str(backward_gate).lower().strip()
    if _bwd not in {"standard", "pathwise"}:
        raise ValueError("backward_gate must be 'standard' or 'pathwise'")
    _weight_fn = (resolve_alpha_spending(alpha_spending, max_depth=max_rounds)
                  if _bwd == "pathwise" else None)
    backward_log: List[Dict[str, float]] = []

    y_arr = np.asarray(y, dtype=float).reshape(-1)
    t_arr = np.asarray(t, dtype=float).reshape(-1)
    Z, feature_names, cluster = _candidate_matrix(w, cluster)
    total = Z.shape[1]  # m: every column of w is in the one candidate pool

    gcm_kwargs: dict = (dict(nuisance=nuisance, n_splits=n_splits,
                             n_estimators=n_estimators, max_depth=max_depth)
                        if test in ("gcm", "pcm") else {})
    pcm_kwargs: dict = (dict(order=pcm_order, projection=pcm_projection,
                             screen_top=pcm_screen_top, combine=pcm_combine)
                        if test == "pcm" else {})

    def _pvalues(S_cur, candidates, return_tstats=False):
        if pvalue_fn is not None:
            return pvalue_fn(y=y_arr, t=t_arr, z=Z, S=S_cur,
                             candidates=candidates,
                             return_tstats=return_tstats)
        if test == "pcm":
            return conditional_interaction_pvalues_pcm(
                y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
                return_tstats=return_tstats, **gcm_kwargs, **pcm_kwargs,
            )
        if test == "gcm":
            return conditional_interaction_pvalues_gcm(
                y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
                return_tstats=return_tstats, **gcm_kwargs,
            )
        return conditional_interaction_pvalues(
            y=y_arr, t=t_arr, z=Z, S=S_cur, candidates=candidates,
            return_tstats=return_tstats, cluster=cluster, hc1=hc1,
        )

    selected: List[int] = []
    last_pvals = np.ones(total, dtype=float)
    selected_pvals = np.ones(total, dtype=float)
    t_selected: List[float] = []

    need_tstats = rho is not None

    S_prev: List[int] = [-1]  # sentinel
    round_num = 0
    _gate2_stop = False
    n_rejections = 0

    while selected != S_prev and not _gate2_stop:
        if max_rounds is not None and round_num >= max_rounds:
            break
        S_prev = list(selected)

        # ── Forward step ──────────────────────────────────────────────────────
        remaining = [j for j in range(total) if j not in selected]
        if remaining:
            if need_tstats:
                pvals, tstats = _pvalues(selected, remaining, return_tstats=True)
            else:
                pvals = _pvalues(selected, remaining)
                tstats = None
            last_pvals = pvals.copy()

            # Gate 1: significance filter
            _adj = adjust.upper() if adjust is not None else None
            if _adj == "FDR":
                pv_rem = np.array([pvals[j] for j in remaining])
                order_rem = np.argsort(pv_rem)
                m_rem = len(remaining)
                bh_thr = (np.arange(1, m_rem + 1) / m_rem) * alpha
                below_bh = pv_rem[order_rem] <= bh_thr
                if below_bh.any():
                    kstar_rem = int(np.where(below_bh)[0].max())
                    eligible = [remaining[order_rem[i]] for i in range(kstar_rem + 1)]
                else:
                    eligible = []
                gate_fwd = float("nan")  # no single threshold for verbose
            else:
                gate_fwd = alpha if _adj is None else alpha / len(remaining)
                eligible = [j for j in remaining if pvals[j] <= gate_fwd]

            if eligible:
                if tstats is not None:
                    j_star = max(eligible, key=lambda j: abs(tstats[j]))
                else:
                    j_star = min(eligible, key=lambda j: pvals[j])
                p_star = float(pvals[j_star])

                if verbose:
                    t_str = (f" |t|={abs(tstats[j_star]):.2f}" if tstats is not None else "")
                    gate_str = "FDR" if _adj == "FDR" else f"{gate_fwd:.2e}"
                    print(f"  round {round_num+1:2d} fwd | remaining={len(remaining):5d} "
                          f"gate={gate_str} eligible={len(eligible)} "
                          f"| best=j{j_star} p={p_star:.2e}{t_str}", flush=True)

                # Gate 2: relative stopping (rho)
                if rho is not None and tstats is not None and len(t_selected) > 0:
                    t_new = float(abs(tstats[j_star]))
                    t_min_found = min(t_selected)
                    if t_min_found > 0 and t_new < rho * t_min_found:
                        if verbose:
                            print(f"    → auto-stopped (Gate 2): "
                                  f"|t_new|={t_new:.2f} < f={rho} × "
                                  f"t_min={t_min_found:.2f}", flush=True)
                        _gate2_stop = True

                if not _gate2_stop:
                    selected_pvals[j_star] = p_star
                    selected.append(j_star)
                    if tstats is not None:
                        t_selected.append(float(abs(tstats[j_star])))
                    if verbose:
                        print(f"    → added j{j_star}  S={selected}", flush=True)
            elif verbose:
                print(f"  round {round_num+1:2d} fwd | no eligible candidate "
                      f"(gate={gate_fwd:.2e})", flush=True)

        if _gate2_stop:
            break

        # ── Backward step ─────────────────────────────────────────────────────
        if not backward:
            round_num += 1
            continue

        if _adj == "FDR":
            # Batch: compute all backward p-values with the current S, then BH.
            js = list(selected)
            back_pv = [float(_pvalues([s for s in selected if s != j], [j])[j]) for j in js]
            pv_arr = np.array(back_pv)
            order_back = np.argsort(pv_arr)
            m_sel = len(js)
            bh_thr = (np.arange(1, m_sel + 1) / m_sel) * alpha
            below = pv_arr[order_back] <= bh_thr
            keep = (
                {js[order_back[i]] for i in range(int(np.where(below)[0].max()) + 1)}
                if below.any() else set()
            )
            for idx, j in enumerate(js):
                if j not in keep:
                    selected.remove(j)
                    n_rejections += 1
                    if verbose:
                        print(f"  round {round_num+1:2d} bwd | removed j{j} "
                              f"(BH) p={back_pv[idx]:.2e}  S={selected}", flush=True)
        else:
            for j in list(selected):
                if not selected:
                    break
                if j not in selected:
                    continue
                S_minus_j = [s for s in selected if s != j]
                pvals_back = _pvalues(S_minus_j, [j])
                p_j = float(pvals_back[j])
                s_cur = len(selected)

                if _bwd == "pathwise":
                    # g_s = alpha * w_s / (m * C(m-1, s-1)); m = candidates entering NEXIS
                    gate_bwd, log10_gate, log10_ns = pathwise_backward_gate(
                        alpha, total, s_cur, _weight_fn)
                    depth_budget = alpha * _weight_fn(s_cur)
                    retained = _passes_log10(p_j, log10_gate)
                else:
                    gate_bwd = alpha if _adj is None else alpha / s_cur
                    log10_gate = log10(gate_bwd) if gate_bwd > 0 else -np.inf
                    log10_ns = log10(s_cur) if s_cur > 0 else -np.inf
                    depth_budget = alpha
                    retained = p_j <= gate_bwd

                backward_log.append({
                    "round": float(round_num + 1), "j": float(j), "s": float(s_cur),
                    "log10_N_s": float(log10_ns), "depth_budget": float(depth_budget),
                    "gate": float(gate_bwd), "log10_gate": float(log10_gate),
                    "p": p_j, "retained": bool(retained),
                })

                if verbose:
                    extra = (f" N_s=1e{log10_ns:.2f} budget={depth_budget:.2e}"
                             if _bwd == "pathwise" else "")
                    print(f"  round {round_num+1:2d} bwd | j={j} s={s_cur} p={p_j:.2e} "
                          f"gate=1e{log10_gate:.2f}{extra}", flush=True)

                if not retained:
                    selected.remove(j)
                    n_rejections += 1
                    if verbose:
                        print(f"    → removed j{j}  S={selected}", flush=True)

        round_num += 1

    # ── Terminal filter ───────────────────────────────────────────────────────
    terminal_log: List[Dict[str, object]] = []
    terminal_candidates = list(selected)
    terminal_tests = 0
    gate_terminal = alpha / total
    if terminal_filter and selected:
        if terminal_max_size is not None and len(selected) > terminal_max_size:
            raise ValueError(f"terminal filter: |S~|={len(selected)} exceeds "
                             f"terminal_max_size={terminal_max_size}")
        keep: List[int] = []
        for j in terminal_candidates:
            others = [s for s in terminal_candidates if s != j]
            worst_p, worst_A = 0.0, []
            for r in range(len(others), -1, -1):
                for A in combinations(others, r):
                    p = float(_pvalues(list(A), [j])[j])
                    terminal_tests += 1
                    if p > worst_p:
                        worst_p, worst_A = p, list(A)
                    if p > gate_terminal:
                        break
                if worst_p > gate_terminal:
                    break
            retained = worst_p <= gate_terminal
            terminal_log.append({"j": int(j), "worst_p": worst_p,
                                 "worst_A": worst_A, "retained": retained})
            if retained:
                keep.append(j)
            if verbose:
                print(f"  terminal | j={j} worst p={worst_p:.2e} at A={worst_A} "
                      f"gate={gate_terminal:.2e} → {'kept' if retained else 'removed'}",
                      flush=True)
        selected = keep

    # Recompute final conditional p-values: p(j | S \ {j}) for every selected j.
    for j in list(selected):
        S_minus_j = [s for s in selected if s != j]
        selected_pvals[j] = float(_pvalues(S_minus_j, [j])[j])

    out_pvals = last_pvals.copy()
    for j in selected:
        out_pvals[j] = selected_pvals[j]

    if test == "gcm":
        test_label = "gcm_quadratic" if nuisance == "poly2" else "gcm_lgbm"
    elif test == "pcm":
        test_label = f"pcm_{pcm_projection}"
    else:
        test_label = test
    method_str = f"nexis_{test_label}"
    if not backward:
        method_str += "_fwd"
    if _bwd == "pathwise":
        method_str += "_pwbwd"
    if rho is not None:
        method_str += f"_sg{rho}"
    if terminal_filter:
        method_str += "_tf"
    if _adj is None:
        method_str += "_noadj"
    elif _adj == "FDR":
        method_str += "_fdr"

    meta: Dict[str, object] = {
        "m": float(total),
        "steps": float(len(selected)),
        "rejections": float(n_rejections),
        "rounds": float(round_num),
        "test": test,
        "backward": backward,
        "backward_gate": _bwd,
        "backward_tests": float(len(backward_log)),
        "backward_removed": float(sum(1 for r in backward_log if not r["retained"])),
        "backward_log": backward_log,
        "terminal_filter": terminal_filter,
    }
    if terminal_filter:
        meta.update({"terminal_candidates": terminal_candidates,
                     "terminal_gate": float(gate_terminal),
                     "terminal_tests": float(terminal_tests),
                     "terminal_removed": float(len(terminal_candidates) - len(selected)),
                     "terminal_log": terminal_log})
    if test in ("gcm", "pcm"):
        meta.update({"nuisance": nuisance, "n_splits": float(n_splits),
                     "n_estimators": float(n_estimators)})
    if test == "pcm":
        meta.update({"pcm_projection": pcm_projection, "pcm_order": float(pcm_order),
                     "pcm_combine": pcm_combine,
                     "pcm_screen_top": float(pcm_screen_top)})
    if rho is not None:
        meta["rho"] = float(rho)

    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method=method_str,
        alpha=alpha,
        metadata=meta,
        feature_names=feature_names,
    )
