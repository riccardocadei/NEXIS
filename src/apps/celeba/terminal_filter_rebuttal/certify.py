"""Post-hoc certification of a frozen NEXIS candidate set.

Two one-shot procedures applied to S_cand AFTER NEXIS returns.  Neither is allowed
to re-run NEXIS or to re-certify on a reduced set.

Option 1 — Global pathwise Bonferroni
    Pre-specify K = max certifiable |S|.  The hypothesis universe is every
    (coordinate, conditioning set) pair reachable by a path of length <= K-1:
        N_K = m * sum_{q=0}^{K-1} C(m-1, q)
    If |S_cand| > K the run is UNCERTIFIABLE (K may not be raised after the fact).
    Otherwise keep j iff  p(j | S_cand \ {j})  <=  alpha / N_K.

Option 2 — Subset-robust certification
    On the recall event S* subseteq S_cand, so only subsets of S_cand matter.
    p_j^robust = max over A subseteq S_cand\{j} of p(j | A)      [includes A = {} ]
    Keep j iff p_j^robust <= alpha / m.
    Cost: |S_cand| * 2^(|S_cand|-1) tests.
"""
from __future__ import annotations

from itertools import combinations
from math import lgamma, log10
from typing import Dict, List, Sequence

import numpy as np
from scipy.special import logsumexp

from method.nexis import conditional_interaction_pvalues

NEG_INF = -np.inf


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return NEG_INF
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


def log10_N_K(m: int, K: int) -> float:
    """log10 of  m * sum_{q=0}^{K-1} C(m-1, q)  (natural-log logsumexp internally)."""
    terms = [_log_comb(m - 1, q) for q in range(K)]
    return (np.log(m) + logsumexp(terms)) / np.log(10)


def _p_one(y, t, Z, j: int, A: Sequence[int]) -> float:
    """p-value for H0(j | A), computed on the full feature matrix Z."""
    pv = conditional_interaction_pvalues(y=y, t=t, z=Z, S=list(A), candidates=[j])
    return float(pv[int(j)])


def _passes(p: float, log10_thr: float) -> bool:
    """p <= 10**log10_thr, evaluated in log space so tiny thresholds don't underflow."""
    if p <= 0.0:
        return True          # underflowed to exactly 0 -> passes any threshold
    return log10(p) <= log10_thr


def certify_global(y, t, Z, S_cand: Sequence[int], alpha: float, K: int) -> Dict:
    """Option 1. Returns certified set, threshold, per-coordinate p-values, status."""
    m = Z.shape[1]
    S_cand = [int(j) for j in S_cand]
    log10_thr = log10(alpha) - log10_N_K(m, K)

    if len(S_cand) > K:
        return {"status": "uncertifiable", "K": K, "log10_thr": log10_thr,
                "candidate": S_cand, "certified": [], "discarded": [],
                "pvalues": {}, "n_hypotheses_log10": log10_N_K(m, K)}

    pvals = {j: _p_one(y, t, Z, j, [s for s in S_cand if s != j]) for j in S_cand}
    certified = [j for j in S_cand if _passes(pvals[j], log10_thr)]
    return {"status": "ok", "K": K, "log10_thr": log10_thr,
            "candidate": S_cand, "certified": certified,
            "discarded": [j for j in S_cand if j not in certified],
            "pvalues": pvals, "n_hypotheses_log10": log10_N_K(m, K)}


def certify_subset_robust(y, t, Z, S_cand: Sequence[int], alpha: float) -> Dict:
    """Option 2. Returns certified set, robust p-values and each argmax subset."""
    m = Z.shape[1]
    S_cand = [int(j) for j in S_cand]
    log10_thr = log10(alpha) - log10(m)

    robust: Dict[int, float] = {}
    worst_subset: Dict[int, List[int]] = {}
    n_tests = 0
    for j in S_cand:
        others = [s for s in S_cand if s != j]
        best_p, best_A = -1.0, []
        for r in range(len(others) + 1):
            for A in combinations(others, r):
                p = _p_one(y, t, Z, j, A)
                n_tests += 1
                if p > best_p:
                    best_p, best_A = p, list(A)
        robust[j] = best_p
        worst_subset[j] = best_A

    certified = [j for j in S_cand if _passes(robust[j], log10_thr)]
    return {"status": "ok", "log10_thr": log10_thr, "candidate": S_cand,
            "certified": certified,
            "discarded": [j for j in S_cand if j not in certified],
            "robust_pvalues": robust, "worst_subset": worst_subset,
            "n_tests": n_tests}
