"""NEXIS (Neural EXposure Interaction Search) and the marginal-testing baselines.

NEXIS (Algorithm 1 of the paper, with the practical options of its appendix):

  forward step           S <- {}; repeat: among the candidates j not in S that pass the
                         gate (FWER: p_j(S) <= alpha / |S_bar|, default; no correction:
                         p_j(S) <= alpha; FDR: Benjamini-Hochberg over S_bar), admit the
                         strongest one, j*, unless the spectral-gap gate stops the search:
                         |T(j* | S)| < rho * min_{j in S} |T_j|, where T_j is the statistic
                         of j when it was admitted.  Stop when nothing enters.
  interleaved backward   optional, after every forward round: remove j in S if
  step                   p_j(S \\ {j}) > alpha / |S| (Benjamini-Hochberg over S for FDR).
  terminal backward      once the forward step has stopped at S~, keep j in S~ only if
  step                   p_j(A) <= alpha / m for EVERY A subset of S~ \\ {j}, the empty set
                         included (largest subsets first); removals are simultaneous.

Default: linear test, FWER forward gate, rho = 0.5, no interleaved backward step, terminal
backward step, alpha = 0.05.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Sequence

import numpy as np

from .cate_tests import linear_pvalues, make_test


@dataclass
class SelectionResult:
    selected: List[int]                                       # selected coordinates of z
    forward_path: List[int] = field(default_factory=list)     # S~, before the terminal step
    metadata: Dict[str, float] = field(default_factory=dict)


def nexis(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    test: str = "linear",
    adjust: Optional[str] = "FWER",
    rho: Optional[float] = 0.5,
    interleaved_backward: bool = False,
    terminal_backward: bool = True,
    max_rounds: Optional[int] = None,
    n_splits: int = 3,
) -> SelectionResult:
    """Run NEXIS on outcome y, binary treatment t and pre-treatment representation z (n, m).

    Args:
        alpha:      significance level.
        test:       CATE-equivalence test, one of nexis.cate_tests.TESTS.
        adjust:     forward-step correction: "FWER" (Bonferroni, default), "FDR"
                    (Benjamini-Hochberg) or None.  The terminal step always uses alpha / m.
        rho:        spectral-gap gate in [0, 1]; 0 or None disables it.  When it is on,
                    the forward step admits the eligible candidate with the largest |T|,
                    otherwise the one with the smallest p-value.
        interleaved_backward: run the interleaved backward step after every round.
        terminal_backward:    run the terminal backward step.
        max_rounds: cap on the number of rounds (None: no cap).
        n_splits:   cross-fitting folds of the GCM / PCM nuisances.
    """
    pvalues = make_test(test, n_splits=n_splits)
    if rho is not None and rho == 0:
        rho = None
    adj = adjust.upper() if adjust is not None else None
    if adj not in (None, "FWER", "FDR"):
        raise ValueError("adjust must be None, 'FWER' or 'FDR'")

    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape

    def p_of(A, j) -> float:
        """p_j(A), the p-value of H0(j | A)."""
        return float(pvalues(y, t, Z, list(A), [j])[j])

    selected: List[int] = []
    t_selected: List[float] = []
    S_prev: List[int] = [-1]
    round_num = 0
    gap_stop = False
    n_removed = 0

    while selected != S_prev and not gap_stop:
        if max_rounds is not None and round_num >= max_rounds:
            break
        S_prev = list(selected)

        # ── forward step ──────────────────────────────────────────────────────
        remaining = [j for j in range(m) if j not in selected]
        if remaining:
            if rho is not None:
                pvals, tstats = pvalues(y, t, Z, selected, remaining, return_tstats=True)
            else:
                pvals, tstats = pvalues(y, t, Z, selected, remaining), None

            if adj == "FDR":
                pv_rem = np.array([pvals[j] for j in remaining])
                order = np.argsort(pv_rem)
                bh = (np.arange(1, len(remaining) + 1) / len(remaining)) * alpha
                below = pv_rem[order] <= bh
                eligible = ([remaining[order[i]]
                             for i in range(int(np.where(below)[0].max()) + 1)]
                            if below.any() else [])
            else:
                gate = alpha if adj is None else alpha / len(remaining)
                eligible = [j for j in remaining if pvals[j] <= gate]

            if eligible:
                if tstats is not None:
                    j_star = max(eligible, key=lambda j: abs(tstats[j]))
                else:
                    j_star = min(eligible, key=lambda j: pvals[j])
                # spectral-gap gate (the first candidate is governed by the p-value gate only)
                if rho is not None and t_selected:
                    t_min = min(t_selected)
                    if t_min > 0 and float(abs(tstats[j_star])) < rho * t_min:
                        gap_stop = True
                if not gap_stop:
                    selected.append(j_star)
                    if tstats is not None:
                        t_selected.append(float(abs(tstats[j_star])))
        if gap_stop:
            break

        # ── interleaved backward step (optional) ─────────────────────────────
        if interleaved_backward:
            if adj == "FDR":
                js = list(selected)
                back = np.array([p_of([s for s in selected if s != j], j) for j in js])
                order = np.argsort(back)
                bh = (np.arange(1, len(js) + 1) / len(js)) * alpha
                below = back[order] <= bh
                keep = ({js[order[i]] for i in range(int(np.where(below)[0].max()) + 1)}
                        if below.any() else set())
                for j in js:
                    if j not in keep:
                        selected.remove(j)
                        n_removed += 1
            else:
                for j in list(selected):
                    if j not in selected:
                        continue
                    p_j = p_of([s for s in selected if s != j], j)
                    gate = alpha if adj is None else alpha / len(selected)
                    if p_j > gate:
                        selected.remove(j)
                        n_removed += 1
        round_num += 1

    # ── terminal backward step ────────────────────────────────────────────────
    forward_path = list(selected)
    terminal_tests = 0
    if terminal_backward and selected:
        gate = alpha / m
        keep_final: List[int] = []
        for j in forward_path:
            others = [s for s in forward_path if s != j]
            retained = True
            for r in range(len(others), -1, -1):          # largest subsets first
                for A in combinations(others, r):
                    terminal_tests += 1
                    if p_of(A, j) > gate:                  # dropped at its first failure
                        retained = False
                        break
                if not retained:
                    break
            if retained:
                keep_final.append(j)
        selected = keep_final

    return SelectionResult(
        selected=selected,
        forward_path=forward_path,
        metadata={"m": float(m), "rounds": float(round_num),
                  "interleaved_removed": float(n_removed),
                  "terminal_tests": float(terminal_tests)},
    )


def marginal_select(y, t, z, alpha: float = 0.05,
                    adjust: Optional[str] = None) -> SelectionResult:
    """Marginal screening: test every coordinate alone, H0(j | {}), with the linear test.

    adjust: None (level alpha), "FWER" (Bonferroni, alpha / m) or "FDR" (Benjamini-Hochberg).
    """
    pvals = linear_pvalues(y, t, z, S=[])
    m = len(pvals)
    adj = adjust.upper() if adjust is not None else None
    if adj is None:
        selected = np.where(pvals <= alpha)[0].tolist()
    elif adj == "FWER":
        selected = np.where(pvals <= alpha / max(m, 1))[0].tolist()
    elif adj == "FDR":
        order = np.argsort(pvals)
        below = pvals[order] <= (np.arange(1, m + 1) / m) * alpha
        selected = order[: int(np.where(below)[0].max()) + 1].tolist() if below.any() else []
    else:
        raise ValueError("adjust must be None, 'FWER' or 'FDR'")
    return SelectionResult(selected=selected, metadata={"m": float(m)})


def iou_score(selected: Sequence[int], truth: Sequence[int]) -> float:
    S, T = set(int(x) for x in selected), set(int(x) for x in truth)
    union = S | T
    return 1.0 if not union else len(S & T) / len(union)
