
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Dict
import numpy as np
from scipy import stats


@dataclass
class SelectionResult:
    selected: List[int]
    pvalues: np.ndarray
    method: str
    alpha: float
    metadata: Dict[str, float]
    selected_groups: List[str] = field(default_factory=list)  # group per selection


def _residualize_against(D: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Residualize columns of V against the column space of D using QR.
    D: (n, q), V: (n, k) or (n,)
    """
    V2 = np.asarray(V, dtype=float)
    vec = (V2.ndim == 1)
    if vec:
        V2 = V2[:, None]

    # If D is empty, return V
    if D.size == 0 or D.shape[1] == 0:
        return V if not vec else V2[:, 0]

    # Reduced QR; works well when q is small
    Q, _ = np.linalg.qr(D, mode="reduced")
    R = V2 - Q @ (Q.T @ V2)
    return R[:, 0] if vec else R


def conditional_interaction_pvalues(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    S: Optional[Sequence[int]] = None,
    candidates: Optional[Sequence[int]] = None,
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    interaction_only: Optional[set] = None,
) -> np.ndarray:
    """
    Vectorized p-values for H0(j|S) over j in candidates.
    Working model:
      Y = beta0 + betaT T + beta_S' Z_S + beta_j Z_j + gamma_S'(T*Z_S) + gamma_j (T*Z_j)
          + beta_W' W + gamma_W'(T*W) + e
    Tests gamma_j = 0 for each j.

    controls: optional (n, q) matrix W always included in the nuisance design as
    [W, T*W].  Removes W-based heterogeneity (main effect + interaction) from
    residuals without consuming Bonferroni budget.

    main_controls: optional (n, q) matrix W included in the nuisance design as [W]
    only (main effects, NOT T*W).  Use this when W's interaction with T should be
    tested as a candidate rather than partialled out unconditionally.

    interaction_only: optional set of candidate column indices for which only
    T*Z_j is tested (1-regressor, 1 dof) rather than the standard [Z_j, T*Z_j]
    (2-regressor, 2 dof).  Use for W candidates when W main effects are already
    in main_controls: Z_j = W_k would residualize to ~0 against D (which already
    contains W_k), making the 2-regressor system degenerate.  The 1-regressor
    test correctly recovers H0: gamma_k = 0 with W_k partialled out via D.

    Uses Frisch-Waugh-Lovell residualization against D=[1, T, W, T*W, Z_S, T*Z_S],
    then a 2-regressor OLS per candidate on [Z_j, T*Z_j] (after residualization),
    or a 1-regressor OLS on [T*Z_j] for interaction_only candidates.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    t = np.asarray(t, dtype=float).reshape(-1)
    Z = np.asarray(z, dtype=float)
    n, m = Z.shape
    if y.shape[0] != n or t.shape[0] != n:
        raise ValueError("Shape mismatch among y, t, z")

    S = [] if S is None else sorted(set(int(k) for k in S))
    if candidates is None:
        cand = np.array([j for j in range(m) if j not in S], dtype=int)
    else:
        cand = np.array([int(j) for j in candidates if int(j) not in S], dtype=int)

    pvals = np.ones(m, dtype=float)
    if cand.size == 0:
        return pvals

    # Split candidates into regular (2-reg) and interaction-only (1-reg)
    io_set = interaction_only if interaction_only is not None else set()
    reg_cand = np.array([j for j in cand if j not in io_set], dtype=int)
    io_cand  = np.array([j for j in cand if j in io_set],     dtype=int)

    # Common nuisance design D = [1, T, W, T*W, main_W, Z_S, T*Z_S]
    D_cols = [np.ones(n), t]
    if controls is not None:
        W = np.asarray(controls, dtype=float)
        for col in range(W.shape[1]):
            D_cols.append(W[:, col])
        for col in range(W.shape[1]):
            D_cols.append(t * W[:, col])
    if main_controls is not None:
        Wm = np.asarray(main_controls, dtype=float)
        for col in range(Wm.shape[1]):
            D_cols.append(Wm[:, col])
    for k in S:
        D_cols.append(Z[:, k])
    for k in S:
        D_cols.append(t * Z[:, k])
    D = np.column_stack(D_cols) if len(D_cols) > 0 else np.empty((n, 0), dtype=float)

    y_tilde = _residualize_against(D, y)  # (n,)
    yy = np.sum(y_tilde ** 2)

    # ── Standard 2-regressor test for regular candidates ──────────────────────
    if reg_cand.size > 0:
        Z_c = Z[:, reg_cand]                      # (n, K)
        TZ_c = (t[:, None] * Z_c)                 # (n, K)
        Z_tilde = _residualize_against(D, Z_c)
        X_tilde = _residualize_against(D, TZ_c)

        zz = np.sum(Z_tilde * Z_tilde, axis=0)
        xx = np.sum(X_tilde * X_tilde, axis=0)
        zx = np.sum(Z_tilde * X_tilde, axis=0)
        zy = np.sum(Z_tilde * y_tilde[:, None], axis=0)
        xy = np.sum(X_tilde * y_tilde[:, None], axis=0)

        # 2x2 OLS algebra for coeff on X_tilde (interaction term)
        det = zz * xx - zx * zx
        valid = det > 1e-12

        # Degrees of freedom = n - (#columns in full model)
        # full model columns = dim(D) + 2  [Z_j and T*Z_j]
        p_full = D.shape[1] + 2
        dof = n - p_full

        if dof > 0:
            beta_x = np.zeros_like(det)
            beta_z = np.zeros_like(det)
            beta_x[valid] = (zz[valid] * xy[valid] - zx[valid] * zy[valid]) / det[valid]
            beta_z[valid] = (xx[valid] * zy[valid] - zx[valid] * xy[valid]) / det[valid]

            # RSS = y'y - beta'X'y ; here X=[z,x]
            rss = np.full_like(det, np.nan, dtype=float)
            rss[valid] = yy - beta_z[valid] * zy[valid] - beta_x[valid] * xy[valid]
            rss = np.maximum(rss, 0.0)

            sigma2 = np.full_like(det, np.nan, dtype=float)
            sigma2[valid] = rss[valid] / dof

            # Var(beta_x) = sigma^2 * (G^{-1})_{22} = sigma^2 * zz/det
            var_bx = np.full_like(det, np.nan, dtype=float)
            var_bx[valid] = sigma2[valid] * (zz[valid] / det[valid])

            ok = valid & np.isfinite(var_bx) & (var_bx > 0)
            tstat = np.zeros_like(det, dtype=float)
            tstat[ok] = beta_x[ok] / np.sqrt(var_bx[ok])

            p = np.ones_like(det, dtype=float)
            p[ok] = 2.0 * stats.t.sf(np.abs(tstat[ok]), df=dof)
            p = np.clip(np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
            pvals[reg_cand] = p

    # ── 1-regressor test for interaction_only candidates ──────────────────────
    # Tests H0: gamma_j = 0 using only T*Z_j, with Z_j already in D (main_controls).
    # D already contains W_k main effects, so T*W_k is the only free regressor.
    if io_cand.size > 0:
        TX_c = t[:, None] * Z[:, io_cand]          # (n, K_io)
        TX_tilde = _residualize_against(D, TX_c)   # (n, K_io)

        # dof = n - dim(D) - 1  (only 1 free regressor per candidate)
        dof1 = n - (D.shape[1] + 1)
        if dof1 > 0:
            tx_ty = np.sum(TX_tilde * y_tilde[:, None], axis=0)  # (K_io,)
            tx_tx = np.sum(TX_tilde * TX_tilde,         axis=0)  # (K_io,)
            valid_io = tx_tx > 1e-12

            beta_io = np.zeros_like(tx_tx)
            beta_io[valid_io] = tx_ty[valid_io] / tx_tx[valid_io]

            rss_io = np.full_like(tx_tx, yy)
            rss_io[valid_io] -= beta_io[valid_io] * tx_ty[valid_io]
            rss_io = np.maximum(rss_io, 0.0)

            sigma2_io = rss_io / dof1
            var_io = np.full_like(tx_tx, np.nan)
            var_io[valid_io] = sigma2_io[valid_io] / tx_tx[valid_io]

            ok_io = valid_io & np.isfinite(var_io) & (var_io > 0)
            tstat_io = np.zeros_like(tx_tx)
            tstat_io[ok_io] = beta_io[ok_io] / np.sqrt(var_io[ok_io])

            p_io = np.ones_like(tx_tx)
            p_io[ok_io] = 2.0 * stats.t.sf(np.abs(tstat_io[ok_io]), df=dof1)
            p_io = np.clip(np.nan_to_num(p_io, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0)
            pvals[io_cand] = p_io

    return pvals


def interaction_test_pvalue(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    j: int,
    S: Optional[Sequence[int]] = None,
) -> float:
    pvals = conditional_interaction_pvalues(y=y, t=t, z=z, S=S, candidates=[j])
    return float(pvals[int(j)])


def marginal_interaction_pvalues(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    interaction_only: Optional[set] = None,
) -> np.ndarray:
    return conditional_interaction_pvalues(y=y, t=t, z=z, S=[],
                                           interaction_only=interaction_only)


def marginal_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    adjust: str = "none",  # "none" or "bonferroni"
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    groups: Optional[Dict[str, List[int]]] = None,
    interaction_only: Optional[set] = None,
) -> SelectionResult:
    """Marginal interaction test with optional multiple-testing adjustment.

    groups: if provided and adjust='bonferroni', each group gets its own
            Bonferroni budget (α / group_size), matching nems_select_grouped.
            Without groups, a single α/M correction is applied over all M
            candidates (standard Bonferroni).
    """
    pvals = conditional_interaction_pvalues(y=y, t=t, z=z, S=[], controls=controls,
                                            main_controls=main_controls,
                                            interaction_only=interaction_only)
    m = len(pvals)

    if adjust.lower() in {"none", "raw", "unadjusted"}:
        selected = np.where(pvals <= alpha)[0].tolist()
        method = "marginal_raw"
        metadata: Dict[str, float] = {"threshold": float(alpha), "m": float(m)}
    elif adjust.lower() in {"bonf", "bonferroni"}:
        if groups is not None:
            # Per-group Bonferroni: each group corrects for its own size only.
            mask = np.zeros(m, dtype=bool)
            for gname, gidxs in groups.items():
                thr_g = alpha / max(len(gidxs), 1)
                for j in gidxs:
                    if pvals[j] <= thr_g:
                        mask[j] = True
            selected = np.where(mask)[0].tolist()
            method = "marginal_bonferroni_grouped"
            metadata = {"m": float(m), **{
                f"thr_{g}": alpha / max(len(idxs), 1)
                for g, idxs in groups.items()
            }}
        else:
            thr = alpha / max(m, 1)
            selected = np.where(pvals <= thr)[0].tolist()
            method = "marginal_bonferroni"
            metadata = {"threshold": float(thr), "m": float(m)}
    else:
        raise ValueError("adjust must be 'none' or 'bonferroni'")

    return SelectionResult(
        selected=selected,
        pvalues=pvals,
        method=method,
        alpha=alpha,
        metadata=metadata,
    )


def nems_select(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    alpha: float = 0.05,
    max_steps: Optional[int] = None,
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    interaction_only: Optional[set] = None,
    verbose: bool = False,
) -> SelectionResult:
    Z = np.asarray(z, dtype=float)
    m = Z.shape[1]
    selected: List[int] = []
    last_pvals = np.ones(m, dtype=float)

    # Store the p-value at the step each feature is selected (before it moves into S
    # and gets excluded from subsequent computations, which would reset it to 1.0).
    selected_pvals = np.ones(m, dtype=float)
    step = 0
    while True:
        if max_steps is not None and step >= max_steps:
            break

        remaining = [j for j in range(m) if j not in selected]
        if not remaining:
            break

        pvals = conditional_interaction_pvalues(y=y, t=t, z=Z, S=selected,
                                                candidates=remaining, controls=controls,
                                                main_controls=main_controls,
                                                interaction_only=interaction_only)
        last_pvals = pvals.copy()

        rem_p = pvals[remaining]
        j_star = remaining[int(np.argmin(rem_p))]
        p_star = pvals[j_star]
        gate = alpha / len(remaining)  # Bonferroni gate over remaining coordinates
        n_pass = int(np.sum(pvals[remaining] <= gate))

        if verbose:
            print(f"    step {step+1:2d} | remaining={len(remaining):5d} "
                  f"gate={gate:.2e} | passing={n_pass:4d} "
                  f"| best=j{j_star} p={p_star:.2e}", flush=True)

        if p_star <= gate:
            selected_pvals[j_star] = p_star   # record before j_star enters S
            selected.append(j_star)
            step += 1
        else:
            if verbose:
                print(f"    → stopped: best p={p_star:.2e} > gate={gate:.2e}")
            break

    # Merge: selected features use their selection-step p-value; others use last step.
    out_pvals = last_pvals.copy()
    for j in selected:
        out_pvals[j] = selected_pvals[j]

    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method="nems",
        alpha=alpha,
        metadata={"m": float(m), "steps": float(len(selected))},
    )


def nems_select_grouped(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    groups: Dict[str, List[int]],
    alpha: float = 0.05,
    max_steps: Optional[int] = None,
    controls: Optional[np.ndarray] = None,
    main_controls: Optional[np.ndarray] = None,
    interaction_only: Optional[set] = None,
    priority_groups: Optional[List[str]] = None,
    verbose: bool = False,
) -> SelectionResult:
    """Stratified NEMS: each group gets its own Bonferroni budget.

    At every forward step all candidates are scored jointly (same residuals),
    but the gate for candidate j is  α / |remaining in j's group|  rather than
    α / |all remaining|.  The candidate with the lowest p-value that clears its
    group gate is selected.  This controls FWER within each group at level α
    without forcing the small W pool to compete against 3072 SAE features.

    groups: dict mapping group name → list of column indices in z.
            Every column index should appear in exactly one group.

    priority_groups: optional list of group names that get first pick.  If any
            candidate in a priority group clears its gate, the best such
            candidate is selected — even if a non-priority group has a lower
            p-value.  Falls back to the normal (lowest-p-across-all-groups)
            rule only when no priority-group candidate passes its gate.
    """
    Z = np.asarray(z, dtype=float)
    m = Z.shape[1]

    # Build reverse lookup: column index → group name
    col_to_group: Dict[int, str] = {}
    for gname, idxs in groups.items():
        for idx in idxs:
            col_to_group[idx] = gname

    priority_set: set = set(priority_groups) if priority_groups else set()

    selected: List[int] = []
    selected_groups_list: List[str] = []
    last_pvals = np.ones(m, dtype=float)
    selected_pvals = np.ones(m, dtype=float)
    step = 0

    while True:
        if max_steps is not None and step >= max_steps:
            break

        remaining = [j for j in range(m) if j not in selected]
        if not remaining:
            break

        pvals = conditional_interaction_pvalues(
            y=y, t=t, z=Z, S=selected, candidates=remaining,
            controls=controls, main_controls=main_controls,
            interaction_only=interaction_only,
        )
        last_pvals = pvals.copy()

        # For each group find its best remaining candidate and check its gate
        priority_best_j: Optional[int] = None
        priority_best_p: float = 1.0
        fallback_best_j: Optional[int] = None
        fallback_best_p: float = 1.0
        group_stats: List[str] = []
        for gname, gidxs in groups.items():
            rem_g = [j for j in gidxs if j not in selected]
            if not rem_g:
                group_stats.append(f"{gname}:exhausted")
                continue
            gate = alpha / len(rem_g)
            j_g = min(rem_g, key=lambda j: pvals[j])
            p_g = pvals[j_g]
            n_pass_g = int(sum(1 for j in rem_g if pvals[j] <= gate))
            group_stats.append(f"{gname}:rem={len(rem_g)},pass={n_pass_g},best=p{p_g:.2e}")
            if p_g <= gate:
                if gname in priority_set and p_g < priority_best_p:
                    priority_best_p = p_g
                    priority_best_j = j_g
                elif p_g < fallback_best_p:
                    fallback_best_p = p_g
                    fallback_best_j = j_g

        if verbose:
            print(f"    step {step+1:2d} | S={selected} | " + "  ".join(group_stats),
                  flush=True)

        # Priority groups take precedence; fall back to best overall if none pass
        if priority_best_j is not None:
            best_j, best_p = priority_best_j, priority_best_p
        elif fallback_best_j is not None:
            best_j, best_p = fallback_best_j, fallback_best_p
        else:
            if verbose:
                print(f"    → stopped: no group cleared its gate")
            break

        if verbose:
            print(f"      → selected j={best_j} (group={col_to_group[best_j]}) "
                  f"p={best_p:.2e}", flush=True)
        selected_pvals[best_j] = best_p
        selected.append(best_j)
        selected_groups_list.append(col_to_group[best_j])
        step += 1

    out_pvals = last_pvals.copy()
    for j in selected:
        out_pvals[j] = selected_pvals[j]

    group_sizes = {gname: len(idxs) for gname, idxs in groups.items()}
    return SelectionResult(
        selected=selected,
        pvalues=out_pvals,
        method="nems_grouped",
        alpha=alpha,
        metadata={"m": float(m), "steps": float(len(selected)), **{
            f"m_{g}": float(s) for g, s in group_sizes.items()
        }},
        selected_groups=selected_groups_list,
    )


def iou_score(selected: Sequence[int], truth: Sequence[int]) -> float:
    S = set(int(x) for x in selected)
    T = set(int(x) for x in truth)
    union = S | T
    if len(union) == 0:
        return 1.0
    return len(S & T) / len(union)


def evaluate_methods_on_dataset(
    y: np.ndarray,
    t: np.ndarray,
    z: np.ndarray,
    truth: Sequence[int],
    alpha: float = 0.05,
    nems_max_steps: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    out = {}
    truth_set = set(int(x) for x in truth)
    n_truth = len(truth_set)

    def _metrics(selected: Sequence[int]) -> Dict[str, float]:
        selected_set = set(int(x) for x in selected)
        tp = float(len(selected_set & truth_set))
        fp = float(len(selected_set - truth_set))
        n_selected = float(len(selected_set))
        recall = tp / n_truth if n_truth > 0 else 1.0
        if n_selected > 0:
            precision = tp / n_selected
        else:
            precision = 1.0 if n_truth == 0 else 0.0
        return {
            "iou": iou_score(selected_set, truth_set),
            "n_selected": n_selected,
            "tp": tp,
            "fp": fp,
            "recall": float(recall),
            "precision": float(precision),
        }

    res_nems = nems_select(y=y, t=t, z=z, alpha=alpha, max_steps=nems_max_steps)
    out["NEMS"] = _metrics(res_nems.selected)

    res_raw = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="none")
    out["Marginal"] = _metrics(res_raw.selected)

    res_bonf = marginal_select(y=y, t=t, z=z, alpha=alpha, adjust="bonferroni")
    out["Marginal (Bonferroni)"] = _metrics(res_bonf.selected)

    return out
