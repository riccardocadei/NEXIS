"""NEXIS: Neural EXposure Interaction Search.

Selects the pre-treatment features that modify a treatment effect (heterogeneous
treatment effects) from a large pool of candidates, with family-wise error control.

    from nexis import nexis
    res = nexis(y, t, w)           # w: (n, M) ndarray or pandas DataFrame
    res.selected, res.feature_names

Submodules, imported on demand:

    nexis.design       level-aware clustered test for multilevel data (a `pvalue_fn`)
    nexis.checks       post-hoc design checks: randomization test, wild cluster bootstrap
    nexis.estimation   HC1-robust OLS, ATE and GATE/CATE reporting
    nexis.sae          TopK sparse autoencoder (needs the `gpu` extra: torch, overcomplete)
"""
from .pvalues import conditional_interaction_pvalues
from .search import SelectionResult, marginal_select, nexis

__version__ = "0.1"

__all__ = [
    "nexis",
    "marginal_select",
    "SelectionResult",
    "conditional_interaction_pvalues",
    "__version__",
]
