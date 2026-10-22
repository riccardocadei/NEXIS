"""NEXIS: Neural EXposure Interaction Search.

Selects the pre-treatment features that modify a treatment effect (heterogeneous
treatment effects) from a large pool of candidates, with family-wise error control.

    import nexis
    data = nexis.Dataset(units, sources=[nexis.Source(table, time="pre"), ...],
                         design=nexis.Randomized(treatment="T", assigned_at="group_id"))
    data.info()
    result = nexis.discover(data, outcome="y")
    result.table(); result.interpret()

The search itself, on a candidate matrix:

    from nexis import nexis
    res = nexis(y, t, w)           # w: (n, M) ndarray or pandas DataFrame
    res.selected, res.feature_names

Modules:

    nexis.data         Source and Dataset
    nexis.assignment   the designs: Randomized, Discontinuity
    nexis.api          discover() and its Result

    nexis.design       level-aware clustered test for multilevel data (a `pvalue_fn`)
    nexis.checks       post-hoc design checks: randomization test, wild cluster bootstrap
    nexis.effects      HC1-robust OLS and the ATE
    nexis.sae          TopK sparse autoencoder (needs the `gpu` extra: torch, overcomplete)
"""
from .api import Result, discover
from .assignment import Design, Discontinuity, Randomized
from .data import Dataset, Source
from .pvalues import conditional_interaction_pvalues
from .search import SelectionResult, marginal_select, nexis

__version__ = "0.1"

__all__ = [
    "Source",
    "Dataset",
    "Design",
    "Randomized",
    "Discontinuity",
    "discover",
    "Result",
    "nexis",
    "marginal_select",
    "SelectionResult",
    "conditional_interaction_pvalues",
    "__version__",
]
