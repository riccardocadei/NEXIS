"""NEXIS: select the pre-treatment features that modify a treatment effect, from a large
pool of candidates (e.g. sparse-autoencoder atoms of images), with family-wise error control.

    import nexis
    data = nexis.Dataset(units, sources=[nexis.Source(table, time="pre"), ...],
                         design=nexis.Randomized(treatment="T", assigned_at="village"))
    data.info()
    result = nexis.discover(data, outcome="y")
    result.table(); result.interpret()

Architecture.  The public layer describes a study and runs the search on it:

    data.py        Source (one table: level, time, origin) and Dataset (the units, the
                   sources broadcast onto them, the design; missing-value policy)
    assignment.py  Design, Randomized, Discontinuity: treatment, assignment / sampling
                   / strata levels, and their checks
    api.py         discover() -> Result (modifiers, table(), interpret())
    interpret.py   Interpretation of the selected learned modifiers

The engine works on arrays and can be used directly:

    search.py      nexis(y, t, w): the search (forward Bonferroni gates, terminal filter)
                   and marginal_select, the one-test-per-column baseline
    pvalues.py     the conditional interaction tests nexis() calls (linear, GCM, PCM)
    design.py      the level-aware clustered test (LevelAwareTest, a `pvalue_fn`) and
                   the clustering and support rule (build_rule)
    checks.py      post-hoc checks of a selected set: randomization test, wild cluster
                   bootstrap, worst conditioning subset
    effects.py     HC1-robust OLS and the ATE
    sae.py         TopK sparse autoencoder (needs the `gpu` extra; not imported here)

Data flow: Dataset joins every Source onto the units and checks the design; discover()
takes the pre columns as one candidate matrix, applies build_rule (cluster level and
support of each candidate), runs nexis() with LevelAwareTest as its test, and returns
a Result indexed by covariate names.
"""
from .api import Result, discover
from .assignment import Design, Discontinuity, Randomized
from .data import Dataset, Source
from .interpret import Interpretation
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
    "Interpretation",
    "nexis",
    "marginal_select",
    "SelectionResult",
    "conditional_interaction_pvalues",
    "__version__",
]
