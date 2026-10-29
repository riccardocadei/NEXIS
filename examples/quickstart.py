"""NEXIS quickstart on synthetic data: households in villages, treatment randomized by
village within districts, candidates from a household survey and from sparse features
learned on village images.  Run: python examples/quickstart.py
"""
import numpy as np
import pandas as pd

import nexis

rng = np.random.default_rng(0)
n_villages, per_village = 60, 10
village = np.repeat(np.arange(n_villages), per_village)
households = pd.Index(np.arange(n_villages * per_village), name="household_id")
villages = pd.Index(np.arange(n_villages), name="village_id")

# Units: one row per household, with the ids of the coarser levels and the treatment.
district = np.arange(n_villages) // 10
treated = np.zeros(n_villages)
for d in range(6):                                     # half of each district's villages
    treated[rng.choice(np.arange(10 * d, 10 * d + 10), 5, replace=False)] = 1
units = pd.DataFrame({"village_id": village, "district_id": district[village],
                      "grant": treated[village]}, index=households)

# Sources: one table per level and time, indexed by the ids of its level.
survey = pd.DataFrame({"head_female": rng.integers(0, 2, len(households)),
                       "hh_size": rng.poisson(5, len(households))}, index=households)
atoms = pd.DataFrame(rng.exponential(size=(n_villages, 20)) * (rng.random((n_villages, 20)) < 0.5),
                     columns=[f"atom_{k}" for k in range(20)], index=villages)
# the raw image behind each village: brighter where atom_3 is active
tiles = {v: 0.2 * rng.random((8, 8)) + 0.8 * atoms.loc[v, "atom_3"] / atoms["atom_3"].max()
         for v in villages}
rain = pd.DataFrame({"rain_2016": rng.normal(size=6)}, index=pd.Index(range(6), name="district_id"))
effect = 1.0 + 1.5 * atoms["atom_3"].to_numpy()[village]
consumption = (effect * units["grant"] + 0.3 * survey["hh_size"]
               + rng.normal(size=n_villages)[village] + rng.normal(size=len(households)))
endline = pd.DataFrame({"consumption": consumption}, index=households)


class MeanBrightness:
    """A stand-in interpreter; a real one would show the images to a vision-language model."""

    def describe(self, top, bottom):
        return f"brightness {np.mean(top):.2f} (top) vs {np.mean(bottom):.2f} (bottom)"


# Each source declares its level (the ids its rows are indexed by), its time relative to
# treatment and the support of its columns; modality, access and reference are optional
# descriptions shown in the tables.
data = nexis.Dataset(
    units=units,
    sources=[nexis.Source(survey, level="household_id", time="pre", name="survey",
                          support={"head_female": "binary", "hh_size": "count"},
                          modality="survey", access="restricted"),
             nexis.Source(atoms, level="village_id", time="pre", support="sparse",
                          origin="learned", raw=tiles, interpreter=MeanBrightness(),
                          name="satellite", modality="image", access="public"),
             nexis.Source(rain, level="district_id", time="during", support="continuous",
                          name="rainfall"),
             nexis.Source(endline, level="household_id", time="post", support="continuous",
                          name="endline")],
    design=nexis.Randomized(treatment="grant", assigned_at="village_id",
                            stratified_by="district_id"),
)
data.info()

result = nexis.discover(data, outcome="consumption")   # warns: rain_2016 is excluded
print(result)
print(result.table().head(6).to_string())
interpretation = result.interpret(k=4)
print(interpretation)
# interpretation.show()    # draws the most and least active tiles of each modifier
