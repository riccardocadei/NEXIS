"""Interpretation of the learned effect modifiers that discover() selected.

For every selected covariate of a learned source, the ids (at the source's level) with
the highest and the lowest values are fetched as raw items (e.g. image paths) and given
to the source's interpreter, describe(top, bottom) -> str, which returns a label.
"""

from __future__ import annotations

from pathlib import Path
from textwrap import shorten, wrap

import numpy as np
import pandas as pd


def extremes(values: pd.Series, k: int, seed: int = 0) -> tuple[list, list]:
    """The ids of the k highest and the k lowest values (missing values ignored).

    Ties are broken at random with `seed` (with sparse features many ids are tied at
    zero; taking the first ids would show the same few items for every feature)."""
    v = values.dropna()
    perm = np.random.default_rng(seed).permutation(len(v))
    v = v.iloc[perm]
    order = np.argsort(v.to_numpy(), kind="stable")
    ids = v.index[order]
    return list(ids[::-1][:k]), list(ids[:k])


class Interpretation:
    """Labels of the selected learned modifiers, with their most and least active ids.

    frame: DataFrame with one row per covariate: covariate, source, label, top (ids,
        most active first), bottom (ids, least active first).
    items: covariate -> (raw items of top, raw items of bottom).
    message: why the interpretation is empty, if it is.
    """

    def __init__(self, frame: pd.DataFrame, items: dict[str, tuple[list, list]], message: str = ""):
        self.frame, self.items, self.message = frame, items, message

    @property
    def labels(self) -> dict[str, str]:
        """covariate -> label."""
        return dict(zip(self.frame["covariate"], self.frame["label"]))

    def __len__(self) -> int:
        return len(self.frame)

    def __repr__(self) -> str:
        if not len(self):
            return f"Interpretation: {self.message}"
        lines = [f"Interpretation of {len(self)} learned modifier{'s' * (len(self) > 1)}:"]
        for r in self.frame.itertuples():
            lines.append(f"  {r.covariate} ({r.source}): {r.label}")
            lines.append(f"    top ids    {shorten(', '.join(map(str, r.top)), 70, placeholder=' ...')}")
            lines.append(f"    bottom ids {shorten(', '.join(map(str, r.bottom)), 70, placeholder=' ...')}")
        return "\n".join(lines)

    def show(self, covariates: list[str] | None = None, size: float = 1.6):
        """Draw, per covariate, a row of its most active raw items above a row of its
        least active ones, titled by its label.  Raw items must be images: arrays
        (H x W or H x W x C) or paths to image files readable by matplotlib.  Returns the
        matplotlib Figure, or None (with a message) when there is nothing to draw."""
        if not len(self):
            print(f"Interpretation: {self.message}")
            return None
        import matplotlib.pyplot as plt

        covs = list(self.frame["covariate"]) if covariates is None else list(covariates)
        k = max(max(len(self.items[c][0]), len(self.items[c][1])) for c in covs)
        fig, axes = plt.subplots(2 * len(covs), k, figsize=(size * k, 2.3 * size * len(covs)),
                                 squeeze=False)
        labels = self.labels
        for i, c in enumerate(covs):
            for row, (name, items) in enumerate(zip(("most active", "least active"), self.items[c])):
                axs = axes[2 * i + row]
                for j, ax in enumerate(axs):
                    ax.set_xticks([]), ax.set_yticks([])
                    if j < len(items):
                        img = _image(items[j])
                        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                    else:
                        ax.axis("off")
                axs[0].set_ylabel(name, fontsize=8)
            axes[2 * i, 0].set_title("\n".join(wrap(f"{c}: {labels[c]}", 30 * k // 2)),
                                     loc="left", fontsize=9)
        fig.tight_layout()
        return fig


def _image(item) -> np.ndarray:
    if isinstance(item, np.ndarray) and item.ndim in (2, 3):
        return item
    if isinstance(item, (str, Path)) and Path(item).is_file():
        import matplotlib.image as mpimg
        return mpimg.imread(item)
    raise TypeError(f"show() draws images: raw items must be image arrays or paths to image "
                    f"files, got {shorten(repr(item), 60)}")
