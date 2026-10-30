"""Interpretation of the learned effect modifiers that discover() selected.

For every selected covariate of a learned source, the ids (at the source's level) with
the highest and the lowest values are fetched as raw items (e.g. image paths) and given
to the source's interpreter, describe(name, top, bottom) -> str (name: the covariate),
which returns a label.
"""

from __future__ import annotations

from collections.abc import Callable
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
    render: covariate -> its source's render (raw item -> image array) or None.
    """

    def __init__(self, frame: pd.DataFrame, items: dict[str, tuple[list, list]], message: str = "",
                 render: dict[str, Callable | None] | None = None):
        self.frame, self.items, self.message = frame, items, message
        self.render = dict(render or {})

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
        least active ones, titled by its label.  A source with a `render` draws
        render(item), which must be an H x W or H x W x 3 array of floats in [0, 1] or
        uint8; otherwise raw items must be images: arrays (H x W or H x W x C) or paths to
        image files readable by matplotlib.  An item that cannot be drawn is an error
        naming the covariate and its source.  Returns the matplotlib Figure, or None
        (with a message) when there is nothing to draw."""
        if not len(self):
            print(f"Interpretation: {self.message}")
            return None
        import matplotlib.pyplot as plt

        covs = list(self.frame["covariate"]) if covariates is None else list(covariates)
        k = max(max(len(self.items[c][0]), len(self.items[c][1])) for c in covs)
        fig, axes = plt.subplots(2 * len(covs), k, figsize=(size * k, 2.3 * size * len(covs)),
                                 squeeze=False)
        labels = self.labels
        source = dict(zip(self.frame["covariate"], self.frame["source"]))
        for i, c in enumerate(covs):
            for row, (name, items) in enumerate(zip(("most active", "least active"), self.items[c])):
                axs = axes[2 * i + row]
                for j, ax in enumerate(axs):
                    ax.set_xticks([]), ax.set_yticks([])
                    if j < len(items):
                        img = _drawable(items[j], self.render.get(c), c, source.get(c))
                        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
                    else:
                        ax.axis("off")
                axs[0].set_ylabel(name, fontsize=8)
            axes[2 * i, 0].set_title("\n".join(wrap(f"{c}: {labels[c]}", 30 * k // 2)),
                                     loc="left", fontsize=9)
        fig.tight_layout()
        return fig


def _drawable(item, render: Callable | None, covariate: str, source: str | None) -> np.ndarray:
    """The image show() draws for a raw item, through the source's render if any."""
    where = f"{covariate!r} (source {source!r})"
    if render is None:
        try:
            return _image(item)
        except TypeError as e:
            raise TypeError(f"show() cannot draw a raw item of {where}: {e}; give the source a "
                            "render= (raw item -> H x W x 3 array)") from None
    try:
        img = render(item)
    except Exception as e:
        raise ValueError(f"render of source {source!r} failed on a raw item of {covariate!r} "
                         f"({shorten(repr(item), 60)}): {type(e).__name__}: {e}") from e
    if not isinstance(img, np.ndarray):
        raise TypeError(f"render of source {source!r} must return a numpy array, got {type(img).__name__}")
    if not (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3)) or 0 in img.shape:
        raise ValueError(f"render of source {source!r} must return an H x W or H x W x 3 array, got "
                         f"shape {img.shape}")
    if img.dtype == np.uint8:
        return img
    if not np.issubdtype(img.dtype, np.floating):
        raise ValueError(f"render of source {source!r} must return floats in [0, 1] or uint8, got dtype "
                         f"{img.dtype}")
    if not np.isfinite(img).all() or img.min() < 0 or img.max() > 1:
        raise ValueError(f"render of source {source!r} must return floats in [0, 1], got values in "
                         f"[{np.nanmin(img):g}, {np.nanmax(img):g}]" + (" with non-finite values"
                                                                     if not np.isfinite(img).all() else ""))
    return img


def _image(item) -> np.ndarray:
    if isinstance(item, np.ndarray):
        if item.ndim == 2 or (item.ndim == 3 and item.shape[2] in (3, 4)):
            return item
        raise TypeError(f"an array of shape {item.shape} is not an image (H x W, H x W x 3 or 4)")
    if isinstance(item, (str, Path)) and Path(item).is_file():
        import matplotlib.image as mpimg
        try:
            return mpimg.imread(item)
        except Exception as e:      # e.g. a multi-band GeoTIFF matplotlib cannot read
            raise TypeError(f"matplotlib cannot read {shorten(str(item), 60)} "
                            f"({type(e).__name__}: {shorten(str(e), 80)})") from None
    raise TypeError(f"raw items must be image arrays or paths to image files, got "
                    f"{shorten(repr(item), 60)}")
