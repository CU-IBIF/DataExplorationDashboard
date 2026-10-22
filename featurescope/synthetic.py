"""Synthetic multiplexed-imaging-like dataset for demos and tests.

Cell types with marker programs, arranged in spatial tissue domains, imaged on
several slides with multiplicative/additive batch effects and lognormal noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CELL_TYPES = {
    "T cell":      {"CD45": 3, "CD3": 3, "CD4": 1.5, "CD8": 1.5},
    "B cell":      {"CD45": 3, "CD20": 3, "HLA-DR": 2},
    "Macrophage":  {"CD45": 2.5, "CD68": 3, "CD163": 2, "HLA-DR": 1.5},
    "Tumor":       {"PanCK": 3, "Ki67": 1.5, "ECad": 2.5},
    "Fibroblast":  {"aSMA": 3, "Vimentin": 3, "CD90": 2},
    "Endothelial": {"CD31": 3, "Vimentin": 1.5},
}
MARKERS = ["CD45", "CD3", "CD4", "CD8", "CD20", "CD68", "CD163", "HLA-DR", "PanCK", "Ki67", "ECad",
           "aSMA", "Vimentin", "CD90", "CD31", "DAPI"]
# tissue domains: mixing proportions of cell types
DOMAINS = {
    "tumor core":     [0.05, 0.0, 0.10, 0.75, 0.05, 0.05],
    "immune stroma":  [0.35, 0.25, 0.20, 0.05, 0.10, 0.05],
    "fibrotic":       [0.05, 0.02, 0.08, 0.05, 0.70, 0.10],
}


def make_dataset(n: int = 20_000, n_slides: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    types = list(CELL_TYPES)
    rows = []
    per = n // n_slides
    for s in range(n_slides):
        x, y = rng.uniform(0, 2000, per), rng.uniform(0, 2000, per)
        # three domains as angular sectors around a random center
        cx, cy = rng.uniform(700, 1300, 2)
        ang = (np.arctan2(y - cy, x - cx) + np.pi) / (2 * np.pi)
        dom = np.minimum((ang * 3).astype(int), 2)
        dom_names = np.array(list(DOMAINS))[dom]
        probs = np.array(list(DOMAINS.values()))[dom]
        ct = (rng.random((per, 1)) > np.cumsum(probs, 1)).sum(1).clip(0, len(types) - 1)  # vectorized categorical draw
        base = np.full((per, len(MARKERS)), 0.3)
        for k, t in enumerate(types):
            for m, v in CELL_TYPES[t].items():
                base[ct == k, MARKERS.index(m)] = v
        base[:, MARKERS.index("DAPI")] = 2.5
        expr = np.exp(base + rng.normal(0, 0.35, base.shape))          # lognormal intensities
        gain = rng.uniform(0.6, 1.6, len(MARKERS))                      # slide batch effect
        offset = rng.uniform(0, 0.5, len(MARKERS))
        expr = expr * gain + offset
        area = rng.lognormal(4.5, 0.3, per) * np.where(np.array(types)[ct] == "Tumor", 1.6, 1.0)
        df = pd.DataFrame(expr * 100, columns=MARKERS)
        df.insert(0, "cell_id", [f"s{s}_{i}" for i in range(per)])
        df["area"] = area
        df["eccentricity"] = rng.beta(2, 5, per)
        df["x_centroid"], df["y_centroid"] = x, y
        df["slide"] = f"slide_{s + 1}"
        df["true_cell_type"] = np.array(types)[ct]
        df["true_domain"] = dom_names
        rows.append(df)
    return pd.concat(rows, ignore_index=True)
