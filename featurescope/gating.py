"""Hierarchical gates (FlowJo-style) defined on any pair of columns.

A gate is a plain dict, so gates serialize to ``gates.json`` and can be
referenced by pipeline steps (``subset_gate``):

    {"name": "CD3+", "parent": "Live", "x": "feat:CD3", "y": "feat:CD45",
     "type": "polygon", "vertices": [[x, y], ...]}
    {"name": "Large", "x": "obs:area", "type": "range", "range": [lo, hi]}
    {"name": "T or B", "type": "boolean", "expr": "or", "of": ["T", "B"]}
"""
from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath

from .data import get_vector


def points_in_polygon(x: np.ndarray, y: np.ndarray, vertices) -> np.ndarray:
    pts = np.column_stack([x, y])
    ok = np.isfinite(pts).all(1)
    out = np.zeros(len(x), bool)
    out[ok] = MplPath(np.asarray(vertices, float)).contains_points(pts[ok])
    return out


def region_mask(adata, region: dict) -> np.ndarray:
    """Mask for a single geometric region (no parent)."""
    t = region.get("type", "polygon")
    if t == "boolean":
        raise ValueError("boolean gates need evaluate_gate")
    x = np.asarray(get_vector(adata, region["x"]), float)
    if t == "range":
        lo, hi = region["range"]
        return (x >= lo) & (x <= hi)
    y = np.asarray(get_vector(adata, region["y"]), float)
    if t == "rect":
        (x0, x1), (y0, y1) = sorted(region["xrange"]), sorted(region["yrange"])
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)
    if t == "quadrant":
        cx, cy, q = region["center"][0], region["center"][1], region["quadrant"]  # q in ++ +- -+ --
        mx = x >= cx if q[0] == "+" else x < cx
        my = y >= cy if q[1] == "+" else y < cy
        return mx & my
    return points_in_polygon(x, y, region["vertices"])


def evaluate_gate(adata, gates: dict, name: str, _memo: dict | None = None) -> np.ndarray:
    memo = {} if _memo is None else _memo
    if name in memo:
        return memo[name]
    if name not in gates:
        raise KeyError(f"gate {name!r} not defined")
    g = gates[name]
    if g.get("type") == "boolean":
        parts = [evaluate_gate(adata, gates, n, memo) for n in g["of"]]
        expr = g.get("expr", "and")
        m = np.logical_and.reduce(parts) if expr == "and" else np.logical_or.reduce(parts)
        if expr == "not":
            m = ~parts[0]
    else:
        m = region_mask(adata, g)
    if g.get("parent"):
        m = m & evaluate_gate(adata, gates, g["parent"], memo)
    memo[name] = m
    return m


def gate_stats(adata, gates: dict) -> list[dict]:
    """Counts and % of parent for every gate (the gating tree view)."""
    memo: dict = {}
    out = []
    for name, g in gates.items():
        try:
            m = evaluate_gate(adata, gates, name, memo)
            parent_n = evaluate_gate(adata, gates, g["parent"], memo).sum() if g.get("parent") else adata.n_obs
            out.append({"name": name, "parent": g.get("parent"), "count": int(m.sum()),
                        "pct_parent": float(100 * m.sum() / max(parent_n, 1)),
                        "pct_total": float(100 * m.mean()) if adata.n_obs else 0.0,
                        "x": g.get("x"), "y": g.get("y"), "type": g.get("type"), "color": g.get("color")})
        except Exception as e:  # a gate on a column that no longer exists after a pipeline change
            out.append({"name": name, "parent": g.get("parent"), "error": str(e)})
    return out


def gates_to_obs(adata, gates: dict) -> dict:
    """Boolean obs columns 'gate:<name>' for colouring / exporting."""
    memo: dict = {}
    out = {}
    for name in gates:
        try:
            out[f"gate:{name}"] = evaluate_gate(adata, gates, name, memo)
        except Exception:
            pass
    return out
