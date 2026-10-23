"""The output of one pipeline step, and its on-disk form.

Storing per-step deltas (instead of a full AnnData snapshot per step) keeps the
cache small: a clustering step stores one label column, an embedding stores an
n x k array, and only steps that rewrite X store a matrix.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class Delta:
    X: np.ndarray | None = None                  # replaces X (same shape unless var_mask given)
    var_mask: np.ndarray | None = None           # keep these features (applied before X)
    obs_mask: np.ndarray | None = None           # keep these objects (applied first)
    layers: dict = field(default_factory=dict)
    obs: dict = field(default_factory=dict)      # column -> array / Categorical
    var: dict = field(default_factory=dict)
    obsm: dict = field(default_factory=dict)
    obsp: dict = field(default_factory=dict)
    uns: dict = field(default_factory=dict)
    log: str = ""

    # ------------------------------------------------------------------ apply
    def apply(self, adata: ad.AnnData) -> ad.AnnData:
        if self.obs_mask is not None:
            adata = adata[np.asarray(self.obs_mask, bool)].copy()
        if self.var_mask is not None:
            adata = adata[:, np.asarray(self.var_mask, bool)].copy()
        if self.X is not None:
            adata.X = self.X
        for k, v in self.layers.items():
            adata.layers[k] = v
        for k, v in self.obs.items():
            adata.obs[k] = pd.Categorical(v) if _is_labels(v) else v
        for k, v in self.var.items():
            adata.var[k] = v
        for k, v in self.obsm.items():
            adata.obsm[k] = v
        for k, v in self.obsp.items():
            adata.obsp[k] = v
        for k, v in self.uns.items():
            adata.uns[k] = v
        return adata

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        manifest: dict = {"log": self.log, "layers": [], "obsm": [], "obsp": [], "obs": [], "var": []}
        arrays = {}
        if self.X is not None:
            arrays["X"] = self.X
        if self.var_mask is not None:
            arrays["var_mask"] = np.asarray(self.var_mask, bool)
        if self.obs_mask is not None:
            arrays["obs_mask"] = np.asarray(self.obs_mask, bool)
        for k, v in self.layers.items():
            arrays[f"layer::{k}"] = v
            manifest["layers"].append(k)
        for k, v in self.obsm.items():
            arrays[f"obsm::{k}"] = np.asarray(v)
            manifest["obsm"].append(k)
        for k, v in arrays.items():
            if sp.issparse(v):
                sp.save_npz(tmp / f"{_safe(k)}.sparse.npz", v.tocsr())
            else:
                np.save(tmp / f"{_safe(k)}.npy", np.asarray(v), allow_pickle=False)
        manifest["arrays"] = {k: _safe(k) + (".sparse.npz" if sp.issparse(v) else ".npy") for k, v in arrays.items()}
        for k, v in self.obsp.items():
            sp.save_npz(tmp / f"obsp__{_safe(k)}.npz", sp.csr_matrix(v))
            manifest["obsp"].append(k)
        if self.obs:
            pd.DataFrame({k: _col(v) for k, v in self.obs.items()}).to_parquet(tmp / "obs.parquet")
            manifest["obs"] = list(self.obs)
        if self.var:
            pd.DataFrame({k: _col(v) for k, v in self.var.items()}).to_parquet(tmp / "var.parquet")
            manifest["var"] = list(self.var)
        if self.uns:
            with open(tmp / "uns.pkl", "wb") as fh:
                pickle.dump(self.uns, fh, protocol=pickle.HIGHEST_PROTOCOL)
        (tmp / "delta.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        if path.exists():
            import shutil
            shutil.rmtree(path)
        tmp.rename(path)  # atomic publish: a half-written delta is never visible

    @classmethod
    def load(cls, path: str | Path) -> "Delta":
        path = Path(path)
        manifest = json.loads((path / "delta.json").read_text(encoding="utf-8"))
        arrays = {}
        for k, fname in manifest.get("arrays", {}).items():
            f = path / fname
            arrays[k] = sp.load_npz(f) if fname.endswith(".sparse.npz") else np.load(f, allow_pickle=False)
        d = cls(log=manifest.get("log", ""))
        d.X = arrays.get("X")
        d.var_mask = arrays.get("var_mask")
        d.obs_mask = arrays.get("obs_mask")
        d.layers = {k: arrays[f"layer::{k}"] for k in manifest["layers"]}
        d.obsm = {k: arrays[f"obsm::{k}"] for k in manifest["obsm"]}
        d.obsp = {k: sp.load_npz(path / f"obsp__{_safe(k)}.npz") for k in manifest["obsp"]}
        if manifest["obs"]:
            df = pd.read_parquet(path / "obs.parquet")
            d.obs = {k: df[k].values for k in manifest["obs"]}
        if manifest["var"]:
            df = pd.read_parquet(path / "var.parquet")
            d.var = {k: df[k].values for k in manifest["var"]}
        if (path / "uns.pkl").exists():
            with open(path / "uns.pkl", "rb") as fh:
                d.uns = pickle.load(fh)
        return d


def _safe(k: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in k)


def _col(v):
    if isinstance(v, (pd.Categorical, pd.Series)):
        return v
    return np.asarray(v)


def _is_labels(v) -> bool:
    if isinstance(v, pd.Categorical):
        return True
    a = np.asarray(v)
    return a.dtype.kind in "OUS"
