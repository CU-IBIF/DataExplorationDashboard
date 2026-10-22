"""Import of object-by-feature tables and helpers to address data columns.

A *column spec* is a short string that names any 1-D vector in the dataset so
that plots, gates and pipeline steps can refer to it uniformly:

    feat:<name>          current (processed) value of feature <name> in X
    raw:<name>           value of feature <name> as imported (layers["raw"])
    obs:<column>         per-object metadata / result column (clusters, gates, ...)
    obsm:<key>:<dim>     one dimension of an embedding (X_pca, X_umap, spatial, ...)
"""
from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

# Column names commonly used for object centroids by segmentation tools
# (CellProfiler, QuPath, HALO, MCMICRO, Xenium, CosMx, MERSCOPE, ...).
_SPATIAL_PAIRS = [
    ("x", "y"),
    ("x_centroid", "y_centroid"),
    ("centroid_x", "centroid_y"),
    ("centroidx", "centroidy"),
    ("location_center_x", "location_center_y"),
    ("areashape_center_x", "areashape_center_y"),
    ("centroid x µm", "centroid y µm"),
    ("centroid x um", "centroid y um"),
    ("x_position", "y_position"),
    ("center_x", "center_y"),
    ("xmin", "ymin"),
    ("cellcentroidx", "cellcentroidy"),
]
_BATCH_NAMES = ("batch", "sample", "slide", "plate", "run", "donor", "image", "imagenumber", "fov", "roi")
# Morphology / QC measurements default to metadata so they don't drive clustering
_MORPH = re.compile(r"(^|_)(area|perimeter|eccentricity|solidity|extent|circularity|compactness|"
                    r"major_?axis(_length)?|minor_?axis(_length)?|orientation|euler|form_?factor|"
                    r"aspect_?ratio|roundness|convex_?area|nucleus_?area|cell_?area|size)($|_)", re.I)
_ID_NAMES = ("id", "cell_id", "object_id", "objectnumber", "cellid", "label", "barcode")


def read_any(path: str | Path, **kw) -> pd.DataFrame | ad.AnnData:
    """Read a table (CSV/TSV/Parquet/Feather/Excel) or an AnnData file."""
    path = Path(path)
    suffix = "".join(path.suffixes[-2:]).lower() if path.suffix == ".gz" else path.suffix.lower()
    if suffix in (".h5ad",):
        return ad.read_h5ad(path)
    if suffix in (".zarr",) or (path.is_dir() and (path / ".zattrs").exists()):
        return ad.read_zarr(path)
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".feather", ".arrow"):
        return pd.read_feather(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix in (".tsv", ".txt", ".tsv.gz", ".txt.gz"):
        return pd.read_csv(path, sep="\t", **kw)
    if suffix in (".csv", ".csv.gz"):
        return pd.read_csv(path, **kw)
    return pd.read_csv(path, sep=None, engine="python", **kw)  # sniff the delimiter


def detect_roles(df: pd.DataFrame) -> dict:
    """Guess which columns are features, metadata, spatial coordinates, batch and id."""
    lower = {c.lower().strip(): c for c in df.columns}
    spatial = None
    for xs, ys in _SPATIAL_PAIRS:
        if xs in lower and ys in lower:
            spatial = [lower[xs], lower[ys]]
            break
    batch = next((lower[n] for n in _BATCH_NAMES if n in lower), None)
    id_col = next((lower[n] for n in _ID_NAMES if n in lower and df[lower[n]].is_unique), None)
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    skip = set(spatial or []) | {batch, id_col}
    features, metadata = [], []
    for c in df.columns:
        if c in skip:
            continue
        if c in numeric and _MORPH.search(str(c)):
            metadata.append(c)
        elif c in numeric:
            # integer columns with few levels are usually labels (e.g. ImageNumber)
            s = df[c]
            if pd.api.types.is_integer_dtype(s) and s.nunique() <= min(50, max(2, len(s) // 100)) and len(s) > 200:
                metadata.append(c)
            else:
                features.append(c)
        else:
            metadata.append(c)
    if batch is not None:
        metadata.insert(0, batch)
    return {"features": features, "metadata": metadata, "spatial": spatial, "batch": batch, "id": id_col}


def import_table(
    data: pd.DataFrame | str | Path,
    features: list[str] | None = None,
    metadata: list[str] | None = None,
    spatial: list[str] | None = None,
    id_col: str | None = None,
    batch: str | None = None,
    keep_raw: bool = True,
    dtype=np.float32,
) -> ad.AnnData:
    """Convert a rows=objects / columns=features table into AnnData.

    Unspecified roles are auto-detected (see :func:`detect_roles`).
    """
    if not isinstance(data, pd.DataFrame):
        loaded = read_any(data)
        if isinstance(loaded, ad.AnnData):
            return prepare_anndata(loaded, keep_raw=keep_raw)
        data = loaded
    df = data
    roles = detect_roles(df)
    features = features if features is not None else roles["features"]
    spatial = spatial if spatial is not None else roles["spatial"]
    id_col = id_col if id_col is not None else roles["id"]
    batch = batch if batch is not None else roles["batch"]
    if metadata is None:
        used = set(features) | set(spatial or []) | {id_col}
        metadata = [c for c in df.columns if c not in used]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"feature columns not found: {missing[:5]}")
    if not features:
        raise ValueError("no numeric feature columns found")

    X = df[features].to_numpy(dtype=dtype, na_value=np.nan)
    obs = df[metadata].copy() if metadata else pd.DataFrame(index=df.index)
    for c in obs.columns:
        if obs[c].dtype == object or pd.api.types.is_string_dtype(obs[c]):
            obs[c] = obs[c].astype("category")
    names = df[id_col].astype(str).to_numpy() if id_col else np.arange(len(df)).astype(str)
    obs.index = pd.Index(names, name="object_id")
    adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=pd.Index([str(f) for f in features], name="feature")))
    if spatial:
        adata.obsm["spatial"] = df[spatial].to_numpy(dtype=np.float64)
    if batch and batch in adata.obs:
        adata.uns["batch_key"] = batch
        if not isinstance(adata.obs[batch].dtype, pd.CategoricalDtype):
            adata.obs[batch] = adata.obs[batch].astype(str).astype("category")
    return prepare_anndata(adata, keep_raw=keep_raw)


def prepare_anndata(adata: ad.AnnData, keep_raw: bool = True) -> ad.AnnData:
    """Normalize conventions: unique names, dense float32 X, raw layer copy."""
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    if keep_raw and "raw" not in adata.layers:
        adata.layers["raw"] = adata.X.copy()
    adata.uns.setdefault("featurescope", {"version": 1})
    return adata


# --------------------------------------------------------------------------- column specs


def list_columns(adata: ad.AnnData) -> dict:
    """All addressable vectors, grouped for the UI."""
    obsm = {}
    for k, v in adata.obsm.items():
        n = v.shape[1] if v.ndim == 2 else 1
        obsm[k] = min(n, 50)
    obs_num, obs_cat = [], []
    for c in adata.obs.columns:
        s = adata.obs[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object or pd.api.types.is_bool_dtype(s):
            obs_cat.append(c)
        else:
            obs_num.append(c)
    return {
        "features": list(map(str, adata.var_names)),
        "has_raw": "raw" in adata.layers,
        "obs_numeric": obs_num,
        "obs_categorical": obs_cat,
        "obsm": obsm,
    }


_SPEC = re.compile(r"^(feat|raw|obs|obsm):(.+)$")


def parse_spec(spec: str) -> tuple[str, str, int | None]:
    m = _SPEC.match(spec)
    if not m:
        raise ValueError(f"bad column spec {spec!r}; expected feat:/raw:/obs:/obsm:")
    kind, rest = m.groups()
    if kind == "obsm":
        key, _, dim = rest.rpartition(":")
        return kind, key, int(dim)
    return kind, rest, None


def get_vector(adata: ad.AnnData, spec: str) -> np.ndarray | pd.Categorical:
    """Resolve a column spec to a numpy vector (or pandas Categorical)."""
    kind, name, dim = parse_spec(spec)
    if kind == "feat":
        j = adata.var_names.get_loc(name)
        col = adata.X[:, j]
        return np.asarray(col.toarray()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
    if kind == "raw":
        j = adata.var_names.get_loc(name)
        col = adata.layers["raw"][:, j]
        return np.asarray(col.toarray()).ravel() if sp.issparse(col) else np.asarray(col).ravel()
    if kind == "obs":
        if name not in adata.obs and name in adata.var_names:   # tolerate feat/obs confusion
            return get_vector(adata, f"feat:{name}")
        s = adata.obs[name]
        if isinstance(s.dtype, pd.CategoricalDtype):
            return s.values
        if pd.api.types.is_bool_dtype(s):
            return pd.Categorical(s.map({True: "in", False: "out"}), categories=["out", "in"])
        if s.dtype == object:
            return pd.Categorical(s.astype(str))
        return s.to_numpy(dtype=float)
    m = adata.obsm[name]
    m = m.to_numpy() if isinstance(m, pd.DataFrame) else m
    return np.asarray(m[:, dim], dtype=float)


def feature_matrix(adata: ad.AnnData, layer: str | None = None, features: list[str] | None = None) -> np.ndarray:
    """Dense float32 matrix of X (or a layer), optionally restricted to features."""
    M = adata.layers[layer] if layer else adata.X
    if features:
        M = M[:, adata.var_names.get_indexer(features)]
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    return M.astype(np.float32, copy=False)
