"""Quality control, transformation, normalization and feature selection."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..data import feature_matrix, get_vector
from ..pipeline.delta import Delta
from ..pipeline.registry import Param, op

P = Param


def _feature_subset(adata, features) -> np.ndarray:
    """Column indices for a feature list or a regex string ('' / None = all)."""
    if not features:
        return np.arange(adata.n_vars)
    if isinstance(features, str):
        rx = re.compile(features)
        return np.array([i for i, f in enumerate(adata.var_names) if rx.search(f)], dtype=int)
    return adata.var_names.get_indexer(features)


# ----------------------------------------------------------------------------- QC / filtering

@op("filter_objects", "QC & filtering", "Filter objects", params=[
    P("column", "column", None, help="Column spec, e.g. obs:area or raw:DAPI"),
    P("min", "float", None), P("max", "float", None),
    P("quantile_low", "float", None, help="Drop below this quantile (0-1)", min=0, max=1),
    P("quantile_high", "float", None, help="Drop above this quantile (0-1)", min=0, max=1),
    P("drop_nan", "bool", True, help="Drop objects with any NaN feature"),
])
def filter_objects(adata, ctx, column, min, max, quantile_low, quantile_high, drop_nan):
    """Remove objects by a threshold on any column (size, intensity, QC score) and/or NaN rows."""
    keep = np.ones(adata.n_obs, bool)
    if drop_nan:
        keep &= ~np.isnan(feature_matrix(adata)).any(1)
    if column:
        v = np.asarray(get_vector(adata, column), float)
        if min is not None:
            keep &= v >= min
        if max is not None:
            keep &= v <= max
        if quantile_low is not None:
            keep &= v >= np.nanquantile(v, quantile_low)
        if quantile_high is not None:
            keep &= v <= np.nanquantile(v, quantile_high)
    return Delta(obs_mask=keep, log=f"kept {keep.sum()}/{adata.n_obs} objects")


@op("subset_gate", "QC & filtering", "Subset to gate", uses_gates=True, params=[
    P("gate", "gate", None, help="Keep objects inside this gate"),
    P("invert", "bool", False),
])
def subset_gate(adata, ctx, gate, invert):
    """Keep only objects inside a gate drawn in the dashboard (e.g. live, single, CD45+)."""
    from ..gating import evaluate_gate
    m = evaluate_gate(adata, ctx.gates, gate)
    if invert:
        m = ~m
    return Delta(obs_mask=m, log=f"gate {gate}: kept {m.sum()}/{adata.n_obs}")


@op("select_features", "QC & filtering", "Select features", params=[
    P("features", "features", None, help="Features to keep (list) or regex"),
    P("exclude", "features", None, help="Features to drop (list) or regex"),
    P("min_variance", "float", None, help="Drop near-constant features"),
])
def select_features(adata, ctx, features, exclude, min_variance):
    """Keep/drop features by name or pattern (e.g. drop DNA/background channels)."""
    keep = np.zeros(adata.n_vars, bool)
    keep[_feature_subset(adata, features)] = True
    if exclude:
        keep[_feature_subset(adata, exclude)] = False
    if min_variance is not None:
        keep &= np.nanvar(feature_matrix(adata), 0) >= min_variance
    return Delta(var_mask=keep, log=f"kept {keep.sum()}/{adata.n_vars} features")


@op("impute", "QC & filtering", "Impute missing values", params=[
    P("method", "choice", "median", choices=["median", "mean", "zero", "knn"]),
    P("n_neighbors", "int", 5),
])
def impute(adata, ctx, method, n_neighbors):
    """Fill NaN feature values."""
    X = feature_matrix(adata).copy()
    n_missing = int(np.isnan(X).sum())
    if method == "knn":
        from sklearn.impute import KNNImputer
        X = KNNImputer(n_neighbors=n_neighbors).fit_transform(X).astype(np.float32)
    else:
        fill = {"median": np.nanmedian, "mean": np.nanmean}.get(method)
        vals = fill(X, 0) if fill else np.zeros(X.shape[1], np.float32)
        r, c = np.where(np.isnan(X))
        X[r, c] = vals[c]
    return Delta(X=X, log=f"imputed {n_missing} values")


# ----------------------------------------------------------------------------- transforms

@op("transform", "Transform & normalize", "Transform", params=[
    P("method", "choice", "arcsinh", choices=["arcsinh", "log1p", "log2p", "sqrt", "yeo-johnson", "none"]),
    P("cofactor", "float", 5.0, help="arcsinh cofactor (5 for CyTOF, 150+ for flow / imaging intensities)"),
    P("features", "features", None, help="Restrict to these features (default all)"),
])
def transform(adata, ctx, method, cofactor, features):
    """Variance-stabilizing transform of intensities or counts."""
    X = feature_matrix(adata).copy()
    j = _feature_subset(adata, features)
    sub = X[:, j]
    if method == "arcsinh":
        sub = np.arcsinh(sub / cofactor)
    elif method == "log1p":
        sub = np.log1p(np.clip(sub, 0, None))
    elif method == "log2p":
        sub = np.log2(np.clip(sub, 0, None) + 1)
    elif method == "sqrt":
        sub = np.sqrt(np.clip(sub, 0, None))
    elif method == "yeo-johnson":
        from sklearn.preprocessing import PowerTransformer
        sub = PowerTransformer(method="yeo-johnson", standardize=False).fit_transform(sub)
    X[:, j] = sub
    return Delta(X=X.astype(np.float32), log=f"{method} on {len(j)} features")


@op("clip", "Transform & normalize", "Clip outliers (winsorize)", params=[
    P("low", "float", 0.001, help="Lower quantile", min=0, max=1),
    P("high", "float", 0.999, help="Upper quantile", min=0, max=1),
    P("by_batch", "bool", False),
])
def clip(adata, ctx, low, high, by_batch):
    """Per-feature percentile clipping; standard for imaging intensities with hot pixels/debris."""
    X = feature_matrix(adata).copy()
    groups = _batches(adata) if by_batch else [np.arange(adata.n_obs)]
    for idx in groups:
        lo, hi = np.nanquantile(X[idx], [low, high], axis=0)
        X[idx] = np.clip(X[idx], lo, hi)
    return Delta(X=X, log=f"clipped to [{low}, {high}] quantiles")


@op("normalize", "Transform & normalize", "Normalize", params=[
    P("method", "choice", "zscore", choices=["zscore", "robust", "minmax", "total", "quantile", "rank_gauss", "l2_row"]),
    P("target_sum", "float", 1e4, help="For 'total' (count data): scale each object to this sum"),
    P("max_value", "float", None, help="Clip z-scores at +/- this value"),
    P("by_batch", "bool", False, help="Normalize within each batch separately (simple batch alignment)"),
])
def normalize(adata, ctx, method, target_sum, max_value, by_batch):
    """Per-feature (column) or per-object (row) normalization.

    zscore / robust (median, IQR) / minmax / rank_gauss / quantile operate on
    features; total and l2_row operate on objects (library-size style).
    """
    X = feature_matrix(adata).copy()
    groups = _batches(adata) if by_batch else [np.arange(adata.n_obs)]
    for idx in groups:
        X[idx] = _normalize_block(X[idx], method, target_sum)
    if max_value is not None:
        X = np.clip(X, -max_value, max_value)
    return Delta(X=X.astype(np.float32), log=f"{method}{' per batch' if by_batch else ''}")


def _normalize_block(X, method, target_sum):
    if method == "zscore":
        sd = np.nanstd(X, 0)
        return (X - np.nanmean(X, 0)) / np.where(sd > 0, sd, 1)
    if method == "robust":
        q1, med, q3 = np.nanquantile(X, [0.25, 0.5, 0.75], axis=0)
        iqr = q3 - q1
        return (X - med) / np.where(iqr > 0, iqr, 1)
    if method == "minmax":
        lo, hi = np.nanmin(X, 0), np.nanmax(X, 0)
        return (X - lo) / np.where(hi > lo, hi - lo, 1)
    if method == "total":
        s = np.nansum(X, 1, keepdims=True)
        return X / np.where(s > 0, s, 1) * target_sum
    if method == "l2_row":
        n = np.linalg.norm(np.nan_to_num(X), axis=1, keepdims=True)
        return X / np.where(n > 0, n, 1)
    if method in ("quantile", "rank_gauss"):
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=min(1000, len(X)), subsample=200_000,
                                 output_distribution="normal" if method == "rank_gauss" else "uniform")
        return qt.fit_transform(X)
    raise ValueError(method)


def _batches(adata):
    key = adata.uns.get("batch_key")
    if not key or key not in adata.obs:
        raise ValueError("no batch column set; choose one in the Data panel or set adata.uns['batch_key']")
    codes = adata.obs[key].astype("category").cat.codes.to_numpy()
    return [np.where(codes == c)[0] for c in np.unique(codes)]


@op("highly_variable", "Transform & normalize", "Highly variable features", params=[
    P("n_top", "int", 2000),
    P("flavor", "choice", "dispersion", choices=["dispersion", "seurat_v3", "cell_ranger", "variance"]),
    P("subset", "bool", True),
])
def highly_variable(adata, ctx, n_top, flavor, subset):
    """Select informative features (transcriptomics-style HVG; 'variance' for any data)."""
    if flavor == "variance":
        v = np.nanvar(feature_matrix(adata), 0)
        hv = np.zeros(adata.n_vars, bool)
        hv[np.argsort(-v)[:n_top]] = True
    else:
        import scanpy as sc
        tmp = adata.copy()
        kw = {"layer": "raw"} if flavor == "seurat_v3" and "raw" in tmp.layers else {}
        f = "seurat" if flavor == "dispersion" else flavor
        sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_top, adata.n_vars), flavor=f, **kw)
        hv = tmp.var["highly_variable"].to_numpy()
    d = Delta(var={"highly_variable": hv}, log=f"{hv.sum()} highly variable features")
    if subset:
        d.var_mask = hv
    return d


@op("regress_out", "Transform & normalize", "Regress out covariates", params=[
    P("covariates", "columns", None, help="Numeric obs columns, e.g. obs:area, obs:total_intensity"),
])
def regress_out(adata, ctx, covariates):
    """Remove linear effects of nuisance covariates (cell size, cycle score, ...) from every feature."""
    X = feature_matrix(adata)
    C = np.column_stack([np.ones(adata.n_obs)] + [np.asarray(get_vector(adata, c), float) for c in covariates or []])
    beta, *_ = np.linalg.lstsq(C, np.nan_to_num(X), rcond=None)
    resid = X - C[:, 1:] @ beta[1:]
    return Delta(X=resid.astype(np.float32), log=f"regressed {len(covariates or [])} covariates")


@op("restore_raw", "Transform & normalize", "Restore raw values", params=[])
def restore_raw(adata, ctx):
    """Reset X to the imported values (keeps current object/feature subset)."""
    return Delta(X=np.asarray(adata.layers["raw"], dtype=np.float32).copy(), log="X <- raw")


@op("save_layer", "Transform & normalize", "Snapshot X to layer", params=[P("name", "str", "normalized")])
def save_layer(adata, ctx, name):
    """Keep a copy of the current matrix (e.g. before scaling) for plotting/markers."""
    return Delta(layers={name: feature_matrix(adata).copy()}, log=f"layer '{name}'")


@op("set_batch", "Batch correction", "Set batch column", params=[P("column", "str", None, help="obs column")])
def set_batch(adata, ctx, column):
    """Declare which metadata column identifies batches (slide, plate, run, donor)."""
    if column not in adata.obs:
        raise KeyError(column)
    return Delta(uns={"batch_key": column}, obs={column: pd.Categorical(adata.obs[column].astype(str))})
