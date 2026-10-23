"""Batch-effect correction."""
from __future__ import annotations

import numpy as np

from ..data import feature_matrix
from ..pipeline.delta import Delta
from ..pipeline.registry import Param, op

P = Param


def _batch_key(adata, batch_key):
    key = batch_key or adata.uns.get("batch_key")
    if not key or key not in adata.obs:
        raise ValueError("batch column required (param batch_key or 'Set batch column' step)")
    return key


@op("combat", "Batch correction", "ComBat", params=[
    P("batch_key", "str", None, help="obs column (default: dataset batch column)"),
    P("covariates", "columns", None, help="Biological covariates to preserve (obs columns)"),
])
def combat(adata, ctx, batch_key, covariates):
    """Empirical-Bayes location/scale adjustment of every feature (Johnson et al. 2007)."""
    import scanpy as sc
    key = _batch_key(adata, batch_key)
    tmp = adata.copy()
    tmp.X = feature_matrix(adata).astype(np.float64)
    cov = [c.split(":", 1)[-1] for c in covariates] if covariates else None
    X = sc.pp.combat(tmp, key=key, covariates=cov, inplace=False)
    return Delta(X=np.asarray(X, np.float32), log=f"ComBat over '{key}'")


@op("harmony", "Batch correction", "Harmony (embedding)", params=[
    P("batch_key", "str", None),
    P("basis", "str", "X_pca", help="Embedding to correct (run PCA first)"),
    P("theta", "float", 2.0, help="Diversity penalty"),
    P("max_iter", "int", 20),
    P("implementation", "choice", "builtin", choices=["builtin", "harmonypy"],
      help="builtin = pure NumPy, same result on every machine; harmonypy needs that package installed"),
])
def harmony(adata, ctx, batch_key, basis, theta, max_iter, implementation):
    """Iterative soft-clustering integration of an embedding (Korsunsky et al. 2019). Writes X_harmony."""
    key = _batch_key(adata, batch_key)
    if basis not in adata.obsm:
        raise ValueError(f"{basis} not found; add a PCA step before Harmony")
    Z = np.asarray(adata.obsm[basis], np.float64)
    if implementation == "harmonypy":
        import harmonypy
        ho = harmonypy.run_harmony(Z, adata.obs[[key]], [key], theta=theta, max_iter_harmony=max_iter,
                                   random_state=ctx.seed, verbose=False)
        Zc = np.asarray(ho.Z_corr)
        if Zc.shape[0] != Z.shape[0]:   # harmonypy < 0.1 returns (dims x cells)
            Zc = Zc.T
    else:
        from ._harmony import run_harmony
        Zc = run_harmony(Z, adata.obs[key].to_numpy(), theta=theta, max_iter=max_iter, seed=ctx.seed)
    return Delta(obsm={"X_harmony": Zc.astype(np.float32)}, log=f"Harmony ({implementation}) over '{key}'")


@op("batch_center", "Batch correction", "Per-batch centering/scaling", params=[
    P("batch_key", "str", None),
    P("scale", "bool", True),
    P("reference", "str", None, help="Align every batch to this batch's distribution instead of 0/1"),
])
def batch_center(adata, ctx, batch_key, scale, reference):
    """Align per-batch feature medians (and spreads); robust for imaging/cytometry slides & plates."""
    key = _batch_key(adata, batch_key)
    X = feature_matrix(adata).copy()
    labels = adata.obs[key].astype(str).to_numpy()
    if reference is not None:
        ref = X[labels == reference]
        tgt_med, tgt_sd = np.nanmedian(ref, 0), np.nanstd(ref, 0)
    else:
        tgt_med, tgt_sd = 0.0, 1.0
    for b in np.unique(labels):
        i = labels == b
        med = np.nanmedian(X[i], 0)
        X[i] -= med
        if scale:
            sd = np.nanstd(X[i], 0)
            X[i] /= np.where(sd > 0, sd, 1)
            X[i] *= tgt_sd
        X[i] += tgt_med
    return Delta(X=X, log=f"centered {len(np.unique(labels))} batches")
