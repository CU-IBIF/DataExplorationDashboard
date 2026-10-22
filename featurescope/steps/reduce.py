"""Dimensionality reduction and kNN graphs.

When ``rapids-singlecell`` + CuPy are installed and a GPU is visible, PCA,
neighbors, UMAP and t-SNE run on the GPU (10-100x faster on millions of objects).
"""
from __future__ import annotations

import numpy as np

from ..data import feature_matrix
from ..pipeline.delta import Delta
from ..pipeline.registry import Param, op

P = Param
BACKENDS = ["auto", "cpu", "gpu"]


def _use_gpu(ctx, backend):
    return backend == "gpu" or (backend == "auto" and ctx.gpu_available())


def _rep(adata, use_rep):
    if use_rep in (None, "", "X"):
        return np.nan_to_num(feature_matrix(adata))
    if use_rep not in adata.obsm:
        raise ValueError(f"representation {use_rep!r} not found (available: {list(adata.obsm)})")
    return np.asarray(adata.obsm[use_rep], np.float32)


@op("pca", "Dimensionality reduction", "PCA", params=[
    P("n_comps", "int", 30, min=2),
    P("backend", "choice", "auto", choices=BACKENDS),
], gpu=True)
def pca(adata, ctx, n_comps, backend):
    """Principal components of the current feature matrix. Writes obsm['X_pca']."""
    X = np.nan_to_num(feature_matrix(adata))
    k = int(min(n_comps, X.shape[1] - 1, X.shape[0] - 1))
    if _use_gpu(ctx, backend):
        import cupy as cp
        from cuml.decomposition import PCA as cuPCA
        model = cuPCA(n_components=k, random_state=ctx.seed)
        Z = cp.asnumpy(model.fit_transform(cp.asarray(X)))
        evr = cp.asnumpy(model.explained_variance_ratio_)
        loadings = cp.asnumpy(model.components_).T
    else:
        from sklearn.decomposition import PCA
        solver = "randomized" if X.shape[0] > 50_000 and k < X.shape[1] // 2 else "auto"
        model = PCA(n_components=k, svd_solver=solver, random_state=ctx.seed)
        Z = model.fit_transform(X)
        evr, loadings = model.explained_variance_ratio_, model.components_.T
    return Delta(obsm={"X_pca": Z.astype(np.float32)},
                 uns={"pca": {"variance_ratio": evr.tolist(), "loadings": loadings.astype(np.float32),
                              "features": list(map(str, adata.var_names))}},
                 log=f"{k} PCs, {100 * evr.sum():.1f}% variance")


@op("neighbors", "Graph", "kNN graph", params=[
    P("n_neighbors", "int", 15, min=2),
    P("use_rep", "str", "X_pca", help="Embedding to build the graph on (X = all features)"),
    P("metric", "choice", "euclidean", choices=["euclidean", "cosine", "manhattan", "correlation"]),
    P("backend", "choice", "auto", choices=BACKENDS),
], gpu=True, heavy=True)
def neighbors(adata, ctx, n_neighbors, use_rep, metric, backend):
    """Approximate k-nearest-neighbor graph (UMAP fuzzy connectivities). Writes obsp['connectivities']."""
    tmp = adata.copy()
    tmp.obsm["_rep"] = _rep(adata, use_rep)
    if _use_gpu(ctx, backend):
        import rapids_singlecell as rsc
        rsc.pp.neighbors(tmp, n_neighbors=n_neighbors, use_rep="_rep", metric=metric)
    else:
        import scanpy as sc
        sc.pp.neighbors(tmp, n_neighbors=n_neighbors, use_rep="_rep", metric=metric, random_state=ctx.seed)
    return Delta(obsp={"connectivities": tmp.obsp["connectivities"], "distances": tmp.obsp["distances"]},
                 uns={"neighbors": {"connectivities_key": "connectivities", "distances_key": "distances",
                                    "params": {"n_neighbors": n_neighbors, "method": "umap", "use_rep": use_rep,
                                               "metric": metric}}},
                 log=f"k={n_neighbors} on {use_rep}")


@op("umap", "Dimensionality reduction", "UMAP", params=[
    P("min_dist", "float", 0.3), P("spread", "float", 1.0),
    P("n_components", "int", 2, min=2, max=3),
    P("key", "str", "X_umap"),
    P("backend", "choice", "auto", choices=BACKENDS),
], gpu=True, heavy=True)
def umap(adata, ctx, min_dist, spread, n_components, key, backend):
    """UMAP layout of the kNN graph (run 'kNN graph' first)."""
    if "connectivities" not in adata.obsp:
        raise ValueError("UMAP needs a kNN graph: add a 'kNN graph' step first")
    tmp = adata.copy()
    if _use_gpu(ctx, backend):
        import rapids_singlecell as rsc
        rsc.tl.umap(tmp, min_dist=min_dist, spread=spread, n_components=n_components)
    else:
        import scanpy as sc
        sc.tl.umap(tmp, min_dist=min_dist, spread=spread, n_components=n_components, random_state=ctx.seed)
    return Delta(obsm={key: tmp.obsm["X_umap"].astype(np.float32)}, log=f"{key}")


@op("tsne", "Dimensionality reduction", "t-SNE", params=[
    P("perplexity", "float", 30.0),
    P("use_rep", "str", "X_pca"),
    P("backend", "choice", "auto", choices=BACKENDS),
], gpu=True, heavy=True)
def tsne(adata, ctx, perplexity, use_rep, backend):
    """Barnes-Hut / FFT t-SNE of a representation. Writes obsm['X_tsne']."""
    Z = _rep(adata, use_rep)
    if _use_gpu(ctx, backend):
        import cupy as cp
        from cuml.manifold import TSNE
        Y = cp.asnumpy(TSNE(perplexity=perplexity, random_state=ctx.seed).fit_transform(cp.asarray(Z)))
    else:
        from sklearn.manifold import TSNE
        Y = TSNE(perplexity=perplexity, random_state=ctx.seed, init="pca", n_jobs=ctx.n_jobs).fit_transform(Z)
    return Delta(obsm={"X_tsne": Y.astype(np.float32)}, log=f"t-SNE perplexity={perplexity}")


@op("diffmap", "Dimensionality reduction", "Diffusion map + pseudotime", params=[
    P("n_comps", "int", 15),
    P("root_column", "column", None, help="Optional: obs column; root = object with min value"),
])
def diffmap(adata, ctx, n_comps, root_column):
    """Diffusion components of the kNN graph; with a root, diffusion pseudotime (DPT)."""
    import scanpy as sc
    from ..data import get_vector
    if "connectivities" not in adata.obsp:
        raise ValueError("needs a kNN graph")
    tmp = adata.copy()
    sc.tl.diffmap(tmp, n_comps=n_comps)
    d = Delta(obsm={"X_diffmap": tmp.obsm["X_diffmap"].astype(np.float32)}, log="diffusion map")
    if root_column:
        tmp.uns["iroot"] = int(np.nanargmin(np.asarray(get_vector(adata, root_column), float)))
        sc.tl.dpt(tmp)
        d.obs["dpt_pseudotime"] = tmp.obs["dpt_pseudotime"].to_numpy()
    return d
