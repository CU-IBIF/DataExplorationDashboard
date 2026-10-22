"""Clustering and marker discovery."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import feature_matrix
from ..pipeline.delta import Delta
from ..pipeline.registry import Param, op
from .reduce import BACKENDS, _rep, _use_gpu

P = Param


def _labels(a) -> pd.Categorical:
    a = np.asarray(a)
    order = pd.Series(a).value_counts().index  # cluster 0 = largest
    remap = {old: str(i) for i, old in enumerate(order)}
    lab = [remap[x] for x in a]
    return pd.Categorical(lab, categories=[str(i) for i in range(len(order))])


@op("leiden", "Clustering", "Leiden (graph)", params=[
    P("resolution", "float", 1.0),
    P("key", "str", "leiden"),
    P("backend", "choice", "auto", choices=BACKENDS),
], gpu=True)
def leiden(adata, ctx, resolution, key, backend):
    """Community detection on the kNN graph (Traag et al. 2019); PhenoGraph-style when k~30 on PCA."""
    if "connectivities" not in adata.obsp:
        raise ValueError("Leiden needs a kNN graph: add a 'kNN graph' step first")
    tmp = adata.copy()
    if _use_gpu(ctx, backend):
        import rapids_singlecell as rsc
        rsc.tl.leiden(tmp, resolution=resolution, key_added=key, random_state=ctx.seed)
    else:
        import scanpy as sc
        sc.tl.leiden(tmp, resolution=resolution, key_added=key, random_state=ctx.seed,
                     flavor="igraph", n_iterations=2, directed=False)
    lab = _labels(tmp.obs[key])
    return Delta(obs={key: lab}, log=f"{len(lab.categories)} clusters")


@op("louvain", "Clustering", "Louvain (graph)", params=[
    P("resolution", "float", 1.0), P("key", "str", "louvain"),
])
def louvain(adata, ctx, resolution, key):
    """Louvain modularity clustering on the kNN graph (igraph implementation)."""
    import igraph as ig
    from scipy import sparse
    if "connectivities" not in adata.obsp:
        raise ValueError("needs a kNN graph")
    A = sparse.triu(adata.obsp["connectivities"], 1).tocoo()
    g = ig.Graph(n=adata.n_obs, edges=list(zip(A.row.tolist(), A.col.tolist())), edge_attrs={"weight": A.data})
    np.random.seed(ctx.seed)
    part = g.community_multilevel(weights="weight", resolution=resolution)
    lab = _labels(part.membership)
    return Delta(obs={key: lab}, log=f"{len(lab.categories)} clusters")


@op("kmeans", "Clustering", "k-means", params=[
    P("n_clusters", "int", 10, min=2),
    P("use_rep", "str", "X_pca"),
    P("key", "str", "kmeans"),
])
def kmeans(adata, ctx, n_clusters, use_rep, key):
    """k-means (mini-batch above 100k objects)."""
    from sklearn.cluster import KMeans, MiniBatchKMeans
    Z = _rep(adata, use_rep)
    model = (MiniBatchKMeans(n_clusters, random_state=ctx.seed, batch_size=4096, n_init=3)
             if len(Z) > 100_000 else KMeans(n_clusters, random_state=ctx.seed, n_init=4))
    lab = _labels(model.fit_predict(Z))
    return Delta(obs={key: lab}, log=f"k={n_clusters}")


@op("gmm", "Clustering", "Gaussian mixture", params=[
    P("n_components", "int", 8), P("use_rep", "str", "X_pca"), P("key", "str", "gmm"),
    P("covariance", "choice", "full", choices=["full", "diag", "tied", "spherical"]),
])
def gmm(adata, ctx, n_components, use_rep, key, covariance):
    """Gaussian mixture model; also stores max posterior probability as a confidence score."""
    from sklearn.mixture import GaussianMixture
    Z = _rep(adata, use_rep)
    m = GaussianMixture(n_components, covariance_type=covariance, random_state=ctx.seed)
    sub = np.random.default_rng(ctx.seed).choice(len(Z), min(len(Z), 200_000), replace=False)
    m.fit(Z[sub])
    proba = m.predict_proba(Z)
    return Delta(obs={key: _labels(proba.argmax(1)), f"{key}_confidence": proba.max(1)}, log=f"{n_components} components")


@op("flowsom", "Clustering", "FlowSOM-style SOM + metaclustering", params=[
    P("grid", "int", 10, help="SOM grid side (grid x grid nodes)"),
    P("n_meta", "int", 12),
    P("use_rep", "str", "X"),
    P("iterations", "int", 10),
    P("key", "str", "flowsom"),
])
def flowsom(adata, ctx, grid, n_meta, use_rep, iterations, key):
    """Self-organizing map followed by hierarchical metaclustering (Van Gassen et al. 2015). Fast on millions."""
    from scipy import sparse
    from scipy.cluster.hierarchy import fcluster, linkage
    Z = _rep(adata, use_rep).astype(np.float32)
    rng = np.random.default_rng(ctx.seed)
    n_nodes = grid * grid
    W = Z[rng.choice(len(Z), n_nodes, replace=len(Z) < n_nodes)].copy()
    gx, gy = np.divmod(np.arange(n_nodes), grid)
    coords = np.column_stack([gx, gy]).astype(np.float32)
    grid_d2 = ((coords[:, None] - coords[None]) ** 2).sum(-1)

    def assign(W):
        node = np.empty(len(Z), int)
        for s in range(0, len(Z), 50_000):
            blk = Z[s:s + 50_000]
            d = (blk ** 2).sum(1)[:, None] - 2 * blk @ W.T + (W ** 2).sum(1)[None]
            node[s:s + 50_000] = d.argmin(1)
        return node

    # batch SOM: each epoch assigns every object, then moves nodes to neighborhood-weighted means
    for it in range(iterations):
        r = max(grid / 2 * (1 - it / iterations), 0.5)
        node = assign(W)
        H = np.exp(-grid_d2 / (2 * r * r)).astype(np.float32)          # node x node
        counts = np.bincount(node, minlength=n_nodes).astype(np.float32)
        onehot = sparse.csr_matrix((np.ones(len(Z), np.float32), (node, np.arange(len(Z)))), shape=(n_nodes, len(Z)))
        sums = np.asarray(onehot @ Z)
        num, den = H @ sums, H @ counts
        W = np.where(den[:, None] > 0, num / np.maximum(den[:, None], 1e-12), W)
    node = assign(W)
    meta = fcluster(linkage(W, "average"), n_meta, criterion="maxclust") - 1
    return Delta(obs={key: _labels(meta[node]), f"{key}_node": node.astype(np.int32)},
                 uns={f"{key}_som": {"weights": W, "grid": grid}}, log=f"{n_nodes} nodes -> {n_meta} metaclusters")


@op("markers", "Clustering", "Marker features per group", params=[
    P("groupby", "str", "leiden"),
    P("method", "choice", "wilcoxon", choices=["wilcoxon", "t-test", "t-test_overestim_var", "logreg"]),
    P("layer", "str", None, help="Layer to test (default current X)"),
    P("max_cells_per_group", "int", 5000, help="Subsample for speed on large data"),
])
def markers(adata, ctx, groupby, method, layer, max_cells_per_group):
    """Rank features distinguishing each group from the rest (also stores group-mean heatmap)."""
    import scanpy as sc
    tmp = adata.copy()
    tmp.X = feature_matrix(adata, layer)
    g = tmp.obs[groupby].astype(str)
    rng = np.random.default_rng(ctx.seed)
    keep = np.concatenate([rng.permutation(np.where(g == c)[0])[:max_cells_per_group] for c in g.unique()])
    tmp = tmp[np.sort(keep)].copy()
    tmp.obs[groupby] = tmp.obs[groupby].astype(str).astype("category")
    sc.tl.rank_genes_groups(tmp, groupby, method=method, use_raw=False)
    df = sc.get.rank_genes_groups_df(tmp, group=None)
    X = feature_matrix(adata, layer)
    means = pd.DataFrame(X, columns=adata.var_names).groupby(g.values).mean()
    return Delta(uns={f"markers_{groupby}": {"table": df, "group_means": means, "method": method}},
                 log=f"markers for {g.nunique()} groups")
