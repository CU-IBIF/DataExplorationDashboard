"""Spatial / graph-based association analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..data import feature_matrix
from ..pipeline.delta import Delta
from ..pipeline.registry import Param, op

P = Param


def _groups_of(adata, key):
    """Spatial neighborhoods must not cross images/slides: per-group index sets."""
    if key and key in adata.obs:
        lab = adata.obs[key].astype(str).to_numpy()
        return [np.where(lab == g)[0] for g in np.unique(lab)]
    return [np.arange(adata.n_obs)]


@op("spatial_neighbors", "Graph", "Spatial neighbor graph", params=[
    P("method", "choice", "knn", choices=["knn", "radius", "delaunay"]),
    P("n_neighbors", "int", 10),
    P("radius", "float", 50.0, help="Same units as the coordinates (px or µm)"),
    P("spatial_key", "str", "spatial"),
    P("library_key", "str", None, help="obs column separating images/slides (default: batch column)"),
])
def spatial_neighbors(adata, ctx, method, n_neighbors, radius, spatial_key, library_key):
    """Physical-proximity graph between objects (writes obsp['spatial_connectivities'])."""
    from sklearn.neighbors import NearestNeighbors
    if spatial_key not in adata.obsm:
        raise ValueError("no spatial coordinates (import x/y centroid columns)")
    xy = np.asarray(adata.obsm[spatial_key], float)[:, :2]
    rows, cols, dists = [], [], []
    for idx in _groups_of(adata, library_key or adata.uns.get("batch_key")):
        if len(idx) < 2:
            continue
        pts = xy[idx]
        if method == "delaunay" and len(idx) > 3:
            from scipy.spatial import Delaunay
            tri = Delaunay(pts).simplices
            e = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [0, 2]]])
            e = np.unique(np.sort(e, 1), axis=0)
            r, c = np.concatenate([e[:, 0], e[:, 1]]), np.concatenate([e[:, 1], e[:, 0]])
            d = np.linalg.norm(pts[r] - pts[c], axis=1)
        else:
            nn = NearestNeighbors(n_neighbors=min(n_neighbors + 1, len(idx)), radius=radius).fit(pts)
            if method == "radius":
                dist, ind = nn.radius_neighbors(pts, radius=radius)
                r = np.repeat(np.arange(len(idx)), [len(i) for i in ind])
                c, d = np.concatenate(ind), np.concatenate(dist)
            else:
                dist, ind = nn.kneighbors(pts)
                r = np.repeat(np.arange(len(idx)), ind.shape[1])
                c, d = ind.ravel(), dist.ravel()
            m = r != c
            r, c, d = r[m], c[m], d[m]
        rows.append(idx[r]); cols.append(idx[c]); dists.append(d)
    r, c, d = (np.concatenate(a) if a else np.array([], int) for a in (rows, cols, dists))
    n = adata.n_obs
    A = sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float32)
    D = sp.csr_matrix((d.astype(np.float32), (r, c)), shape=(n, n))
    deg = np.asarray(A.sum(1)).ravel()
    return Delta(obsp={"spatial_connectivities": A, "spatial_distances": D},
                 obs={"spatial_degree": deg.astype(np.float32)},
                 log=f"{method}: mean degree {deg.mean():.1f}")


def _graph(adata, graph):
    key = {"spatial": "spatial_connectivities", "knn": "connectivities"}.get(graph, graph)
    if key not in adata.obsp:
        raise ValueError(f"graph {key!r} not found; build it first")
    A = adata.obsp[key].tocsr().astype(np.float32)
    A.data[:] = 1
    return A


@op("nhood_enrichment", "Graph", "Neighborhood enrichment", params=[
    P("cluster_key", "str", "leiden"),
    P("graph", "choice", "spatial", choices=["spatial", "knn"]),
    P("n_perms", "int", 200),
])
def nhood_enrichment(adata, ctx, cluster_key, graph, n_perms):
    """Are cluster pairs neighbors more often than chance? Permutation z-scores (as in histoCAT/squidpy)."""
    A = _graph(adata, graph)
    cat = adata.obs[cluster_key].astype("category")
    codes, names = cat.cat.codes.to_numpy(), list(map(str, cat.cat.categories))
    k = len(names)

    def counts(c):
        M = sp.csr_matrix((np.ones(len(c)), (np.arange(len(c)), c)), shape=(len(c), k))
        return np.asarray((M.T @ A @ M).todense())

    obs = counts(codes)
    rng = np.random.default_rng(ctx.seed)
    perms = np.stack([counts(rng.permutation(codes)) for _ in range(n_perms)])
    mu, sd = perms.mean(0), perms.std(0)
    z = (obs - mu) / np.where(sd > 0, sd, 1)
    return Delta(uns={f"nhood_enrichment_{cluster_key}": {
        "zscore": pd.DataFrame(z, index=names, columns=names), "count": pd.DataFrame(obs, index=names, columns=names),
        "graph": graph}}, log=f"{k}x{k} enrichment, {n_perms} permutations")


@op("cellular_neighborhoods", "Graph", "Cellular neighborhoods (niches)", params=[
    P("cluster_key", "str", "leiden"),
    P("graph", "choice", "spatial", choices=["spatial", "knn"]),
    P("n_neighborhoods", "int", 10),
    P("key", "str", "neighborhood"),
])
def cellular_neighborhoods(adata, ctx, cluster_key, graph, n_neighborhoods, key):
    """Cluster objects by the composition of their neighbors (Schürch et al. 2020, CODEX CN analysis)."""
    from sklearn.cluster import MiniBatchKMeans
    from .cluster import _labels
    A = _graph(adata, graph)
    A = A + sp.identity(A.shape[0], format="csr", dtype=np.float32)
    cat = adata.obs[cluster_key].astype("category")
    codes, names = cat.cat.codes.to_numpy(), list(map(str, cat.cat.categories))
    M = sp.csr_matrix((np.ones(len(codes), np.float32), (np.arange(len(codes)), codes)), shape=(len(codes), len(names)))
    comp = np.asarray((A @ M).todense())
    comp /= np.maximum(comp.sum(1, keepdims=True), 1)
    lab = MiniBatchKMeans(n_neighborhoods, random_state=ctx.seed, n_init=3, batch_size=4096).fit_predict(comp)
    lab = _labels(lab)
    prof = pd.DataFrame(comp, columns=names).groupby(np.asarray(lab)).mean()
    return Delta(obs={key: lab}, obsm={f"X_{key}_composition": comp.astype(np.float32)},
                 uns={f"{key}_profiles": prof}, log=f"{n_neighborhoods} neighborhoods")


@op("neighbor_aggregate", "Graph", "Aggregate neighbor features", params=[
    P("graph", "choice", "spatial", choices=["spatial", "knn"]),
    P("hops", "int", 2, help="Concatenate mean features of 1..hops-hop neighborhoods"),
    P("use_rep", "str", "X_pca"),
    P("key", "str", "X_niche"),
])
def neighbor_aggregate(adata, ctx, graph, hops, use_rep, key):
    """Niche representation = [self, mean of 1-hop, ..., mean of n-hop neighbors] (CellCharter-style)."""
    from .reduce import _rep
    A = _graph(adata, graph)
    deg = np.asarray(A.sum(1)).ravel()
    Dinv = sp.diags(1 / np.maximum(deg, 1))
    P_ = Dinv @ A
    Z = _rep(adata, use_rep)
    parts, cur = [Z], Z
    for _ in range(hops):
        cur = P_ @ cur
        parts.append(cur)
    return Delta(obsm={key: np.hstack(parts).astype(np.float32)}, log=f"{hops}-hop aggregation of {use_rep}")


@op("cluster_connectivity", "Graph", "Cluster connectivity (PAGA-style)", params=[
    P("cluster_key", "str", "leiden"),
    P("graph", "choice", "knn", choices=["knn", "spatial"]),
])
def cluster_connectivity(adata, ctx, cluster_key, graph):
    """How strongly clusters connect: observed inter-cluster edges / expected under a degree-preserving null.

    Same idea as PAGA (Wolf et al. 2019): values near/above 1 mean the clusters form a continuum
    (e.g. a differentiation path); values near 0 mean well-separated populations.
    """
    A = _graph(adata, graph)
    cat = adata.obs[cluster_key].astype("category")
    codes, names = cat.cat.codes.to_numpy(), list(map(str, cat.cat.categories))
    M = sp.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)), shape=(len(codes), len(names)))
    E = np.asarray((M.T @ A @ M).todense(), float)
    deg = E.sum(1)
    expected = np.outer(deg, deg) / max(E.sum(), 1)
    C = np.divide(E, expected, out=np.zeros_like(E), where=expected > 0)
    np.fill_diagonal(C, 0)
    return Delta(uns={f"paga_{cluster_key}": pd.DataFrame(np.clip(C, 0, None), index=names, columns=names)},
                 log=f"{len(names)} clusters on {graph} graph")


@op("spatial_autocorr", "Graph", "Spatial autocorrelation (Moran's I)", params=[
    P("graph", "choice", "spatial", choices=["spatial", "knn"]),
])
def spatial_autocorr(adata, ctx, graph):
    """Moran's I per feature: which markers are spatially patterned vs. salt-and-pepper."""
    A = _graph(adata, graph)
    X = np.nan_to_num(feature_matrix(adata))
    Xc = X - X.mean(0)
    W = A.sum()
    num = (Xc * (A @ Xc)).sum(0)
    den = (Xc ** 2).sum(0)
    I = (len(X) / W) * num / np.where(den > 0, den, 1)
    return Delta(var={"morans_i": I.astype(np.float32)},
                 uns={"morans_i": pd.DataFrame({"morans_i": I}, index=adata.var_names).sort_values("morans_i", ascending=False)},
                 log=f"top: {adata.var_names[np.argmax(I)]}")
