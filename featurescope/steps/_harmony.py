"""Pure-NumPy Harmony integration (Korsunsky et al., Nat. Methods 2019).

Recent ``harmonypy`` releases are C++ extensions without Windows wheels, so
installing them needs a compiler.  This implementation follows the published
algorithm (soft k-means with a batch-diversity penalty, then a mixture-of-experts
ridge correction) with the same defaults, and has no compiled dependencies.
"""
from __future__ import annotations

import numpy as np


def run_harmony(Z: np.ndarray, batch: np.ndarray, theta: float = 2.0, sigma: float = 0.1, lamb: float = 1.0,
                n_clusters: int | None = None, max_iter: int = 10, max_iter_kmeans: int = 20,
                eps_cluster: float = 1e-5, eps_harmony: float = 1e-4, block_size: float = 0.05,
                seed: int = 0) -> np.ndarray:
    """Return the batch-corrected embedding (n x d) for embedding ``Z`` and batch labels ``batch``."""
    from sklearn.cluster import KMeans, MiniBatchKMeans

    rng = np.random.default_rng(seed)
    Z_orig = np.asarray(Z, np.float64).T                      # d x n
    n = Z_orig.shape[1]
    codes = np.unique(np.asarray(batch).astype(str), return_inverse=True)[1]
    B = codes.max() + 1
    Phi = np.zeros((B, n))
    Phi[codes, np.arange(n)] = 1
    Pr_b = Phi.sum(1) / n
    Phi_moe = np.vstack([np.ones(n), Phi])
    lamb_mat = np.diag(np.r_[0.0, np.full(B, lamb)])
    K = n_clusters or int(min(round(n / 30.0), 100))
    K = max(2, min(K, n - 1))
    theta_b = np.full(B, theta)

    def cosnorm(M):
        return M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), 1e-12)

    Z_cos = cosnorm(Z_orig)
    km = (MiniBatchKMeans(K, random_state=seed, n_init=3, batch_size=4096) if n > 100_000
          else KMeans(K, random_state=seed, n_init=4))
    Y = cosnorm(km.fit(Z_cos.T).cluster_centers_.T)            # d x K
    dist = 2 * (1 - Y.T @ Z_cos)                              # K x n
    R = np.exp(-dist / sigma)
    R /= R.sum(0)
    E = np.outer(R.sum(1), Pr_b)                              # expected  K x B
    O = R @ Phi.T                                             # observed  K x B

    def objective():
        kmeans_err = np.sum(R * dist)
        entropy = sigma * np.sum(R * np.log(np.maximum(R, 1e-300)))
        cross = sigma * np.sum(R * ((theta_b[None, :] * np.log((O + 1) / (E + 1))) @ Phi))
        return kmeans_err + entropy + cross

    def update_R():
        nonlocal E, O
        scale = -dist / sigma
        scale -= scale.max(0)
        scale = np.exp(scale)
        order = rng.permutation(n)
        nb = int(np.ceil(1 / block_size))
        for blk in np.array_split(order, nb):
            Rb, Pb = R[:, blk], Phi[:, blk]
            E -= np.outer(Rb.sum(1), Pr_b)
            O -= Rb @ Pb.T
            Rb = scale[:, blk] * ((((E + 1) / (O + 1)) ** theta_b[None, :]) @ Pb)
            Rb /= Rb.sum(0)
            R[:, blk] = Rb
            E += np.outer(Rb.sum(1), Pr_b)
            O += Rb @ Pb.T

    obj_harmony = [objective()]
    Z_corr = Z_orig.copy()
    for _ in range(max_iter):
        # --- clustering with diversity penalty
        obj_k = [objective()]
        for _ in range(max_iter_kmeans):
            Y = cosnorm(Z_cos @ R.T)
            dist = 2 * (1 - Y.T @ Z_cos)
            update_R()
            obj_k.append(objective())
            if len(obj_k) > 3 and abs(np.mean(obj_k[-4:-1]) - np.mean(obj_k[-3:])) < eps_cluster * abs(np.mean(obj_k[-4:-1])):
                break
        # --- mixture-of-experts ridge regression correction
        Z_corr = Z_orig.copy()
        for k in range(K):
            Phi_Rk = Phi_moe * R[k]
            W = np.linalg.solve(Phi_Rk @ Phi_moe.T + lamb_mat, Phi_Rk @ Z_orig.T)
            W[0] = 0                                          # keep the intercept (biology)
            Z_corr -= W.T @ Phi_Rk
        Z_cos = cosnorm(Z_corr)
        obj_harmony.append(objective())
        if abs(obj_harmony[-2] - obj_harmony[-1]) < eps_harmony * abs(obj_harmony[-2]):
            break
    return Z_corr.T
