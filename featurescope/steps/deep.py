"""Deep-learning embeddings (PyTorch; GPU used automatically when available).

* ``vae``          - (conditional) variational autoencoder.  Conditioning on the
                     batch removes batch effects from the latent space, the idea
                     behind scVI/trVAE, but with a Gaussian likelihood suited to
                     continuous imaging / proteomics features.
* ``contrastive``  - SimCLR-style self-supervised encoder with feature-masking and
                     noise augmentations (cf. CLEAR, scCLR, contrastive cytometry).
* ``spatial_dgi``  - graph neural network (GCN) trained with Deep Graph Infomax on
                     the spatial graph; embeds each object *with its microenvironment*
                     for niche / tissue-domain discovery (cf. DGI, GraphST, SpaceFlow).
* ``scvi``         - scvi-tools SCVI for count data, if installed.

All train with mini-batches (or sparse full-graph ops) so they scale to
millions of objects; on a laptop CPU keep epochs small or offload to a GPU node.
"""
from __future__ import annotations

import numpy as np

from ..data import feature_matrix
from ..pipeline.delta import Delta
from ..pipeline.registry import Param, op
from .reduce import _rep

P = Param
DEVICES = ["auto", "cpu", "cuda", "mps"]


def _setup(ctx, device, seed):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device != "auto":
        return torch.device(device)
    return ctx.torch_device()


def _batch_onehot(adata, batch_key):
    key = batch_key or adata.uns.get("batch_key")
    if not key or key not in adata.obs:
        return None, None
    codes = adata.obs[key].astype("category").cat.codes.to_numpy()
    oh = np.zeros((adata.n_obs, codes.max() + 1), np.float32)
    oh[np.arange(adata.n_obs), codes] = 1
    return oh, key


def _mlp(sizes, dropout=0.1, last_act=False):
    import torch.nn as nn
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers += [nn.LayerNorm(sizes[i + 1]), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


def _encode_all(fn, X, device, bs=65536):
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(X), bs):
            out.append(fn(torch.from_numpy(X[s:s + bs]).to(device)).float().cpu().numpy())
    return np.concatenate(out)


@op("vae", "Deep embedding", "Variational autoencoder (batch-aware)", params=[
    P("latent_dim", "int", 16), P("hidden", "int", 256), P("layers", "int", 2),
    P("epochs", "int", 30), P("batch_size", "int", 1024), P("lr", "float", 1e-3),
    P("beta", "float", 0.5, help="KL weight; lower = sharper reconstructions"),
    P("condition_on_batch", "bool", True, help="Condition encoder/decoder on batch to remove batch effects"),
    P("batch_key", "str", None), P("key", "str", "X_vae"),
    P("device", "choice", "auto", choices=DEVICES),
], heavy=True, gpu=True, requires=["torch"])
def vae(adata, ctx, latent_dim, hidden, layers, epochs, batch_size, lr, beta, condition_on_batch, batch_key, key, device):
    """Non-linear latent space; with batch conditioning, a deep alternative to ComBat/Harmony."""
    import torch
    import torch.nn as nn
    dev = _setup(ctx, device, ctx.seed)
    X = np.nan_to_num(feature_matrix(adata))
    mu_, sd_ = X.mean(0), X.std(0) + 1e-6
    X = ((X - mu_) / sd_).astype(np.float32)
    B, bkey = _batch_onehot(adata, batch_key) if condition_on_batch else (None, None)
    nb = 0 if B is None else B.shape[1]
    d = X.shape[1]

    class VAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = _mlp([d + nb] + [hidden] * layers, last_act=True)
            self.mu, self.lv = nn.Linear(hidden, latent_dim), nn.Linear(hidden, latent_dim)
            self.dec = nn.Sequential(_mlp([latent_dim + nb] + [hidden] * layers, last_act=True), nn.Linear(hidden, d))

        def encode(self, x, b):
            h = self.enc(torch.cat([x, b], 1) if nb else x)
            return self.mu(h), self.lv(h).clamp(-10, 10)

        def forward(self, x, b):
            m, lv = self.encode(x, b)
            z = m + torch.randn_like(m) * (0.5 * lv).exp()
            xh = self.dec(torch.cat([z, b], 1) if nb else z)
            return xh, m, lv

    model = VAE().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X)
    Bt = torch.from_numpy(B) if nb else torch.zeros(len(X), 0)
    n, losses = len(X), []
    steps_per_epoch = max(1, n // batch_size)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        kl_w = beta * min(1.0, (ep + 1) / max(1, epochs // 3))     # KL warm-up
        for s in range(steps_per_epoch):
            if ctx.should_cancel():
                raise RuntimeError("cancelled")
            idx = perm[s * batch_size:(s + 1) * batch_size]
            x, b = Xt[idx].to(dev), Bt[idx].to(dev)
            xh, m, lv = model(x, b)
            rec = ((xh - x) ** 2).sum(1).mean()
            kl = (-0.5 * (1 + lv - m ** 2 - lv.exp()).sum(1)).mean()
            loss = rec + kl_w * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        losses.append(tot / steps_per_epoch)
        ctx.progress(f"VAE epoch {ep + 1}/{epochs} loss {losses[-1]:.3f}", (ep + 1) / epochs)
    model.eval()
    XB = np.hstack([X, B]) if nb else X
    Z = _encode_all(lambda t: model.encode(t[:, :d], t[:, d:])[0], XB.astype(np.float32), dev)
    return Delta(obsm={key: Z.astype(np.float32)},
                 uns={f"{key}_training": {"loss": losses, "batch_key": bkey, "device": str(dev)}},
                 log=f"latent {latent_dim}d, final loss {losses[-1]:.3f} on {dev}")


@op("contrastive", "Deep embedding", "Contrastive self-supervised encoder", params=[
    P("latent_dim", "int", 32), P("hidden", "int", 256),
    P("epochs", "int", 20), P("batch_size", "int", 1024), P("lr", "float", 1e-3),
    P("mask_prob", "float", 0.3, help="Augmentation: fraction of features masked"),
    P("noise", "float", 0.2, help="Augmentation: Gaussian noise (in SD units)"),
    P("temperature", "float", 0.2), P("key", "str", "X_contrastive"),
    P("device", "choice", "auto", choices=DEVICES),
], heavy=True, gpu=True, requires=["torch"])
def contrastive(adata, ctx, latent_dim, hidden, epochs, batch_size, lr, mask_prob, noise, temperature, key, device):
    """Learn an embedding invariant to measurement noise / missing channels (NT-Xent loss)."""
    import torch
    import torch.nn.functional as F
    dev = _setup(ctx, device, ctx.seed)
    X = np.nan_to_num(feature_matrix(adata))
    X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
    d = X.shape[1]
    enc = _mlp([d, hidden, hidden, latent_dim]).to(dev)
    head = _mlp([latent_dim, hidden, 64]).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4)
    Xt = torch.from_numpy(X)

    def aug(x):
        keep = (torch.rand_like(x) > mask_prob).float()
        return x * keep + noise * torch.randn_like(x)

    n, losses = len(X), []
    spe = max(1, n // batch_size)
    for ep in range(epochs):
        perm, tot = torch.randperm(n), 0.0
        for s in range(spe):
            if ctx.should_cancel():
                raise RuntimeError("cancelled")
            x = Xt[perm[s * batch_size:(s + 1) * batch_size]].to(dev)
            z1 = F.normalize(head(enc(aug(x))), dim=1)
            z2 = F.normalize(head(enc(aug(x))), dim=1)
            z = torch.cat([z1, z2])
            sim = z @ z.T / temperature
            sim.fill_diagonal_(-1e9)
            m = len(x)
            target = torch.cat([torch.arange(m, 2 * m), torch.arange(0, m)]).to(dev)
            loss = F.cross_entropy(sim, target)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        losses.append(tot / spe)
        ctx.progress(f"contrastive epoch {ep + 1}/{epochs} loss {losses[-1]:.3f}", (ep + 1) / epochs)
    enc.eval()
    Z = _encode_all(enc, X, dev)
    return Delta(obsm={key: Z.astype(np.float32)}, uns={f"{key}_training": {"loss": losses, "device": str(dev)}},
                 log=f"latent {latent_dim}d, final loss {losses[-1]:.3f} on {dev}")


@op("spatial_dgi", "Deep embedding", "Spatial GNN (Deep Graph Infomax)", params=[
    P("latent_dim", "int", 32), P("hidden", "int", 128),
    P("epochs", "int", 200), P("lr", "float", 1e-3),
    P("use_rep", "str", "X_pca", help="Input node features (X = all features)"),
    P("graph", "choice", "spatial", choices=["spatial", "knn"]),
    P("key", "str", "X_spatial_dgi"),
    P("device", "choice", "auto", choices=DEVICES),
], heavy=True, gpu=True, requires=["torch"])
def spatial_dgi(adata, ctx, latent_dim, hidden, epochs, lr, use_rep, graph, key, device):
    """GCN encoder trained to tell real neighborhoods from shuffled ones; embeds microenvironment context.

    Cluster the result (k-means / Leiden on it) to get spatial domains / niches.
    """
    import scipy.sparse as sp
    import torch
    import torch.nn as nn
    dev = _setup(ctx, device, ctx.seed)
    gk = {"spatial": "spatial_connectivities", "knn": "connectivities"}[graph]
    if gk not in adata.obsp:
        raise ValueError(f"needs obsp['{gk}'] - add a graph step first")
    A = adata.obsp[gk].tocsr().astype(np.float32)
    A = A + sp.identity(A.shape[0], dtype=np.float32, format="csr")
    dinv = 1 / np.sqrt(np.asarray(A.sum(1)).ravel())
    A = sp.diags(dinv) @ A @ sp.diags(dinv)
    A = A.tocoo()
    At = torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data, A.shape).coalesce().to(dev)
    X = _rep(adata, use_rep)
    X = ((X - X.mean(0)) / (X.std(0) + 1e-6)).astype(np.float32)
    Xt = torch.from_numpy(X).to(dev)

    class GCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1, self.l2 = nn.Linear(X.shape[1], hidden), nn.Linear(hidden, latent_dim)
            self.act = nn.PReLU()

        def forward(self, x):
            h = self.act(torch.sparse.mm(At, self.l1(x)))
            return torch.sparse.mm(At, self.l2(h))

    enc = GCN().to(dev)
    disc = nn.Bilinear(latent_dim, latent_dim, 1).to(dev)
    opt = torch.optim.Adam(list(enc.parameters()) + list(disc.parameters()), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    losses = []
    for ep in range(epochs):
        if ctx.should_cancel():
            raise RuntimeError("cancelled")
        h_pos = enc(Xt)
        h_neg = enc(Xt[torch.randperm(len(Xt), device=dev)])
        summary = torch.sigmoid(torch.sparse.mm(At, h_pos))      # local (neighborhood) summary vectors
        lp, ln = disc(h_pos, summary), disc(h_neg, summary)
        loss = bce(lp, torch.ones_like(lp)) + bce(ln, torch.zeros_like(ln))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
        if ep % 10 == 0 or ep == epochs - 1:
            ctx.progress(f"DGI epoch {ep + 1}/{epochs} loss {loss.item():.3f}", (ep + 1) / epochs)
    enc.eval()
    with torch.no_grad():
        Z = enc(Xt).cpu().numpy()
    return Delta(obsm={key: Z.astype(np.float32)}, uns={f"{key}_training": {"loss": losses, "device": str(dev)}},
                 log=f"DGI {latent_dim}d on {graph} graph, final loss {losses[-1]:.3f} on {dev}")


@op("scvi", "Deep embedding", "scVI (count data, scvi-tools)", params=[
    P("n_latent", "int", 20), P("max_epochs", "int", 100),
    P("batch_key", "str", None), P("layer", "str", "raw", help="Layer holding raw counts"),
    P("key", "str", "X_scvi"),
], heavy=True, gpu=True, requires=["scvi-tools"])
def scvi(adata, ctx, n_latent, max_epochs, batch_key, layer, key):
    """Probabilistic count model with batch integration (Lopez et al. 2018). Requires raw integer counts."""
    try:
        import scvi as scvi_tools
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("pip install scvi-tools (best on a GPU node)") from e
    tmp = adata.copy()
    key_b = batch_key or adata.uns.get("batch_key")
    scvi_tools.settings.seed = ctx.seed
    scvi_tools.model.SCVI.setup_anndata(tmp, layer=layer, batch_key=key_b)
    m = scvi_tools.model.SCVI(tmp, n_latent=n_latent)
    m.train(max_epochs=max_epochs)
    return Delta(obsm={key: m.get_latent_representation().astype(np.float32)}, log=f"scVI {n_latent}d")
