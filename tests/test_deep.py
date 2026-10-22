import numpy as np
import pytest

from featurescope import Pipeline

torch = pytest.importorskip("torch")


def test_deep_embeddings(project):
    pl = Pipeline("deep")
    for op, kw in [("transform", {"method": "arcsinh", "cofactor": 150}), ("batch_center", {}),
                   ("normalize", {"method": "zscore", "max_value": 5}), ("pca", {"n_comps": 10}),
                   ("vae", {"epochs": 3, "latent_dim": 4, "device": "cpu"}),
                   ("contrastive", {"epochs": 3, "latent_dim": 8, "device": "cpu"}),
                   ("spatial_neighbors", {}), ("spatial_dgi", {"epochs": 20, "device": "cpu"})]:
        pl.add(op, **kw)
    a, rep = project.run(pl)
    assert a.obsm["X_vae"].shape == (a.n_obs, 4)
    assert a.obsm["X_contrastive"].shape == (a.n_obs, 8)
    assert a.obsm["X_spatial_dgi"].shape[0] == a.n_obs
    assert all(np.isfinite(a.obsm[k]).all() for k in ("X_vae", "X_contrastive", "X_spatial_dgi"))
    # deterministic given the seed -> cached result equals a forced recompute
    b, _ = project.run(pl, force=True)
    assert np.allclose(a.obsm["X_vae"], b.obsm["X_vae"], atol=1e-4)
