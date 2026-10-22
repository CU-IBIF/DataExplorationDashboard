import json

import numpy as np
import pytest

from featurescope import Pipeline, Project
from featurescope.data import detect_roles, get_vector, import_table
from featurescope.gating import evaluate_gate, gate_stats
from featurescope.pipeline.delta import Delta
from featurescope.pipeline.engine import chain_keys
from featurescope.pipeline.registry import all_ops


def test_detect_roles(df):
    r = detect_roles(df)
    assert r["spatial"] == ["x_centroid", "y_centroid"]
    assert r["batch"] == "slide" and r["id"] == "cell_id"
    assert "CD3" in r["features"] and "area" not in r["features"]  # morphology -> metadata


def test_import_and_vectors(df):
    a = import_table(df)
    assert a.n_obs == len(df) and "spatial" in a.obsm and "raw" in a.layers
    assert np.allclose(get_vector(a, "feat:CD3"), df["CD3"].to_numpy(np.float32))
    assert get_vector(a, "obsm:spatial:0").shape == (len(df),)
    assert get_vector(a, "obs:slide").categories.tolist() == ["slide_1", "slide_2", "slide_3"]


def test_delta_roundtrip(tmp_path):
    import pandas as pd
    import scipy.sparse as sp
    d = Delta(X=np.ones((4, 2), np.float32), obs={"c": pd.Categorical(list("abab"))}, obsm={"X_e": np.eye(4)},
              obsp={"g": sp.eye(4, format="csr")}, uns={"t": pd.DataFrame({"a": [1]})}, obs_mask=np.array([1, 1, 1, 1], bool))
    d.save(tmp_path / "d")
    e = Delta.load(tmp_path / "d")
    assert np.array_equal(e.X, d.X) and list(e.obs["c"]) == list("abab")
    assert (e.obsp["g"] != d.obsp["g"]).nnz == 0 and e.uns["t"].equals(d.uns["t"])


def test_pipeline_cache_and_invalidation(project):
    pl = Pipeline("p").add("transform", method="arcsinh", cofactor=150).add("normalize").add("pca", n_comps=5)
    pl.add("kmeans", n_clusters=4, use_rep="X_pca")
    a1, r1 = project.run(pl)
    assert [r["status"] for r in r1] == ["computed"] * 4
    a2, r2 = project.run(pl)
    assert [r["status"] for r in r2] == ["cached"] * 4
    assert np.array_equal(a1.obsm["X_pca"], a2.obsm["X_pca"])
    assert (a1.obs["kmeans"] == a2.obs["kmeans"]).all()
    # editing step 3 invalidates only steps 3..4
    pl.steps[2].params["n_comps"] = 6
    _, r3 = project.run(pl)
    assert [r["status"] for r in r3] == ["cached", "cached", "computed", "computed"]
    # reopening the project (new process) restores everything from disk without compute
    p2 = Project(project.root)
    _, r4 = p2.run(pl, cache_only=True)
    assert all(r["status"] == "cached" for r in r4)


def test_pipeline_serialization(tmp_path):
    pl = Pipeline("x").add("leiden", resolution=0.5)
    for ext in ("json", "yaml"):
        f = pl.save(tmp_path / f"p.{ext}")
        q = Pipeline.load(f)
        assert q.steps[0].op == "leiden" and q.steps[0].params == {"resolution": 0.5}
        assert chain_keys(q, "b") == chain_keys(pl, "b")


def test_unknown_param_rejected(project):
    with pytest.raises(ValueError):
        project.run(Pipeline("bad").add("pca", n_components=3))


def test_gates(project):
    a = project.base()
    gates = {
        "big": {"name": "big", "x": "obs:area", "type": "range", "range": [100, 1e9]},
        "cd3": {"name": "cd3", "parent": "big", "x": "raw:CD3", "y": "raw:CD45", "type": "rect",
                "xrange": [500, 1e9], "yrange": [0, 1e9]},
        "either": {"name": "either", "type": "boolean", "expr": "or", "of": ["big", "cd3"]},
    }
    big = evaluate_gate(a, gates, "big")
    cd3 = evaluate_gate(a, gates, "cd3")
    assert cd3.sum() > 0 and not (cd3 & ~big).any()          # child within parent
    assert (evaluate_gate(a, gates, "either") == big).all()
    stats = {s["name"]: s for s in gate_stats(a, gates)}
    assert stats["cd3"]["count"] == cd3.sum()


def test_subset_gate_step_depends_on_gate_geometry(project):
    gates = {"g": {"name": "g", "x": "raw:CD3", "type": "range", "range": [0, 300]}}
    project.save_gates(gates)
    pl = Pipeline("s").add("subset_gate", gate="g")
    a, _ = project.run(pl)
    gates["g"]["range"] = [0, 200]
    project.save_gates(gates)
    b, rep = project.run(pl)
    assert rep[0]["status"] == "computed" and b.n_obs < a.n_obs


def test_full_analysis(project):
    pl = Pipeline("full")
    for op, kw in [("filter_objects", {"column": "obs:area", "quantile_high": 0.99}),
                   ("transform", {"method": "arcsinh", "cofactor": 150}), ("clip", {}),
                   ("batch_center", {}), ("normalize", {"method": "zscore", "max_value": 5}),
                   ("pca", {"n_comps": 10}), ("harmony", {}), ("neighbors", {"use_rep": "X_harmony"}),
                   ("umap", {}), ("leiden", {"resolution": 0.5}), ("flowsom", {"grid": 5, "n_meta": 6}),
                   ("markers", {"groupby": "leiden"}), ("spatial_neighbors", {}),
                   ("nhood_enrichment", {"n_perms": 20}), ("cellular_neighborhoods", {"n_neighborhoods": 3}),
                   ("neighbor_aggregate", {}), ("cluster_connectivity", {}), ("spatial_autocorr", {})]:
        pl.add(op, **kw)
    a, rep = project.run(pl)
    from sklearn.metrics import adjusted_rand_score as ari
    assert ari(a.obs["true_cell_type"], a.obs["flowsom"]) > 0.8
    assert ari(a.obs["true_domain"], a.obs["neighborhood"]) > 0.7
    assert a.uns["nhood_enrichment_leiden"]["zscore"].shape[0] == a.obs["leiden"].nunique()
    out = project.export(a, "full")
    import anndata as ad
    assert ad.read_h5ad(out).n_obs == a.n_obs


def test_annotations_persist(project):
    import pandas as pd
    a = project.base()
    project.save_annotation("manual", pd.Series(["T"] * 10, index=a.obs_names[:10]))
    b, _ = project.run(Pipeline("e").add("normalize"))
    assert (b.obs["manual"][:10] == "T").all() and (b.obs["manual"][10:] == "unlabeled").all()


def test_all_ops_have_schema():
    ops = all_ops()
    names = {o["name"] for o in ops}
    for need in ("combat", "harmony", "vae", "contrastive", "spatial_dgi", "leiden", "umap", "nhood_enrichment"):
        assert need in names
    json.dumps(ops)
