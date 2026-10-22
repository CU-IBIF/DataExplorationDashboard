import base64

import numpy as np
from fastapi.testclient import TestClient

from featurescope.server.app import create_app


def test_server_flow(tmp_path):
    c = TestClient(create_app(tmp_path))
    st = c.post("/api/projects/create", json={"name": "demo", "demo": True, "n": 2000}).json()
    assert st["n_obs"] == 1998 or st["n_obs"] > 1000
    pipe = {"name": "p", "steps": [{"op": "transform", "params": {"method": "arcsinh", "cofactor": 150}},
                                   {"op": "normalize", "params": {}}, {"op": "pca", "params": {"n_comps": 5}},
                                   {"op": "kmeans", "params": {"n_clusters": 4}}]}
    assert c.put("/api/pipeline", json=pipe).status_code == 200
    jid = c.post("/api/pipeline/run", json={}).json()["job"]
    import time
    for _ in range(300):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            break
        time.sleep(0.1)
    assert j["status"] == "done", j
    st = c.get("/api/state").json()
    assert "kmeans" in st["columns"]["obs_categorical"] and all(p["status"] == "cached" for p in st["plan"])

    sc = c.post("/api/plot/scatter", json={"x": "obsm:X_pca:0", "y": "obsm:X_pca:1", "color": "obs:kmeans",
                                          "max_points": 500}).json()
    assert sc["n_shown"] == 500 and sc["color"]["kind"] == "cat"
    xs = np.frombuffer(base64.b64decode(sc["x"]["values"]), np.float32)
    assert len(xs) == 500

    # linked selection evaluated on ALL objects, returned as bitmask
    sel = c.post("/api/select", json={"x": "obsm:X_pca:0", "y": "obsm:X_pca:1", "type": "rect",
                                      "xrange": [0, 100], "yrange": [-100, 100]}).json()
    mask = np.unpackbits(np.frombuffer(base64.b64decode(sel["mask"]), np.uint8), bitorder="little")[:st["n_obs"]]
    assert mask.sum() == sel["count"] > 0
    assert len(c.get("/api/selection/stats").json()) == st["n_vars"]
    h = c.post("/api/plot/hist", json={"x": "feat:CD3"}).json()
    assert sum(h["selected"]) <= sel["count"]

    g = c.post("/api/gates", json={"name": "pc1pos", "x": "obsm:X_pca:0", "y": "obsm:X_pca:1", "type": "rect",
                                   "xrange": [0, 100], "yrange": [-100, 100]}).json()
    assert g["gate_stats"][0]["count"] == sel["count"]
    assert c.post("/api/plot/heatmap", json={"groupby": "obs:kmeans"}).status_code == 200
    assert c.post("/api/plot/groups", json={"y": "feat:CD3", "groupby": "obs:kmeans"}).status_code == 200
    lab = c.post("/api/selection/label", json={"column": "manual", "label": "pos"}).json()
    assert "manual" in lab["columns"]["obs_categorical"]
    assert c.post("/api/pipelines/save", json={"name": "p"}).json()["saved"] == "p"
    # reopen -> results restored from cache with no compute
    st2 = c.post("/api/projects/open", json={"name": "demo"}).json()
    assert "kmeans" in st2["columns"]["obs_categorical"] and "manual" in st2["columns"]["obs_categorical"]
    assert c.get("/api/download/table").status_code == 200
    assert c.get("/").status_code == 200
