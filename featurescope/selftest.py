"""End-to-end check of an installation or a frozen (PyInstaller) build.

``featurescope selftest`` creates a small synthetic project, runs a pipeline that
touches every compiled dependency (numba/pynndescent/UMAP, igraph/leidenalg,
scikit-learn, h5py/anndata, PyTorch when present), reloads it from cache,
starts the web server and requests the UI and API.  Exits non-zero on failure.
"""
from __future__ import annotations

import json
import socket
import tempfile
import threading
import time
import urllib.request
from pathlib import Path


def run(keep: str | None = None) -> int:
    from . import Pipeline, Project
    from .synthetic import make_dataset

    t0 = time.time()
    root = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="featurescope-selftest-"))
    print(f"[selftest] workspace {root}")
    p = Project.create(root / "selftest", make_dataset(1500, n_slides=2), overwrite=True)
    pl = Pipeline("selftest")
    for op, kw in [("transform", {"method": "arcsinh", "cofactor": 150}), ("batch_center", {}),
                   ("normalize", {"max_value": 5}), ("pca", {"n_comps": 8}), ("harmony", {}),
                   ("neighbors", {"use_rep": "X_harmony", "backend": "cpu"}), ("umap", {"backend": "cpu"}),
                   ("leiden", {"resolution": 0.5, "backend": "cpu"}), ("flowsom", {"grid": 4, "n_meta": 5}),
                   ("markers", {"groupby": "leiden"}), ("spatial_neighbors", {}),
                   ("nhood_enrichment", {"n_perms": 10}), ("cellular_neighborhoods", {"n_neighborhoods": 3})]:
        pl.add(op, **kw)
    try:
        import torch  # noqa: F401
        pl.add("vae", epochs=1, latent_dim=4, device="cpu")
        pl.add("spatial_dgi", epochs=5, device="cpu")
        print(f"[selftest] torch {torch.__version__} found: including deep-learning steps")
    except ImportError:
        print("[selftest] torch not installed: skipping deep-learning steps")
    p.save_pipeline(pl)
    adata, rep = p.run(pl)
    for r in rep:
        print(f"[selftest]   {r['status']:9s} {r['op']:22s} {r.get('seconds', 0):6.2f}s {r.get('log', '')}")
    _, rep2 = Project(p.root).run(pl, cache_only=True)
    assert all(r["status"] == "cached" for r in rep2), "cache reload failed"
    p.export(adata, "selftest")

    # web server
    import uvicorn
    from .server.app import create_app
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(root), host="127.0.0.1", port=port, log_level="warning",
                                           loop="asyncio", http="h11", lifespan="off"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(base + "/api/state", timeout=2)
            break
        except OSError:
            time.sleep(0.1)
    html = urllib.request.urlopen(base + "/", timeout=10).read().decode()
    assert "FeatureScope" in html, "UI not served"
    assert urllib.request.urlopen(base + "/vendor/plotly.min.js", timeout=10).status == 200, "plotly not bundled"
    req = urllib.request.Request(base + "/api/projects/open", data=json.dumps({"name": "selftest"}).encode(),
                                 headers={"Content-Type": "application/json"})
    st = json.loads(urllib.request.urlopen(req, timeout=60).read())
    assert "leiden" in st["columns"]["obs_categorical"], "results not restored"
    server.should_exit = True
    th.join(timeout=10)
    print(f"[selftest] OK in {time.time() - t0:.1f}s")
    return 0


def free_port(start: int = 8765) -> int:
    for port in list(range(start, start + 50)) + [0]:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")
