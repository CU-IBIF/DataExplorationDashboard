"""FastAPI backend for the dashboard.

Big-data design: the browser never receives the full table.  Plots get a stable
random display sample (same objects in every view, so linked highlighting is
consistent) as base64 float32 arrays; selections and gates are evaluated on
*all* objects server-side and returned as a packed bitmask (1M objects = 125 kB).
"""
from __future__ import annotations

import base64
import io
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..compute import executors as cx
from ..data import detect_roles, get_vector, list_columns, read_any
from ..gating import evaluate_gate, gate_stats, region_mask
from ..pipeline import engine
from ..pipeline.registry import all_ops, get_op
from ..pipeline.spec import Pipeline
from ..project import Project

WEB = Path(__file__).resolve().parent.parent / "web"


def b64(a: np.ndarray, dtype=np.float32) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def pack_mask(m: np.ndarray) -> str:
    return base64.b64encode(np.packbits(m.astype(bool), bitorder="little").tobytes()).decode()


def _clean(v):
    """JSON-safe conversion of numpy/pandas scalars and containers."""
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return _clean(v.tolist())
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class Session:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        (self.workspace / "uploads").mkdir(parents=True, exist_ok=True)
        self.project: Project | None = None
        self.pipeline = Pipeline("untitled")
        self.result = None
        self.report: list = []
        self.selection: np.ndarray | None = None
        self.order: np.ndarray | None = None
        self.jobs: dict[str, dict] = {}
        self.lock = threading.RLock()

    # ------------------------------------------------------------------ helpers
    def need(self):
        if self.project is None or self.result is None:
            raise HTTPException(400, "no project open")
        return self.result

    def set_result(self, adata, report=None):
        old = self.result
        self.result = adata
        if report is not None:
            self.report = report
        self.order = np.random.default_rng(0).permutation(adata.n_obs)
        if self.selection is not None and old is not None:   # carry the selection across reruns by object id
            sel_ids = old.obs_names[self.selection]
            self.selection = adata.obs_names.isin(sel_ids)
            if not self.selection.any():
                self.selection = None

    def open(self, root: Path):
        self.project = Project(root)
        self.selection = None
        self.result = None
        names = self.project.list_pipelines()
        self.pipeline = self.project.load_pipeline(names[0]) if names else Pipeline("untitled")
        self.load_cached()

    def load_cached(self):
        """Apply every already-cached step of the working pipeline (no computation)."""
        a, rep = self.project.run(self.pipeline, cache_only=True)
        self.set_result(a, rep)

    def sample(self, max_points: int) -> np.ndarray:
        return self.order[: min(max_points, len(self.order))]


def create_app(workspace: str | Path = "projects") -> FastAPI:
    app = FastAPI(title="FeatureScope", version=__version__)
    S = Session(Path(workspace))
    app.state.session = S

    @app.exception_handler(Exception)
    async def _err(request, exc):  # surface analysis errors to the UI
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    # ================================================================== projects / import
    @app.get("/api/ops")
    def ops():
        return all_ops()

    @app.get("/api/projects")
    def projects():
        out = []
        for p in sorted(S.workspace.glob("*/project.json")):
            try:
                out.append(Project(p.parent).info())
            except Exception:
                pass
        return out

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        uid = uuid.uuid4().hex[:10]
        dest = S.workspace / "uploads" / f"{uid}_{Path(file.filename).name}"
        with open(dest, "wb") as fh:
            while chunk := await file.read(1 << 22):
                fh.write(chunk)
        return preview_path(str(dest))

    @app.post("/api/preview")
    def preview(path: str = Body(..., embed=True)):
        return preview_path(path)

    def preview_path(path: str):
        p = Path(path).expanduser()
        if not p.exists():
            raise HTTPException(404, f"{p} not found")
        if p.suffix in (".h5ad", ".zarr"):
            return {"path": str(p), "anndata": True, "columns": [], "roles": {}, "head": []}
        nrows = {"nrows": 2000} if p.suffix not in (".parquet", ".feather", ".pq", ".arrow") else {}
        df = read_any(p, **nrows)
        roles = detect_roles(df)
        return {"path": str(p), "anndata": False, "columns": list(map(str, df.columns)), "roles": roles,
                "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
                "head": _clean(df.head(8).astype(object).where(df.head(8).notna(), None).to_dict("records"))}

    @app.post("/api/projects/create")
    def create(body: dict = Body(...)):
        name = body["name"].strip()
        root = S.workspace / "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        kw = {k: body.get(k) for k in ("features", "metadata", "spatial", "id_col", "batch") if body.get(k)}
        if body.get("demo"):
            from ..synthetic import make_dataset
            src = make_dataset(int(body.get("n", 20000)))
        else:
            src = body["path"]
        with S.lock:
            Project.create(root, src, name=name, overwrite=bool(body.get("overwrite")), **kw)
            S.open(root)
        return state()

    @app.post("/api/projects/open")
    def open_project(name: str = Body(..., embed=True)):
        with S.lock:
            S.open(S.workspace / name)
        return state()

    # ================================================================== state
    @app.get("/api/state")
    def state():
        if S.project is None:
            return {"project": None, "version": __version__}
        a = S.result
        gates = S.project.gates()
        return _clean({
            "version": __version__,
            "project": S.project.info(),
            "n_obs": a.n_obs, "n_vars": a.n_vars,
            "columns": list_columns(a),
            "categories": {c: list(map(str, a.obs[c].astype("category").cat.categories))[:200]
                           for c in list_columns(a)["obs_categorical"]},
            "batch_key": a.uns.get("batch_key"),
            "pipeline": S.pipeline.to_dict(),
            "plan": S.project.plan(S.pipeline),
            "report": S.report,
            "gates": gates, "gate_stats": gate_stats(a, gates),
            "selection": {"count": int(S.selection.sum()) if S.selection is not None else 0},
            "results": _result_keys(a),
            "layers": list(a.layers.keys()),
        })

    def _result_keys(a):
        out = []
        for k, v in a.uns.items():
            if k.startswith(("nhood_enrichment_", "markers_", "paga_")) or k.endswith("_profiles") or k in ("morans_i", "pca"):
                out.append(k)
            elif k.endswith("_training"):
                out.append(k)
        return out

    # ================================================================== pipeline
    @app.put("/api/pipeline")
    def set_pipeline(body: dict = Body(...)):
        p = Pipeline.from_dict(body)
        for s in p.steps:
            get_op(s.op).resolve(s.params)   # validate
        with S.lock:
            S.pipeline = p
        return {"plan": S.project.plan(p) if S.project else []}

    @app.get("/api/pipelines")
    def pipelines():
        return S.project.list_pipelines() if S.project else []

    @app.post("/api/pipelines/save")
    def save_pipeline(name: str = Body(..., embed=True)):
        S.pipeline.name = name
        S.project.save_pipeline(S.pipeline)
        return {"saved": name, "pipelines": S.project.list_pipelines()}

    @app.post("/api/pipelines/load")
    def load_pipeline(name: str = Body(..., embed=True)):
        with S.lock:
            S.pipeline = S.project.load_pipeline(name)
            S.load_cached()
        return state()

    @app.delete("/api/pipelines/{name}")
    def delete_pipeline(name: str):
        S.project.delete_pipeline(name)
        return S.project.list_pipelines()

    @app.post("/api/pipeline/run")
    def run_pipeline(body: dict = Body(default={})):
        S.need()
        jid = uuid.uuid4().hex[:8]
        job = {"id": jid, "status": "running", "progress": 0.0, "message": "starting", "error": None,
               "cancel": False, "started": time.time()}
        S.jobs[jid] = job
        pipe = Pipeline.from_dict(S.pipeline.to_dict())

        def progress(msg, frac):
            job["message"], job["progress"] = msg, frac

        def work():
            try:
                ctx = engine.Context(progress=progress, should_cancel=lambda: job["cancel"],
                                     device=body.get("device", "auto"))
                a, rep = S.project.run(pipe, ctx, upto=body.get("upto"), force=bool(body.get("force")))
                with S.lock:
                    S.set_result(a, rep)
                job["status"] = "done"
            except Exception as e:
                job["status"], job["error"] = ("cancelled" if job["cancel"] else "error"), f"{type(e).__name__}: {e}"
                job["trace"] = traceback.format_exc()[-3000:]
            job["elapsed"] = time.time() - job["started"]

        threading.Thread(target=work, daemon=True).start()
        return {"job": jid}

    @app.get("/api/jobs/{jid}")
    def job_status(jid: str):
        j = S.jobs.get(jid) or {}
        return {k: v for k, v in j.items() if k != "cancel"}

    @app.post("/api/jobs/{jid}/cancel")
    def job_cancel(jid: str):
        S.jobs[jid]["cancel"] = True
        return {"ok": True}

    # ================================================================== plotting data
    def _vec_payload(a, spec, idx):
        v = get_vector(a, spec)
        if isinstance(v, pd.Categorical):
            return {"kind": "cat", "codes": b64(v.codes[idx], np.int16), "categories": list(map(str, v.categories))}
        v = np.asarray(v, float)[idx]
        fin = v[np.isfinite(v)]
        lo, hi = (np.quantile(fin, [0.01, 0.99]) if len(fin) else (0, 1))
        return {"kind": "num", "values": b64(v), "p1": float(lo), "p99": float(hi)}

    @app.post("/api/plot/scatter")
    def scatter(body: dict = Body(...)):
        a = S.need()
        idx = S.sample(int(body.get("max_points", 100_000)))
        if body.get("within_gate"):
            m = evaluate_gate(a, S.project.gates(), body["within_gate"])
            idx = idx[m[idx]]
        if body.get("filter"):
            f = body["filter"]
            v = np.asarray(a.obs[f["column"]].astype(str))
            idx = idx[v[idx] == str(f["value"])]
        out = {"n_total": a.n_obs, "n_shown": len(idx), "idx": b64(idx, np.int32),
               "x": _vec_payload(a, body["x"], idx), "y": _vec_payload(a, body["y"], idx)}
        if body.get("color"):
            out["color"] = _vec_payload(a, body["color"], idx)
        return out

    @app.post("/api/plot/hist")
    def hist(body: dict = Body(...)):
        a = S.need()
        v = np.asarray(get_vector(a, body["x"]), float)
        fin = np.isfinite(v)
        lo, hi = np.quantile(v[fin], [0.001, 0.999]) if body.get("trim", True) else (v[fin].min(), v[fin].max())
        edges = np.linspace(lo, hi if hi > lo else lo + 1, int(body.get("bins", 80)) + 1)
        out = {"edges": edges.tolist(), "all": np.histogram(v[fin], edges)[0].tolist()}
        if S.selection is not None:
            out["selected"] = np.histogram(v[fin & S.selection], edges)[0].tolist()
        return out

    @app.post("/api/plot/groups")
    def groups(body: dict = Body(...)):
        """Box-plot statistics of a feature per group: scales to any number of objects."""
        a = S.need()
        v = np.asarray(get_vector(a, body["y"]), float)
        g = pd.Categorical(get_vector(a, body["groupby"]))
        out = []
        for i, c in enumerate(g.categories):
            x = v[(g.codes == i) & np.isfinite(v)]
            if len(x) == 0:
                continue
            q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
            row = {"group": str(c), "n": int(len(x)), "q": q.tolist(), "mean": float(x.mean())}
            if S.selection is not None:
                xs = v[(g.codes == i) & np.isfinite(v) & S.selection]
                row["n_sel"] = int(len(xs))
            out.append(row)
        return out

    @app.post("/api/plot/heatmap")
    def heatmap(body: dict = Body(...)):
        a = S.need()
        from ..data import feature_matrix
        g = pd.Categorical(get_vector(a, body["groupby"]))
        feats = body.get("features") or list(a.var_names)
        X = feature_matrix(a, body.get("layer") or None, feats)
        df = pd.DataFrame(X, columns=feats).groupby(np.asarray(g), observed=True).mean()
        if body.get("scale", "zscore") == "zscore":
            df = (df - df.mean()) / df.std(ddof=0).replace(0, 1)
        elif body.get("scale") == "minmax":
            df = (df - df.min()) / (df.max() - df.min()).replace(0, 1)
        counts = pd.Series(np.asarray(g)).value_counts()
        # order rows/cols by hierarchical clustering for readability
        if body.get("cluster", True) and df.shape[0] > 2 and df.shape[1] > 2:
            from scipy.cluster.hierarchy import leaves_list, linkage
            df = df.iloc[leaves_list(linkage(df.fillna(0).values, "average")),
                         leaves_list(linkage(df.fillna(0).values.T, "average"))]
        return _clean({"rows": list(map(str, df.index)), "cols": list(map(str, df.columns)), "z": df.values,
                       "counts": [int(counts.get(r, 0)) for r in df.index]})

    @app.get("/api/results/{key}")
    def result(key: str):
        a = S.need()
        v = a.uns[key]
        if isinstance(v, dict):
            for sub in ("zscore", "table", "group_means"):
                if sub in v:
                    v = v[sub] if sub != "table" else v[sub]
                    break
            else:
                if "loss" in v:
                    return {"kind": "curve", "y": _clean(v["loss"])}
                if "variance_ratio" in v:
                    return {"kind": "curve", "y": _clean(v["variance_ratio"])}
        if isinstance(v, pd.DataFrame):
            if key.startswith("markers_"):
                return {"kind": "table", "columns": list(map(str, v.columns)),
                        "rows": _clean(v.head(2000).astype(object).where(v.head(2000).notna(), None).values.tolist())}
            numeric = v.select_dtypes("number")
            if numeric.shape[1] == 1:
                return {"kind": "bar", "x": list(map(str, numeric.index)), "y": _clean(numeric.iloc[:, 0].values)}
            return {"kind": "matrix", "rows": list(map(str, v.index)), "cols": list(map(str, v.columns)),
                    "z": _clean(numeric.values)}
        return {"kind": "json", "value": _clean(v) if not hasattr(v, "shape") else f"array {v.shape}"}

    # ================================================================== linked selection
    def _region_from(body):
        r = {"x": body["x"], "y": body.get("y"), "type": body.get("type", "polygon")}
        for k in ("vertices", "range", "xrange", "yrange"):
            if k in body:
                r[k] = body[k]
        return r

    def _apply_selection(m, mode):
        if mode == "add" and S.selection is not None:
            m = S.selection | m
        elif mode == "intersect" and S.selection is not None:
            m = S.selection & m
        elif mode == "subtract" and S.selection is not None:
            m = S.selection & ~m
        S.selection = m if m.any() else None
        return selection_payload()

    def selection_payload():
        if S.selection is None:
            return {"count": 0, "mask": None}
        return {"count": int(S.selection.sum()), "mask": pack_mask(S.selection)}

    @app.post("/api/select")
    def select(body: dict = Body(...)):
        a = S.need()
        m = region_mask(a, _region_from(body))
        if body.get("within_gate"):
            m &= evaluate_gate(a, S.project.gates(), body["within_gate"])
        return _apply_selection(m, body.get("mode", "replace"))

    @app.post("/api/select/category")
    def select_category(body: dict = Body(...)):
        a = S.need()
        v = get_vector(a, body["column"])
        m = np.isin(np.asarray(v).astype(str), [str(x) for x in body["values"]])
        return _apply_selection(m, body.get("mode", "replace"))

    @app.post("/api/select/gate")
    def select_gate(name: str = Body(..., embed=True)):
        a = S.need()
        return _apply_selection(evaluate_gate(a, S.project.gates(), name), "replace")

    @app.post("/api/select/clear")
    def select_clear():
        S.selection = None
        return selection_payload()

    @app.get("/api/selection")
    def get_selection():
        return selection_payload()

    @app.get("/api/selection/stats")
    def selection_stats():
        a = S.need()
        if S.selection is None:
            return []
        from ..data import feature_matrix
        X = feature_matrix(a)
        s, r = X[S.selection], X[~S.selection]
        mu_s, mu_r = np.nanmean(s, 0), np.nanmean(r, 0) if len(r) else np.zeros(X.shape[1])
        sd = np.sqrt((np.nanvar(s, 0) + (np.nanvar(r, 0) if len(r) else 0)) / 2) + 1e-9
        d = (mu_s - mu_r) / sd
        order = np.argsort(-np.abs(d))
        return _clean([{"feature": a.var_names[i], "mean_sel": mu_s[i], "mean_rest": mu_r[i], "effect": d[i]}
                       for i in order])

    @app.post("/api/selection/label")
    def selection_label(body: dict = Body(...)):
        """Write the selection into an annotation column (manual cell typing)."""
        a = S.need()
        col, label = body["column"].strip(), body["label"].strip()
        if S.selection is None or not col or not label:
            raise HTTPException(400, "need a selection, a column and a label")
        cur = a.obs[col].astype(str).to_numpy().astype(object) if col in a.obs else np.full(a.n_obs, "unlabeled", object)
        cur[S.selection] = label
        a.obs[col] = pd.Categorical(cur)
        S.project.save_annotation(col, pd.Series(cur, index=a.obs_names))
        return state()

    # ================================================================== gates
    @app.post("/api/gates")
    def add_gate(body: dict = Body(...)):
        a = S.need()
        gates = S.project.gates()
        name = body["name"].strip()
        g = _region_from(body)
        g.update({"name": name, "parent": body.get("parent") or None, "color": body.get("color")})
        if body.get("type") == "boolean":
            g = {"name": name, "type": "boolean", "expr": body.get("expr", "and"), "of": body["of"],
                 "parent": body.get("parent")}
        gates[name] = g
        evaluate_gate(a, gates, name)   # validate before persisting
        S.project.save_gates(gates)
        return {"gates": gates, "gate_stats": gate_stats(a, gates)}

    @app.delete("/api/gates/{name}")
    def del_gate(name: str):
        a = S.need()
        gates = S.project.gates()
        gates.pop(name, None)
        for g in gates.values():          # re-parent children
            if g.get("parent") == name:
                g["parent"] = None
        S.project.save_gates(gates)
        return {"gates": gates, "gate_stats": gate_stats(a, gates)}

    # ================================================================== export
    @app.post("/api/export")
    def export(name: str = Body(..., embed=True)):
        a = S.need()
        from ..gating import gates_to_obs
        b = a.copy()
        for k, v in gates_to_obs(a, S.project.gates()).items():
            b.obs[k] = v
        if S.selection is not None:
            b.obs["selected"] = S.selection
        return {"path": str(S.project.export(b, name))}

    @app.get("/api/download/table")
    def download_table(selection_only: bool = False, embeddings: bool = True):
        a = S.need()
        from ..gating import gates_to_obs
        df = a.obs.copy()
        df.insert(0, "object_id", a.obs_names)
        for k, v in gates_to_obs(a, S.project.gates()).items():
            df[k] = v
        if embeddings:
            for k in a.obsm:
                if k.startswith("X_") and a.obsm[k].shape[1] <= 3:
                    for d in range(a.obsm[k].shape[1]):
                        df[f"{k}_{d + 1}"] = np.asarray(a.obsm[k])[:, d]
        if selection_only and S.selection is not None:
            df = df[S.selection]
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                                 headers={"Content-Disposition": "attachment; filename=featurescope_objects.csv"})

    @app.get("/api/download/pipeline")
    def download_pipeline():
        import json
        return JSONResponse(S.pipeline.to_dict(), headers={
            "Content-Disposition": f"attachment; filename={S.pipeline.name}.json"})

    # ================================================================== compute offload
    @app.get("/api/compute/profiles")
    def compute_profiles():
        a = S.need()
        plan = S.project.plan(S.pipeline)
        ops = {o["name"]: o for o in all_ops()}
        pending = [ops[p["op"]] for p in plan if p["status"] == "pending"]
        profs = cx.load_profiles(S.project.root)
        return {"profiles": {k: {kk: vv for kk, vv in v.items() if kk not in ("ssh_opts",)} for k, v in profs.items()},
                "recommendation": cx.recommend(a.n_obs, a.n_vars, pending)}

    @app.post("/api/compute/submit")
    def compute_submit(body: dict = Body(...)):
        S.need()
        S.project.save_pipeline(S.pipeline)
        ex = cx.get_executor(body["profile"], S.project.root, dry_run=bool(body.get("dry_run")))
        job = ex.submit(S.pipeline.to_dict())
        return _clean(job.__dict__)

    @app.get("/api/compute/jobs")
    def compute_jobs():
        if S.project is None:
            return []
        return [_clean(j.__dict__) for j in cx.Job.list(S.project.root)]

    @app.post("/api/compute/jobs/{jid}/refresh")
    def compute_refresh(jid: str):
        job = cx.Job.load(S.project.root, jid)
        ex = cx.get_executor(job.profile, S.project.root, dry_run=job.dry_run)
        if job.status != "fetched":
            st = ex.status(job)
            job.status = st
            job.log(f"status {st}")
            if st == "COMPLETED":
                ex.fetch(job)
                job.status = "fetched"
                job.log("results fetched into cache")
                with S.lock:
                    S.load_cached()
        job.save(S.project.root)
        return _clean(job.__dict__)

    @app.post("/api/compute/jobs/{jid}/cancel")
    def compute_cancel(jid: str):
        job = cx.Job.load(S.project.root, jid)
        cx.get_executor(job.profile, S.project.root, dry_run=job.dry_run).cancel(job)
        job.status = "CANCELLED"
        job.save(S.project.root)
        return _clean(job.__dict__)

    @app.get("/api/compute/jobs/{jid}/file/{name}")
    def compute_file(jid: str, name: str):
        p = (S.project.root / "jobs" / jid / name).resolve()
        if not str(p).startswith(str((S.project.root / "jobs").resolve())) or not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="text/plain")

    # ================================================================== static UI
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
    return app
