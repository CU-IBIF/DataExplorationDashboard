"""On-disk project: imported data, pipelines, gates and the result cache.

    my_project/
      project.json          name, dataset fingerprint, column roles, created
      data/base.h5ad        imported dataset (never modified)
      pipelines/*.json      saved pipelines
      gates.json            gating hierarchy
      cache/<key>/          one directory per computed step (content-addressed)
      results/*.h5ad        exported snapshots (for scanpy/squidpy/R)
      jobs/<id>/            remote job manifests and logs

A project directory is self-contained: copy (rsync / Globus / S3 sync) it to a
cluster, run ``featurescope run``, copy ``cache/`` back, and the dashboard
loads every result without recomputation.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import anndata as ad

from .data import import_table
from .pipeline import engine
from .pipeline.spec import Pipeline


def fingerprint(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()[:24]


class Project:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        meta = self.root / "project.json"
        if not meta.exists():
            raise FileNotFoundError(f"{self.root} is not a FeatureScope project (no project.json)")
        self.meta = json.loads(meta.read_text(encoding="utf-8"))
        self.cache = engine.Cache(self.root / "cache")
        self._base: ad.AnnData | None = None

    # ----------------------------------------------------------------- creation
    @classmethod
    def create(cls, root: str | Path, source, name: str | None = None, overwrite: bool = False, **import_kw) -> "Project":
        root = Path(root)
        if (root / "project.json").exists() and not overwrite:
            raise FileExistsError(f"{root} already contains a project")
        for d in ("data", "pipelines", "cache", "results", "jobs"):
            (root / d).mkdir(parents=True, exist_ok=True)
        adata = source if isinstance(source, ad.AnnData) else import_table(source, **import_kw)
        base = root / "data" / "base.h5ad"
        adata.write_h5ad(base, compression=None)
        meta = {"name": name or root.name, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "base_key": fingerprint(base), "n_obs": adata.n_obs, "n_vars": adata.n_vars,
                "source": str(source) if not isinstance(source, ad.AnnData) else "AnnData",
                "batch_key": adata.uns.get("batch_key"), "has_spatial": "spatial" in adata.obsm,
                "format": 1}
        (root / "project.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        if not (root / "gates.json").exists():
            (root / "gates.json").write_text("{}", encoding="utf-8")
        return cls(root)

    # ----------------------------------------------------------------- data
    @property
    def base_key(self) -> str:
        return self.meta["base_key"]

    def base(self) -> ad.AnnData:
        if self._base is None:
            self._base = ad.read_h5ad(self.root / "data" / "base.h5ad")
        return self._base

    def save_meta(self) -> None:
        (self.root / "project.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- gates
    def gates(self) -> dict:
        p = self.root / "gates.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def save_gates(self, gates: dict) -> None:
        tmp = self.root / "gates.json.tmp"
        tmp.write_text(json.dumps(gates, indent=1), encoding="utf-8")
        tmp.replace(self.root / "gates.json")

    # ----------------------------------------------------------------- manual annotations
    def annotations(self) -> "pd.DataFrame | None":
        import pandas as pd
        p = self.root / "annotations.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def save_annotation(self, column: str, labels: "pd.Series") -> None:
        """Persist labels (indexed by object id); re-applied after every pipeline run."""
        import pandas as pd
        cur = self.annotations()
        cur = pd.DataFrame(index=labels.index) if cur is None else cur.reindex(cur.index.union(labels.index))
        cur[column] = labels.astype(str).reindex(cur.index)
        cur.to_parquet(self.root / "annotations.parquet")

    def apply_annotations(self, adata) -> None:
        import pandas as pd
        ann = self.annotations()
        if ann is None:
            return
        for c in ann.columns:
            v = ann[c].reindex(adata.obs_names).fillna("unlabeled")
            adata.obs[c] = pd.Categorical(v.to_numpy())

    # ----------------------------------------------------------------- pipelines
    def pipeline_path(self, name: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in name).strip() or "pipeline"
        return self.root / "pipelines" / f"{safe}.json"

    def list_pipelines(self) -> list[str]:
        return sorted(Pipeline.load(p).name for p in (self.root / "pipelines").glob("*.json"))

    def save_pipeline(self, pipeline: Pipeline) -> Path:
        return pipeline.save(self.pipeline_path(pipeline.name))

    def load_pipeline(self, name: str) -> Pipeline:
        return Pipeline.load(self.pipeline_path(name))

    def delete_pipeline(self, name: str) -> None:
        self.pipeline_path(name).unlink(missing_ok=True)

    # ----------------------------------------------------------------- execution
    def plan(self, pipeline: Pipeline) -> list[dict]:
        return engine.plan(pipeline, self.base_key, self.cache, self.gates())

    def run(self, pipeline: Pipeline | str, ctx: engine.Context | None = None, **kw):
        if isinstance(pipeline, str):
            pipeline = self.load_pipeline(pipeline)
        ctx = ctx or engine.Context()
        ctx.gates = self.gates()
        adata, report = engine.run(pipeline, self.base(), self.base_key, self.cache, ctx, **kw)
        self.apply_annotations(adata)
        return adata, report

    def export(self, adata: ad.AnnData, name: str) -> Path:
        out = self.root / "results" / f"{name}.h5ad"
        a = adata.copy()
        a.uns = _h5ad_safe(dict(a.uns))
        a.write_h5ad(out)
        return out

    def clear_cache(self, keep: set[str] | None = None) -> int:
        """Delete cached steps (except ``keep``); returns bytes freed."""
        freed = 0
        for d in self.cache.root.iterdir():
            if d.is_dir() and (keep is None or d.name not in keep):
                freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d)
        return freed

    def info(self) -> dict:
        return {**self.meta, "root": str(self.root), "pipelines": self.list_pipelines(),
                "cache_entries": sum(1 for _ in self.cache.root.iterdir()),
                "cache_bytes": self.cache.size_bytes()}


def _h5ad_safe(obj):
    """Convert uns values h5ad cannot store (Series, odd keys/columns) into storable forms."""
    import numpy as np
    import pandas as pd
    if isinstance(obj, dict):
        return {str(k): _h5ad_safe(v) for k, v in obj.items() if v is not None}
    if isinstance(obj, pd.Series):
        return obj.to_frame(name=str(obj.name or "value"))
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.columns = [str(c) for c in df.columns]
        df.index = df.index.astype(str)
        return df
    if isinstance(obj, (list, tuple)) and obj and not isinstance(obj[0], (str, int, float, bool, np.number)):
        return {str(i): _h5ad_safe(v) for i, v in enumerate(obj)}
    return obj
