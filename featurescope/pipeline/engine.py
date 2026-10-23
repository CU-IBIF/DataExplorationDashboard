"""Pipeline execution with content-addressed caching.

cache key(step i) = sha256(key(step i-1), op name, op version, resolved params[, gates])

Because keys depend only on inputs, a pipeline computed on a cluster and
copied back (``cache/<key>/``) is picked up locally with no recomputation, and
editing step k only invalidates steps k..n.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import anndata as ad

from .delta import Delta
from .registry import get_op
from .spec import Pipeline

log = logging.getLogger("featurescope")

ENGINE_VERSION = "1"


@dataclass
class Context:
    """Run-time information passed to every operation."""
    gates: dict = field(default_factory=dict)
    device: str = "auto"                     # auto | cpu | cuda | mps
    seed: int = 0
    n_jobs: int = -1
    progress: Callable[[str, float], None] = lambda msg, frac: None
    should_cancel: Callable[[], bool] = lambda: False

    def torch_device(self):
        import torch
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def gpu_available(self) -> bool:
        if self.device == "cpu":
            return False
        try:
            import cupy  # noqa: F401
            import rapids_singlecell  # noqa: F401
            return True
        except Exception:
            return False


class Cancelled(RuntimeError):
    pass


def _canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))


def step_key(prev_key: str, op_name: str, params: dict, gates: dict | None = None) -> str:
    info = get_op(op_name)
    payload = {"prev": prev_key, "op": op_name, "params": info.resolve(params), "v": ENGINE_VERSION,
               "opv": getattr(info.fn, "version", "1")}
    if info.uses_gates:
        payload["gates"] = gates or {}
    return hashlib.sha256(_canon(payload).encode()).hexdigest()[:24]


def chain_keys(pipeline: Pipeline, base_key: str, gates: dict | None = None) -> list[str | None]:
    """Cache key for every step (None for disabled steps, which are pass-through)."""
    keys, prev = [], base_key
    for s in pipeline.steps:
        if not s.enabled:
            keys.append(None)
            continue
        prev = step_key(prev, s.op, s.params, gates)
        keys.append(prev)
    return keys


class Cache:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key

    def has(self, key: str) -> bool:
        return (self.path(key) / "delta.json").exists()

    def get(self, key: str) -> Delta:
        return Delta.load(self.path(key))

    def put(self, key: str, delta: Delta, meta: dict) -> None:
        delta.save(self.path(key))
        (self.path(key) / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")

    def meta(self, key: str) -> dict:
        p = self.path(key) / "meta.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in self.root.rglob("*") if f.is_file())


def plan(pipeline: Pipeline, base_key: str, cache: Cache, gates: dict | None = None) -> list[dict]:
    """Which steps are already cached (no computation)."""
    out = []
    for s, k in zip(pipeline.steps, chain_keys(pipeline, base_key, gates)):
        status = "disabled" if k is None else ("cached" if cache.has(k) else "pending")
        out.append({"id": s.id, "op": s.op, "key": k, "status": status,
                    "meta": cache.meta(k) if status == "cached" else {}})
    return out


def run(pipeline: Pipeline, adata: ad.AnnData, base_key: str, cache: Cache | None, ctx: Context | None = None,
        upto: str | None = None, force: bool = False, cache_only: bool = False) -> tuple[ad.AnnData, list[dict]]:
    """Execute ``pipeline`` on a copy of ``adata``; returns (result, per-step report).

    upto:        stop after the step with this id (inclusive)
    force:       recompute even if cached
    cache_only:  never compute; stop at the first step that is not cached
    """
    ctx = ctx or Context()
    adata = adata.copy()
    keys = chain_keys(pipeline, base_key, ctx.gates)
    report = []
    n = max(1, sum(k is not None for k in keys))
    done = 0
    for s, k in zip(pipeline.steps, keys):
        if k is None:
            report.append({"id": s.id, "op": s.op, "status": "disabled"})
        else:
            if ctx.should_cancel():
                raise Cancelled("cancelled")
            t0 = time.time()
            info = get_op(s.op)
            if cache is not None and not force and cache.has(k):
                delta = cache.get(k)
                status = "cached"
            elif cache_only:
                report.append({"id": s.id, "op": s.op, "key": k, "status": "pending"})
                break
            else:
                ctx.progress(f"{info.label}", done / n)
                delta = info.fn(adata, ctx, **info.resolve(s.params))
                status = "computed"
                if cache is not None:
                    cache.put(k, delta, {"op": s.op, "params": info.resolve(s.params), "seconds": time.time() - t0,
                                         "n_obs": adata.n_obs, "n_vars": adata.n_vars, "host": platform.node(),
                                         "time": time.strftime("%Y-%m-%d %H:%M:%S"), "log": delta.log})
            adata = delta.apply(adata)
            done += 1
            report.append({"id": s.id, "op": s.op, "key": k, "status": status,
                           "seconds": round(time.time() - t0, 3), "log": delta.log,
                           "shape": [adata.n_obs, adata.n_vars]})
            log.info("%s %s (%s) %.2fs", status, s.op, k, time.time() - t0)
        if upto and s.id == upto:
            break
    ctx.progress("done", 1.0)
    return adata, report
