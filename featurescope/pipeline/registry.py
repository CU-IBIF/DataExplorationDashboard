"""Registry of pipeline operations and their parameter schemas.

Every operation is a pure function ``fn(adata, ctx, **params) -> Delta``.  It
must not mutate ``adata``; it describes its outputs in a :class:`Delta` which
the engine applies and caches.  This is what makes results reloadable without
recomputation: the cache stores deltas keyed by a hash of the upstream chain.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Param:
    name: str
    type: str                     # int | float | bool | str | choice | columns | column | features | gate
    default: Any = None
    choices: list | None = None
    help: str = ""
    min: float | None = None
    max: float | None = None


@dataclass
class OpInfo:
    name: str
    fn: Callable
    category: str
    label: str
    params: list[Param] = field(default_factory=list)
    doc: str = ""
    uses_gates: bool = False      # cache key must include gate geometry
    heavy: bool = False           # suggest remote / GPU execution for large data
    gpu: bool = False             # benefits from a GPU
    requires: list[str] = field(default_factory=list)  # optional python packages

    def schema(self) -> dict:
        return {
            "name": self.name, "category": self.category, "label": self.label, "doc": self.doc,
            "heavy": self.heavy, "gpu": self.gpu, "requires": self.requires,
            "params": [p.__dict__ for p in self.params],
        }

    def resolve(self, params: dict) -> dict:
        out = {p.name: p.default for p in self.params}
        unknown = set(params) - set(out)
        if unknown:
            raise ValueError(f"{self.name}: unknown parameter(s) {sorted(unknown)}")
        out.update({k: v for k, v in params.items()})
        return out


REGISTRY: dict[str, OpInfo] = {}


def op(name: str, category: str, label: str | None = None, params: list[Param] | None = None, **kw):
    def deco(fn):
        REGISTRY[name] = OpInfo(name=name, fn=fn, category=category, label=label or name,
                                params=params or [], doc=(fn.__doc__ or "").strip(), **kw)
        return fn
    return deco


def get_op(name: str) -> OpInfo:
    from .. import steps  # noqa: F401  (populate the registry)
    if name not in REGISTRY:
        raise KeyError(f"unknown operation {name!r}")
    return REGISTRY[name]


def all_ops() -> list[dict]:
    from .. import steps  # noqa: F401
    return [o.schema() for o in REGISTRY.values()]
