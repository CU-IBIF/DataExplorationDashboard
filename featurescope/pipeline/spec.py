"""Serializable pipeline definitions (JSON / YAML)."""
from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


@dataclass
class Step:
    op: str                                  # registered operation name, e.g. "normalize"
    params: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    enabled: bool = True
    note: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(op=d["op"], params=dict(d.get("params", {})), id=d.get("id") or uuid.uuid4().hex[:8],
                   enabled=d.get("enabled", True), note=d.get("note", ""))


@dataclass
class Pipeline:
    name: str = "pipeline"
    steps: list[Step] = field(default_factory=list)
    description: str = ""

    def add(self, op: str, **params) -> "Pipeline":
        self.steps.append(Step(op=op, params=params))
        return self

    def to_dict(self) -> dict:
        return {"name": self.name, "description": self.description, "steps": [asdict(s) for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> "Pipeline":
        return cls(name=d.get("name", "pipeline"), description=d.get("description", ""),
                   steps=[Step.from_dict(s) for s in d.get("steps", [])])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        path = Path(path)
        text = path.read_text()
        d = yaml.safe_load(text) if path.suffix in (".yaml", ".yml") else json.loads(text)
        return cls.from_dict(d)
