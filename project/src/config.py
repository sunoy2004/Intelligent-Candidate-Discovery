"""Configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def load_yaml(name: str) -> dict[str, Any]:
    path = project_path("configs", name)
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pipeline_config() -> dict[str, Any]:
    return load_yaml("pipeline.yaml")


def load_weights_config() -> dict[str, Any]:
    return load_yaml("weights.yaml")


def resolve_path(relative: str) -> Path:
    return (PROJECT_ROOT / relative).resolve()


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
