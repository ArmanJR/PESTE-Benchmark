"""Load, validate, and digest suite/model declarations."""

import json
from pathlib import Path

from pydantic import BaseModel

from peste.constants import PROJECT_ROOT
from peste.digests import canonical_json, sha256_bytes
from peste.schemas import ModelSpec, SuiteSpec


def _load[SpecT: BaseModel](path: Path, model_type: type[SpecT]) -> SpecT:
    with path.open(encoding="utf-8") as handle:
        return model_type.model_validate(json.load(handle))


def load_suite(suite_id: str, root: Path = PROJECT_ROOT) -> SuiteSpec:
    return _load(root / "suites" / suite_id / "suite.json", SuiteSpec)


def load_model(model_id: str, root: Path = PROJECT_ROOT) -> ModelSpec:
    return _load(root / "models" / f"{model_id}.json", ModelSpec)


def spec_digest(spec: BaseModel) -> str:
    return sha256_bytes(canonical_json(spec.model_dump(mode="json")))


def discover_models(root: Path = PROJECT_ROOT) -> list[ModelSpec]:
    return [_load(path, ModelSpec) for path in sorted((root / "models").glob("*.json"))]
