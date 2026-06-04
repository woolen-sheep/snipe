from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .onnxruntime_utils import get_inference_providers
from .paths import CONFIG_DIR, DATA_DIR, MODELS_DIR


DEFAULT_CONFIG_PATH = CONFIG_DIR / "config.yaml"


def compute_default_workers() -> int:
    providers = get_inference_providers()
    if "CUDAExecutionProvider" in providers:
        return 16
    return min(32, (os.cpu_count() or 4))


DEFAULT_WORKERS = compute_default_workers()


class ProcessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source: Path | None = None
    output_dir: Path | None = None
    conf: float = 0.35
    workers: int = Field(default=DEFAULT_WORKERS, ge=1)
    detector_model: Path = MODELS_DIR / "yolo26x-seg.onnx"
    detector_merge_overlap_threshold: float = 0.7
    detector_min_box_area_ratio: float = 0.05
    gps_filter: bool = True
    gps_filter_data: Path = DATA_DIR / "location_species_filter.json"
    region_filter: str | None = None
    language: Literal["CN", "US"] = "CN"
    classifier_model: Path = MODELS_DIR / "classifier.onnx"
    labels: Path = DATA_DIR / "labels.txt"
    taxonomy: Path = DATA_DIR / "common_name.csv"
    bird_keyword: str | None = "bird"
    log_level: str | None = None
    dry_run: bool = False

    @field_validator("bird_keyword")
    @classmethod
    def normalize_bird_keyword(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str) -> str:
        return value.upper()

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.lower() in {"null", "none"}:
            return None
        return normalized.upper()


class PreviewConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source: Path | None = None
    log_level: str | None = "INFO"
    debug: bool = False

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.lower() in {"null", "none"}:
            return None
        return normalized.upper()


class Config(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    command: Literal["process", "preview"] = "process"
    process: ProcessConfig = Field(default_factory=ProcessConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH, exclude=True)

    @classmethod
    def from_file(cls, path: Path | None = None) -> "Config":
        resolved_path = (path or DEFAULT_CONFIG_PATH).expanduser()
        payload: dict[str, Any] = {}

        if resolved_path.exists():
            with resolved_path.open("r", encoding="utf-8") as handle:
                loaded = yaml.safe_load(handle) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Config file must contain a YAML mapping: {resolved_path}")
            payload = loaded

        payload["config_path"] = resolved_path
        return cls.model_validate(payload)

    def apply_cli_overrides(self, command: Literal["process", "preview"], overrides: Mapping[str, Any]) -> "Config":
        merged = self.model_dump(mode="python", exclude={"config_path"})
        merged["command"] = command
        merged.setdefault(command, {})
        merged[command].update(dict(overrides))
        merged["config_path"] = self.config_path
        return type(self).model_validate(merged)

    def replace_with(self, other: "Config") -> "Config":
        self.command = other.command
        self.process = other.process
        self.preview = other.preview
        self.config_path = other.config_path
        return self

    def reload(self, path: Path | None = None) -> "Config":
        return self.replace_with(type(self).from_file(path))

    def save(self, path: Path | None = None) -> Path:
        target = (path or self.config_path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.model_dump(mode="json", exclude={"command", "config_path"}),
                handle,
                allow_unicode=True,
                sort_keys=False,
            )
        self.config_path = target
        return target

    @property
    def active_log_level(self) -> str | None:
        return self.preview.log_level if self.command == "preview" else self.process.log_level


config = Config()


__all__ = [
    "Config",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_WORKERS",
    "PreviewConfig",
    "ProcessConfig",
    "ValidationError",
    "config",
]
