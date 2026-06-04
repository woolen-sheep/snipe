from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

from loguru import logger

from .config import Config, DEFAULT_CONFIG_PATH
from .paths import MODELS_DIR


MODEL_RELEASE_REPO = "woolen-sheep/snipe"
MODEL_RELEASE_TAG = "models-20260604"
MODEL_RELEASE_BASE_URL = f"https://github.com/{MODEL_RELEASE_REPO}/releases/download/{MODEL_RELEASE_TAG}"


@dataclass(frozen=True, slots=True)
class ManagedModelSpec:
    logical_name: str
    filename: str
    sha256: str

    @property
    def url(self) -> str:
        return f"{MODEL_RELEASE_BASE_URL}/{self.filename}"

    @property
    def target_path(self) -> Path:
        return MODELS_DIR / self.filename


MANAGED_MODELS: dict[str, ManagedModelSpec] = {
    "classifier": ManagedModelSpec(
        logical_name="classifier",
        filename="classifier.onnx",
        sha256="a9788c25d1bd5d4d5081cc5093d58f10ff61dd19c2e77e92190d0eb111e3956b",
    ),
    "detector": ManagedModelSpec(
        logical_name="detector",
        filename="yolo26x-seg.onnx",
        sha256="d95acf249a4f431d0d531bde76cdf4b92c55cb5f8cbb6befc3dafe58a3466f8a",
    ),
}

_MODEL_LOCKS = {name: threading.Lock() for name in MANAGED_MODELS}


def managed_model_path(logical_name: str) -> Path:
    return MANAGED_MODELS[logical_name].target_path


def is_managed_model_path(logical_name: str, path: Path) -> bool:
    return path.expanduser().resolve() == managed_model_path(logical_name).expanduser().resolve()


def verify_file_sha256(path: Path, expected_sha256: str) -> bool:
    if not path.exists() or not path.is_file():
        return False

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_sha256.lower()


def _download_file(url: str, target_path: Path, expected_sha256: str) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f"{target_path.name}.", suffix=".tmp", dir=str(target_path.parent))
    tmp_path = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as output_handle:
            with urlopen(url, timeout=300) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output_handle.write(chunk)
                    digest.update(chunk)

        actual_sha256 = digest.hexdigest().lower()
        if actual_sha256 != expected_sha256.lower():
            raise RuntimeError(
                f"Downloaded model checksum mismatch for {target_path.name}: expected {expected_sha256}, got {actual_sha256}"
            )

        tmp_path.replace(target_path)
        return target_path
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def ensure_managed_model(logical_name: str, *, force_verify: bool = False) -> Path:
    spec = MANAGED_MODELS[logical_name]
    target_path = spec.target_path
    lock = _MODEL_LOCKS[logical_name]

    with lock:
        if target_path.exists():
            if not force_verify or verify_file_sha256(target_path, spec.sha256):
                return target_path
            logger.warning("Managed model checksum mismatch at {}; re-downloading", target_path)

        logger.info("Downloading {} model from {}", logical_name, spec.url)
        return _download_file(spec.url, target_path, spec.sha256)


def ensure_required_models(*, force_verify: bool = False) -> dict[str, Path]:
    return {
        name: ensure_managed_model(name, force_verify=force_verify)
        for name in MANAGED_MODELS
    }


def ensure_default_config(config_path: Path | None = None) -> tuple[Path, bool]:
    target_path = (config_path or DEFAULT_CONFIG_PATH).expanduser()
    if target_path.exists():
        return target_path, False
    Config().save(target_path)
    return target_path, True


def initialize_runtime(config_path: Path | None = None) -> tuple[Path, bool, dict[str, Path]]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resolved_config_path, created_config = ensure_default_config(config_path)
    resolved_models = ensure_required_models(force_verify=True)
    return resolved_config_path, created_config, resolved_models
