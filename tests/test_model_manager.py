from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from snipe import model_manager


PAYLOAD_SIZE = 512 * 1024


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._buffer = BytesIO(payload)

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._buffer.close()


def _make_payload(seed: bytes) -> bytes:
    repeats = (PAYLOAD_SIZE // len(seed)) + 1
    return (seed * repeats)[:PAYLOAD_SIZE]


def _sha256_for_bytes(payload: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(payload)
    return digest.hexdigest()


def _sha256_for(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_initialize_runtime_creates_default_config_and_downloads_models(tmp_path: Path, monkeypatch) -> None:
    classifier_payload = _make_payload(b"classifier-bytes")
    detector_payload = _make_payload(b"detector-bytes")
    payloads = {
        "https://example.invalid/classifier.onnx": classifier_payload,
        "https://example.invalid/yolo26x-seg.onnx": detector_payload,
    }

    target_models_dir = tmp_path / "models"
    target_config_path = tmp_path / "config.yaml"
    monkeypatch.setattr(model_manager, "MODELS_DIR", target_models_dir)
    monkeypatch.setattr(
        model_manager,
        "MANAGED_MODELS",
        {
            "classifier": model_manager.ManagedModelSpec(
                logical_name="classifier",
                filename="classifier.onnx",
                sha256=_sha256_for_bytes(classifier_payload),
            ),
            "detector": model_manager.ManagedModelSpec(
                logical_name="detector",
                filename="yolo26x-seg.onnx",
                sha256=_sha256_for_bytes(detector_payload),
            ),
        },
    )
    monkeypatch.setattr(
        model_manager,
        "_MODEL_LOCKS",
        {"classifier": model_manager.threading.Lock(), "detector": model_manager.threading.Lock()},
    )

    original_url_property = model_manager.ManagedModelSpec.url
    monkeypatch.setattr(
        model_manager.ManagedModelSpec,
        "url",
        property(
            lambda self: "https://example.invalid/classifier.onnx"
            if self.logical_name == "classifier"
            else "https://example.invalid/yolo26x-seg.onnx"
        ),
    )
    monkeypatch.setattr(model_manager, "urlopen", lambda url, timeout=300: _FakeResponse(payloads[url]))

    try:
        resolved_config_path, created_config, resolved_models = model_manager.initialize_runtime(target_config_path)
    finally:
        monkeypatch.setattr(model_manager.ManagedModelSpec, "url", original_url_property)

    assert created_config is True
    assert resolved_config_path == target_config_path
    assert target_config_path.exists()
    assert resolved_models["classifier"].read_bytes() == classifier_payload
    assert resolved_models["detector"].read_bytes() == detector_payload


def test_ensure_managed_model_redownloads_when_checksum_mismatches(tmp_path: Path, monkeypatch) -> None:
    payload = _make_payload(b"correct-bytes")
    target_models_dir = tmp_path / "models"
    target_models_dir.mkdir()
    broken_target = target_models_dir / "classifier.onnx"
    broken_target.write_bytes(b"broken-bytes")

    monkeypatch.setattr(model_manager, "MODELS_DIR", target_models_dir)
    monkeypatch.setattr(
        model_manager,
        "MANAGED_MODELS",
        {
            "classifier": model_manager.ManagedModelSpec(
                logical_name="classifier",
                filename="classifier.onnx",
                sha256=_sha256_for_bytes(payload),
            )
        },
    )
    monkeypatch.setattr(model_manager, "_MODEL_LOCKS", {"classifier": model_manager.threading.Lock()})

    original_url_property = model_manager.ManagedModelSpec.url
    monkeypatch.setattr(model_manager.ManagedModelSpec, "url", property(lambda self: "https://example.invalid/classifier.onnx"))
    monkeypatch.setattr(model_manager, "urlopen", lambda url, timeout=300: _FakeResponse(payload))

    try:
        resolved_path = model_manager.ensure_managed_model("classifier", force_verify=True)
    finally:
        monkeypatch.setattr(model_manager.ManagedModelSpec, "url", original_url_property)

    assert resolved_path.read_bytes() == payload
