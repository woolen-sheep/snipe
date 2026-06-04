from __future__ import annotations

from snipe import config as config_module


def test_compute_default_workers_uses_gpu_default(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_inference_providers", lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])

    assert config_module.compute_default_workers() == 16


def test_compute_default_workers_uses_cpu_count_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "get_inference_providers", lambda: ["CPUExecutionProvider"])
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 12)

    assert config_module.compute_default_workers() == 12
