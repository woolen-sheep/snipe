from __future__ import annotations

from pathlib import Path

from snipe.init import run_init


def test_run_init_writes_default_config_when_missing(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "config.yaml"
    ensured_calls: list[str] = []

    monkeypatch.setattr("snipe.init.initialize_runtime", lambda path: (path, True, {"classifier": tmp_path / "classifier.onnx"}))

    run_init(config_path)

    assert ensured_calls == []


def test_run_init_uses_requested_config_path(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "custom-config.yaml"
    captured: list[Path | None] = []

    def fake_initialize_runtime(path: Path | None):
        captured.append(path)
        return config_path, False, {}

    monkeypatch.setattr("snipe.init.initialize_runtime", fake_initialize_runtime)

    run_init(config_path)

    assert captured == [config_path]
