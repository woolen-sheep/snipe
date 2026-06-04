from __future__ import annotations

from pathlib import Path

import pytest

from snipe.processing import compute_gpu_retry_delay, process_image


class _FakeBBox:
    def as_tuple(self) -> tuple[int, int, int, int]:
        return (1, 2, 3, 4)


class _FakeDetection:
    bird_name = "bird"
    score = 0.9
    bbox = _FakeBBox()
    image_width = 100
    image_height = 50


def test_process_image_retries_gpu_overload_error(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"jpg")
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def fake_detect_birds(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("cuLaunchKernel returned error too many resources requested for launch (701)")
        return [_FakeDetection()]

    monkeypatch.setattr("snipe.processing.detect_birds", fake_detect_birds)
    monkeypatch.setattr("snipe.processing.time.sleep", lambda seconds: sleep_calls.append(seconds))
    monkeypatch.setattr("snipe.processing.copy_for_output", lambda source, output_dir: source)
    monkeypatch.setattr("snipe.processing.write_metadata", lambda *args, **kwargs: None)

    target, regions = process_image(image_path)

    assert target == image_path
    assert len(regions) == 1
    assert attempts["count"] == 2
    assert sleep_calls == [0.5]


def test_process_image_caps_non_overload_gpu_error_retries(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"jpg")
    attempts = {"count": 0}
    sleep_calls: list[float] = []

    def fake_detect_birds(*args, **kwargs):
        attempts["count"] += 1
        raise RuntimeError("CUDNN_FE failure 11: CUDNN_BACKEND_API_FAILED")

    monkeypatch.setattr("snipe.processing.detect_birds", fake_detect_birds)
    monkeypatch.setattr("snipe.processing.time.sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(RuntimeError, match="GPU inference failed"):
        process_image(image_path)

    assert attempts["count"] == 3
    assert sleep_calls == [0.5, 1.0]


def test_compute_gpu_retry_delay_caps_then_adds_jitter(monkeypatch) -> None:
    monkeypatch.setattr("snipe.processing.random.random", lambda: 1.0)

    assert compute_gpu_retry_delay(1) == 0.5
    assert compute_gpu_retry_delay(6) == 16.0
    assert compute_gpu_retry_delay(7) == 24.0


def test_process_image_does_not_retry_non_gpu_error(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"jpg")
    sleep_calls: list[float] = []

    def fake_detect_birds(*args, **kwargs):
        raise RuntimeError("taxonomy lookup failed")

    monkeypatch.setattr("snipe.processing.detect_birds", fake_detect_birds)
    monkeypatch.setattr("snipe.processing.time.sleep", lambda seconds: sleep_calls.append(seconds))

    with pytest.raises(RuntimeError, match="taxonomy lookup failed"):
        process_image(image_path)

    assert sleep_calls == []
