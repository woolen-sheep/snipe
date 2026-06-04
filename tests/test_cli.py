from __future__ import annotations

import shutil
import stat
import subprocess
import sys
from pathlib import Path

from exifmwg import ImageMetadata
import pytest

from snipe.cli import parse_args
from snipe.config import DEFAULT_CONFIG_PATH
from snipe.image_io import load_image
from snipe.metadata import extract_annotation_snapshot, extract_regions_from_xmp
from tests.conftest import EXPECTED_DETECTIONS, METADATA_REGION_TOLERANCE_RATIO


def _set_readonly(path: Path) -> None:
    path.chmod(path.stat().st_mode & ~stat.S_IWRITE)


def _is_writable(path: Path) -> bool:
    return bool(path.stat().st_mode & stat.S_IWRITE)


def _keyword_labels(metadata: ImageMetadata) -> list[str]:
    keyword_info = getattr(metadata, "keyword_info", None)
    if keyword_info is None:
        return []
    return [entry.keyword for entry in getattr(keyword_info, "hierarchy", []) if getattr(entry, "keyword", None)]


def _assert_region_value_close(actual: int, expected: int) -> None:
    tolerance = max(1.0, abs(expected) * METADATA_REGION_TOLERANCE_RATIO)
    assert actual == pytest.approx(expected, abs=tolerance)


def test_parse_args_defaults_to_process_command(sample_image_dir: Path) -> None:
    resolved = parse_args([str(sample_image_dir), "--dry-run", "--workers", "1", "--no-gps-filter"])

    assert resolved.command == "process"
    assert resolved.process.source == sample_image_dir
    assert resolved.process.dry_run is True
    assert resolved.process.workers == 1
    assert resolved.process.gps_filter is False


def test_parse_args_supports_init_command() -> None:
    resolved = parse_args(["init"])

    assert resolved.command == "init"
    assert resolved.config_path == DEFAULT_CONFIG_PATH


def test_python_module_cli_process_dry_run_succeeds(
    sample_image_paths: list[Path],
    inference_ready: None,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "dry-run-source"
    source_dir.mkdir()
    for sample_image_path in sample_image_paths:
        shutil.copy2(sample_image_path, source_dir / sample_image_path.name)

    command = [
        sys.executable,
        "-m",
        "snipe",
        str(source_dir),
        "--dry-run",
        "--workers",
        "1",
        "--no-gps-filter",
        "--log-level",
        "INFO",
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Found {len(sample_image_paths)} images to process" in completed.stderr


def test_python_module_cli_process_writes_expected_metadata_from_readonly_sources(
    sample_image_paths: list[Path],
    inference_ready: None,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "readonly-source"
    output_dir = tmp_path / "processed-output"
    source_dir.mkdir()
    output_dir.mkdir()

    for sample_image_path in sample_image_paths:
        copied_source = source_dir / sample_image_path.name
        shutil.copy2(sample_image_path, copied_source)
        _set_readonly(copied_source)
        assert not _is_writable(copied_source)

    command = [
        sys.executable,
        "-m",
        "snipe",
        str(source_dir),
        "-o",
        str(output_dir),
        "--workers",
        "1",
        "--region-filter",
        "CN",
        "--no-bird-keyword",
        "--log-level",
        "INFO",
    ]

    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Found {len(sample_image_paths)} images to process" in completed.stderr

    for sample_image_path in sample_image_paths:
        source_image = source_dir / sample_image_path.name
        output_image = output_dir / sample_image_path.name
        expected = EXPECTED_DETECTIONS[sample_image_path.name]
        expected_bbox = expected["bbox"]
        expected_region = {
            "label": expected["bird_name"],
            "x": expected_bbox[0],
            "y": expected_bbox[1],
            "width": expected_bbox[2] - expected_bbox[0],
            "height": expected_bbox[3] - expected_bbox[1],
        }

        assert source_image.exists()
        assert output_image.exists()
        assert not _is_writable(source_image)
        assert _is_writable(output_image)

        metadata = ImageMetadata(str(output_image))
        assert _keyword_labels(metadata) == [expected["bird_name"]]

        image = load_image(output_image)
        image_size = image.size
        image.close()

        regions = extract_regions_from_xmp(output_image, image_size)
        assert len(regions) == 1
        region = regions[0]
        assert region["label"] == expected_region["label"]
        _assert_region_value_close(region["x"], expected_region["x"])
        _assert_region_value_close(region["y"], expected_region["y"])
        _assert_region_value_close(region["width"], expected_region["width"])
        _assert_region_value_close(region["height"], expected_region["height"])


def test_python_module_cli_reset_clears_keywords_and_regions(
    sample_image_paths: list[Path],
    inference_ready: None,
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "reset-source"
    output_dir = tmp_path / "reset-output"
    source_dir.mkdir()
    output_dir.mkdir()

    for sample_image_path in sample_image_paths:
        shutil.copy2(sample_image_path, source_dir / sample_image_path.name)

    process_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "snipe",
            str(source_dir),
            "-o",
            str(output_dir),
            "--workers",
            "1",
            "--region-filter",
            "CN",
            "--no-bird-keyword",
            "--log-level",
            "INFO",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert process_completed.returncode == 0, process_completed.stderr

    reset_completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "snipe",
            "clear",
            str(output_dir),
            "--workers",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert reset_completed.returncode == 0, reset_completed.stderr

    for sample_image_path in sample_image_paths:
        output_image = output_dir / sample_image_path.name
        assert output_image.exists()

        snapshot = extract_annotation_snapshot(output_image)
        assert snapshot["keywords"] == []
        assert snapshot["has_regions"] is False

        image = load_image(output_image)
        image_size = image.size
        image.close()

        regions = extract_regions_from_xmp(output_image, image_size)
        assert regions == []


def test_python_module_cli_reset_accepts_single_file_path(
    inference_ready: None,
    tmp_path: Path,
) -> None:
    source_image = Path(__file__).resolve().parents[1] / "tests" / "assets" / "DSC_7655.NEF"
    target_image = tmp_path / source_image.name
    shutil.copy2(source_image, target_image)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "snipe",
            "clear",
            str(target_image),
            "--workers",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Resetting Snipe-managed XMP metadata for 1 images under" in completed.stderr
