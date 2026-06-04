from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from snipe import BirdDetection, detect_birds
from snipe.image_io import RAW_SUFFIXES
from tests.conftest import BBOX_TOLERANCE_RATIO, EXPECTED_DETECTIONS


EXPECTED_RAW_SUFFIXES = {
    ".nef",
    ".cr2",
    ".cr3",
    ".arw",
    ".orf",
    ".rw2",
    ".raf",
    ".dng",
    ".pef",
    ".sr2",
}


def _serialize_detection(result: BirdDetection) -> tuple[str, str, float, tuple[int, int, int, int], int, int]:
    return (
        result.species_code,
        result.bird_name,
        round(result.score, 6),
        result.bbox.as_tuple(),
        result.image_width,
        result.image_height,
    )


def _assert_bbox_close(actual: tuple[int, int, int, int], expected: tuple[int, int, int, int]) -> None:
    for actual_value, expected_value in zip(actual, expected, strict=True):
        tolerance = max(1.0, abs(expected_value) * BBOX_TOLERANCE_RATIO)
        assert actual_value == pytest.approx(expected_value, abs=tolerance)


def test_detect_birds_returns_stable_results_for_path_bytes_and_file_handle(
    sample_jpeg_paths: list[Path],
    inference_ready: None,
) -> None:
    image_path = sample_jpeg_paths[0]

    path_results = detect_birds(image_path, region_code="CN", language_code="CN")
    bytes_results = detect_birds(image_path.read_bytes(), region_code="CN", language_code="CN")

    with image_path.open("rb") as handle:
        file_results = detect_birds(handle, region_code="CN", language_code="CN")

    assert path_results
    assert [_serialize_detection(item) for item in path_results] == [
        _serialize_detection(item) for item in bytes_results
    ]
    assert [_serialize_detection(item) for item in path_results] == [
        _serialize_detection(item) for item in file_results
    ]


def test_all_sample_jpegs_are_covered_by_test_inputs(
    sample_test_asset_paths: list[Path],
) -> None:
    covered_names = {path.name for path in sample_test_asset_paths}
    asset_names = {path.name for path in sample_test_asset_paths[0].parent.iterdir() if path.is_file()}

    assert covered_names == asset_names


def test_detect_birds_returns_expected_low_quality_detections(
    low_quality_image_path: Path,
    inference_ready: None,
) -> None:
    results = detect_birds(low_quality_image_path, region_code="CN", language_code="CN")

    assert len(results) == 2

    expected = [
        ("bkcsta1", "黑领椋鸟", (809, 451, 953, 596), 1906, 1102),
        ("bkcsta1", "黑领椋鸟", (839, 498, 1014, 673), 1906, 1102),
    ]

    for result, (species_code, bird_name, bbox, image_width, image_height) in zip(results, expected, strict=True):
        left, top, right, bottom = result.bbox.as_tuple()
        assert 0 <= left <= right <= result.image_width
        assert 0 <= top <= bottom <= result.image_height
        assert result.bbox.width == right - left
        assert result.bbox.height == bottom - top
        assert result.species_code == species_code
        assert result.bird_name == bird_name
        assert result.image_width == image_width
        assert result.image_height == image_height
        assert 0.0 <= result.score <= 1.0
        _assert_bbox_close(result.bbox.as_tuple(), bbox)


@pytest.mark.parametrize("image_name", ["DSC_4626.jpg", "DSC_4768.jpg", "DSC_4853.jpg"])
def test_detect_birds_matches_expected_species_and_bbox_baseline(
    image_name: str,
    sample_image_dir: Path,
    inference_ready: None,
) -> None:
    image_path = sample_image_dir / image_name
    expected = EXPECTED_DETECTIONS[image_name]

    results = detect_birds(BytesIO(image_path.read_bytes()), region_code="CN", language_code="CN")

    assert len(results) == 1

    result = results[0]
    left, top, right, bottom = result.bbox.as_tuple()
    assert 0 <= left <= right <= result.image_width
    assert 0 <= top <= bottom <= result.image_height
    assert result.bbox.width == right - left
    assert result.bbox.height == bottom - top
    assert result.species_code == expected["species_code"]
    assert result.bird_name == expected["bird_name"]
    assert 0.0 <= result.score <= 1.0
    _assert_bbox_close(result.bbox.as_tuple(), expected["bbox"])


def test_raw_suffixes_include_common_camera_formats() -> None:
    assert EXPECTED_RAW_SUFFIXES.issubset(RAW_SUFFIXES)


def test_detect_birds_matches_expected_species_and_bbox_for_raw_path(
    sample_image_dir: Path,
    inference_ready: None,
) -> None:
    image_name = "DSC_7655.NEF"
    image_path = sample_image_dir / image_name
    expected = EXPECTED_DETECTIONS[image_name]

    results = detect_birds(image_path, region_code="CN", language_code="CN")

    assert len(results) == 1

    result = results[0]
    left, top, right, bottom = result.bbox.as_tuple()
    assert 0 <= left <= right <= result.image_width
    assert 0 <= top <= bottom <= result.image_height
    assert result.bbox.width == right - left
    assert result.bbox.height == bottom - top
    assert result.species_code == expected["species_code"]
    assert result.bird_name == expected["bird_name"]
    assert 0.0 <= result.score <= 1.0
    _assert_bbox_close(result.bbox.as_tuple(), expected["bbox"])
