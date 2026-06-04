from __future__ import annotations

from pathlib import Path

import pytest

from snipe.gps_metadata import extract_gps_coordinates


@pytest.mark.parametrize(
    ("image_name", "expected_latitude", "expected_longitude"),
    [
        ("DSC_4626.jpg", 30.243489391710, 119.734743479154),
        ("DSC_4768.jpg", 30.236340000000, 119.736900000000),
        ("DSC_4853.jpg", 30.233010800023, 119.737310400000),
    ],
)
def test_extract_gps_coordinates_reads_sample_jpeg_metadata(
    image_name: str,
    expected_latitude: float,
    expected_longitude: float,
    sample_image_dir: Path,
) -> None:
    coordinates = extract_gps_coordinates(sample_image_dir / image_name)

    assert coordinates is not None
    latitude, longitude = coordinates
    assert latitude == pytest.approx(expected_latitude, abs=1e-6)
    assert longitude == pytest.approx(expected_longitude, abs=1e-6)
