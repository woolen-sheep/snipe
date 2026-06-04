from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from loguru import logger
import pyexiv2


def _to_float(value: object) -> float:
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, str) and "/" in value:
        return float(Fraction(value))
    if isinstance(value, tuple):
        return _to_float(value[0]) / _to_float(value[1])
    return float(value)


def _parse_gps_coordinate(value: object, reference: object) -> float | None:
    if value is None:
        return None

    try:
        if isinstance(value, str) and " " in value:
            parts = value.split()
            if len(parts) == 3:
                degrees = _to_float(parts[0])
                minutes = _to_float(parts[1])
                seconds = _to_float(parts[2])
                coordinate = degrees + minutes / 60 + seconds / 3600
            else:
                coordinate = _to_float(value)
        if isinstance(value, (list, tuple)) and len(value) == 3:
            degrees = _to_float(value[0])
            minutes = _to_float(value[1])
            seconds = _to_float(value[2])
            coordinate = degrees + minutes / 60 + seconds / 3600
        elif not (isinstance(value, str) and " " in value):
            coordinate = _to_float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None

    ref_text = str(reference).strip().upper() if reference is not None else ""
    if ref_text in {"S", "W"}:
        coordinate *= -1
    return coordinate


def extract_gps_coordinates(image_path: Path) -> tuple[float, float] | None:
    try:
        with pyexiv2.Image(str(image_path)) as image:
            exif = image.read_exif()
    except Exception as exc:
        logger.debug("pyexiv2 failed while reading GPS for {}: {}", image_path, exc)
        return None

    latitude = _parse_gps_coordinate(
        exif.get("Exif.GPSInfo.GPSLatitude"),
        exif.get("Exif.GPSInfo.GPSLatitudeRef"),
    )
    longitude = _parse_gps_coordinate(
        exif.get("Exif.GPSInfo.GPSLongitude"),
        exif.get("Exif.GPSInfo.GPSLongitudeRef"),
    )
    if latitude is None or longitude is None:
        logger.debug(
            "No parseable GPSLatitude/GPSLongitude tags found for {}. Available keys include: {}",
            image_path,
            ", ".join(sorted(str(key) for key in exif.keys())[:12]),
        )
        return None

    logger.debug(
        "Extracted GPS for {}: latitude={:.6f}, longitude={:.6f}",
        image_path,
        latitude,
        longitude,
    )
    return latitude, longitude
