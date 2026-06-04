from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from exifmwg import Dimensions, ImageMetadata, KeywordInfo, Region as MwgRegion, RegionInfo as MwgRegionInfo, XmpArea
from loguru import logger
import pyexiv2


@dataclass
class Region:
    label: str
    score: float
    bbox: tuple[int, int, int, int]
    image_size: tuple[int, int]


def normalize_region(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    width, height = image_size
    center_x = (x1 + x2) / 2 / width
    center_y = (y1 + y2) / 2 / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return center_x, center_y, box_width, box_height


def write_metadata(target: Path, regions: Sequence[Region], keywords: Iterable[str]) -> None:
    keyword_list = list(dict.fromkeys(keywords))
    logger.info("Writing {} regions and {} keywords to {}", len(regions), len(keyword_list), target)

    metadata = ImageMetadata(str(target))

    if keyword_list:
        metadata.keyword_info = KeywordInfo(delimited_strings=keyword_list)

    if regions:
        width, height = regions[0].image_size
        xmp_regions: list[MwgRegion] = []
        for region in regions:
            center_x, center_y, box_width, box_height = normalize_region(region.bbox, region.image_size)
            area = XmpArea(h=box_height, w=box_width, x=center_x, y=center_y, unit="normalized")
            xmp_regions.append(MwgRegion(area=area, name=region.label, type_="Object"))

        metadata.region_info = MwgRegionInfo(
            applied_to_dimensions=Dimensions(h=float(height), w=float(width), unit="pixel"),
            region_list=xmp_regions,
        )

    metadata.save()
    logger.info("EXIF write succeeded for {}", target)


def _normalize_xmp_values(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def extract_annotation_snapshot(path: Path) -> dict[str, list[str] | bool]:
    combined_keywords: list[str] = []
    has_regions = False

    try:
        with pyexiv2.Image(str(path)) as image:
            xmp = image.read_xmp()
    except Exception as exc:  # pragma: no cover - best effort helper
        logger.debug("Failed to inspect XMP for {}: {}", path, exc)
        return {"keywords": combined_keywords, "has_regions": has_regions}

    for key in ("Xmp.dc.subject", "Xmp.lr.hierarchicalSubject"):
        for keyword in _normalize_xmp_values(xmp.get(key)):
            if keyword not in combined_keywords:
                combined_keywords.append(keyword)

    has_regions = any(key.startswith("Xmp.mwg-rs.") for key in xmp)

    return {"keywords": combined_keywords, "has_regions": has_regions}


def reset_metadata(target: Path) -> None:
    logger.info("Resetting Snipe-managed XMP metadata for {}", target)

    changed = False
    try:
        # exifmwg does not reliably clear hierarchical keywords/regions on reset,
        # so reset uses pyexiv2 directly until https://github.com/stumpylog/exifmwg/issues/62 is resolved.
        with pyexiv2.Image(str(target)) as image:
            xmp_keys = image.read_xmp().keys()
            payload = {
                "Xmp.dc.subject": None,
                "Xmp.lr.hierarchicalSubject": None,
            }
            payload.update({key: None for key in xmp_keys if key.startswith("Xmp.mwg-rs.")})
            image.modify_xmp(
                payload
            )
        changed = True
    except Exception as exc:
        logger.debug("pyexiv2 embedded XMP reset skipped for {}: {}", target, exc)

    if changed:
        logger.info("Reset write succeeded for {}", target)
        return

    logger.info("No Snipe-managed XMP fields found for {}", target)


def extract_regions_from_xmp(path: Path, image_size: tuple[int, int]) -> list[dict[str, Any]]:
    overlays: list[dict[str, Any]] = []

    try:
        metadata = ImageMetadata(str(path))
    except Exception as exc:  # pragma: no cover - preview best effort
        logger.debug("Failed to read metadata for {}: {}", path, exc)
        return overlays

    region_info = getattr(metadata, "region_info", None)
    if not region_info or not getattr(region_info, "region_list", None):
        return overlays

    dims = getattr(region_info, "applied_to_dimensions", None)
    base_width, base_height = image_size
    if dims and dims.w and dims.h:
        base_width, base_height = int(dims.w), int(dims.h)

    for region in region_info.region_list:
        area = getattr(region, "area", None)
        if not area:
            continue

        try:
            if getattr(area, "unit", "normalized") == "normalized":
                center_x, center_y, box_width, box_height = area.x, area.y, area.w, area.h
                x1 = (center_x - box_width / 2) * base_width
                y1 = (center_y - box_height / 2) * base_height
                x2 = (center_x + box_width / 2) * base_width
                y2 = (center_y + box_height / 2) * base_height
            else:
                x1 = area.x
                y1 = area.y
                x2 = area.x + area.w
                y2 = area.y + area.h

            left = int(max(0.0, min(float(base_width), x1)))
            top = int(max(0.0, min(float(base_height), y1)))
            right = int(max(0.0, min(float(base_width), x2)))
            bottom = int(max(0.0, min(float(base_height), y2)))
            overlays.append(
                {
                    "label": getattr(region, "name", "") or "Unlabeled",
                    "x": left,
                    "y": top,
                    "width": max(0, right - left),
                    "height": max(0, bottom - top),
                }
            )
        except Exception as exc:  # pragma: no cover - malformed metadata resilience
            logger.debug("Skipping malformed region in {}: {}", path, exc)

    return overlays
