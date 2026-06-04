from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from loguru import logger

from .gps_metadata import extract_gps_coordinates


@dataclass(frozen=True)
class GPSFilterInfo:
    latitude: float
    longitude: float
    region_code: str | None
    species_count: int
    used_fallback: bool


@dataclass(frozen=True)
class RegionBounds:
    region_code: str
    south: float
    north: float
    west: float
    east: float
    area: float


class InMemorySpeciesFilter:
    def __init__(self, data_path: Path, labels_path: Path) -> None:
        self.data_path = data_path
        self.labels_path = labels_path
        self._labels = self._load_labels(labels_path) if labels_path.exists() else []
        self._region_bounds: tuple[RegionBounds, ...] = ()
        self._region_to_indices: dict[str, np.ndarray] = {}
        self._region_to_bounds: dict[str, tuple[float, float, float, float]] = {}
        self._available = False

        if self.data_path.exists() and self.labels_path.exists():
            self._load()

    @staticmethod
    def _load_labels(labels_path: Path) -> list[str]:
        with labels_path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def _load(self) -> None:
        payload = json.loads(self.data_path.read_text(encoding="utf-8"))
        label_to_index = {label: idx for idx, label in enumerate(self._labels)}

        bounds: list[RegionBounds] = []
        for item in payload.get("region_bounds", []):
            region_code = str(item["region_code"]).upper()
            south = float(item["south"])
            north = float(item["north"])
            west = float(item["west"])
            east = float(item["east"])
            area = float(item.get("area", (north - south) * (east - west)))
            bounds.append(RegionBounds(region_code, south, north, west, east, area))
            self._region_to_bounds[region_code] = (south, north, west, east)

        missing_codes = 0
        region_to_indices: dict[str, np.ndarray] = {}
        for region_code, species_codes in payload.get("region_species", {}).items():
            normalized_region_code = str(region_code).upper()
            indices = sorted({label_to_index[code] for code in species_codes if code in label_to_index})
            missing_codes += sum(1 for code in species_codes if code not in label_to_index)
            if indices:
                region_to_indices[normalized_region_code] = np.asarray(indices, dtype=np.int32)

        bounds.sort(key=lambda item: item.area)
        self._region_bounds = tuple(bounds)
        self._region_to_indices = region_to_indices
        self._available = True

        logger.info(
            "Loaded in-memory GPS species filter from {}: {} bounds, {} regions with labels, {} classifier labels, {} unmapped species codes skipped",
            self.data_path,
            len(self._region_bounds),
            len(self._region_to_indices),
            len(self._labels),
            missing_codes,
        )

    def is_available(self) -> bool:
        return self._available

    def get_allowed_indices_by_region(self, region_code: str) -> np.ndarray | None:
        return self._region_to_indices.get(region_code.upper())

    def get_region_bounds(self, region_code: str) -> tuple[float, float, float, float] | None:
        return self._region_to_bounds.get(region_code.upper())

    def describe_allowed_indices(self, allowed_indices: Sequence[int] | np.ndarray, limit: int = 10) -> list[str]:
        indices = np.asarray(allowed_indices, dtype=np.intp)
        sample_count = min(limit, int(indices.size))
        return [self._labels[int(index)] for index in indices[:sample_count] if 0 <= int(index) < len(self._labels)]

    def get_allowed_indices_by_gps(self, latitude: float, longitude: float) -> tuple[np.ndarray | None, str | None, bool]:
        candidates = [
            bounds.region_code
            for bounds in self._region_bounds
            if bounds.south <= latitude <= bounds.north and bounds.west <= longitude <= bounds.east
        ]

        logger.debug(
            "GPS lookup for ({:.6f}, {:.6f}) matched {} region candidates: {}",
            latitude,
            longitude,
            len(candidates),
            candidates[:10],
        )

        for region_code in candidates:
            allowed_indices = self.get_allowed_indices_by_region(region_code)
            if allowed_indices is not None and allowed_indices.size > 0:
                bounds = self.get_region_bounds(region_code)
                logger.debug(
                    "Using region {} for ({:.6f}, {:.6f}); bounds={}; label_count={}",
                    region_code,
                    latitude,
                    longitude,
                    bounds,
                    int(allowed_indices.size),
                )
                return allowed_indices, region_code, False

            if "-" in region_code:
                country_code = region_code.split("-", 1)[0]
                fallback_indices = self.get_allowed_indices_by_region(country_code)
                if fallback_indices is not None and fallback_indices.size > 0:
                    bounds = self.get_region_bounds(region_code)
                    logger.debug(
                        "Falling back from region {} to country {} for ({:.6f}, {:.6f}); region_bounds={}; label_count={}",
                        region_code,
                        country_code,
                        latitude,
                        longitude,
                        bounds,
                        int(fallback_indices.size),
                    )
                    return fallback_indices, country_code, True

        logger.debug("GPS lookup for ({:.6f}, {:.6f}) did not resolve any species list", latitude, longitude)
        return None, None, False


class GPSBirdSpeciesFilter:
    def __init__(self, data_path: Path, labels_path: Path) -> None:
        self.backend = InMemorySpeciesFilter(data_path, labels_path)

    def is_available(self) -> bool:
        return self.backend.is_available()

    def describe_allowed_indices(self, allowed_indices: Sequence[int] | np.ndarray, limit: int = 10) -> list[str]:
        return self.backend.describe_allowed_indices(allowed_indices, limit=limit)

    def get_allowed_indices_for_region(self, region_code: str) -> tuple[np.ndarray | None, GPSFilterInfo | None]:
        if not self.backend.is_available():
            return None, None

        normalized_region_code = region_code.upper()
        allowed_indices = self.backend.get_allowed_indices_by_region(normalized_region_code)
        if allowed_indices is None or allowed_indices.size == 0:
            logger.debug("No location-filtered labels found for manual region {}", normalized_region_code)
            return None, None

        info = GPSFilterInfo(
            latitude=0.0,
            longitude=0.0,
            region_code=normalized_region_code,
            species_count=int(allowed_indices.size),
            used_fallback=False,
        )
        logger.debug(
            "Loaded {} location-filtered labels for manual region {}",
            info.species_count,
            info.region_code,
        )
        return allowed_indices, info

    def get_allowed_indices(self, image_path: Path) -> tuple[np.ndarray | None, GPSFilterInfo | None]:
        if not self.backend.is_available():
            return None, None

        coordinates = extract_gps_coordinates(image_path)
        if coordinates is None:
            logger.debug("No GPS coordinates found for {}", image_path)
            return None, None

        latitude, longitude = coordinates
        allowed_indices, region_code, used_fallback = self.backend.get_allowed_indices_by_gps(latitude, longitude)
        if allowed_indices is None or allowed_indices.size == 0:
            logger.debug(
                "No location-filtered labels found for {} at ({:.6f}, {:.6f})",
                image_path,
                latitude,
                longitude,
            )
            return None, GPSFilterInfo(
                latitude=latitude,
                longitude=longitude,
                region_code=region_code,
                species_count=0,
                used_fallback=used_fallback,
            )

        info = GPSFilterInfo(
            latitude=latitude,
            longitude=longitude,
            region_code=region_code,
            species_count=int(allowed_indices.size),
            used_fallback=used_fallback,
        )
        logger.debug(
            "Loaded {} location-filtered labels for {} using region {}{}",
            info.species_count,
            image_path,
            info.region_code,
            " (country fallback)" if info.used_fallback else "",
        )
        return allowed_indices, info
