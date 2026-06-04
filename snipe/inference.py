from __future__ import annotations

import csv
import threading
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Literal
from urllib.parse import urlparse
from urllib.request import url2pathname, urlopen

import numpy as np
from PIL import Image
from loguru import logger

from .classifier import Classifier
from .config import config
from .detector import Detector, resolve_detector_model_path
from .image_io import load_image
from .model_manager import ensure_managed_model, is_managed_model_path
from .species_filter import GPSBirdSpeciesFilter


ImageInput = str | Path | bytes | bytearray | memoryview | IO[bytes] | Image.Image | np.ndarray
LanguageCode = Literal["CN", "US"]


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


@dataclass(frozen=True, slots=True)
class BirdDetection:
    species_code: str
    bird_name: str
    score: float
    bbox: BoundingBox
    image_width: int
    image_height: int


_detector_cache: dict[tuple[Path, float, float], Detector] = {}
_classifier_cache: dict[tuple[Path, Path], Classifier] = {}
_gps_species_filter_cache: dict[tuple[Path, Path], GPSBirdSpeciesFilter] = {}
_taxonomy_cache: dict[tuple[Path, str], dict[str, str]] = {}
_detector_cache_lock = threading.Lock()
_classifier_cache_lock = threading.Lock()
_gps_species_filter_cache_lock = threading.Lock()
_taxonomy_cache_lock = threading.Lock()


def _resolve_runtime_model_path(logical_name: str, model_path: Path) -> Path:
    expanded_path = model_path.expanduser()
    if is_managed_model_path(logical_name, expanded_path):
        return ensure_managed_model(logical_name)
    return expanded_path


def get_detector(model_path: Path, overlap_threshold: float = 0.7, min_relative_area: float = 0.05) -> Detector:
    key = (resolve_detector_model_path(_resolve_runtime_model_path("detector", model_path)).resolve(), overlap_threshold, min_relative_area)
    if key not in _detector_cache:
        with _detector_cache_lock:
            if key not in _detector_cache:
                logger.info("Loading detector model from {}", key[0])
                _detector_cache[key] = Detector(key[0], overlap_threshold=overlap_threshold, min_relative_area=min_relative_area)
    return _detector_cache[key]


def get_classifier(model_path: Path, labels_path: Path) -> Classifier:
    resolved_model_path = _resolve_runtime_model_path("classifier", model_path)
    key = (resolved_model_path.resolve(), labels_path.resolve())
    if key not in _classifier_cache:
        with _classifier_cache_lock:
            if key not in _classifier_cache:
                logger.info("Loading classifier model from {}", key[0])
                _classifier_cache[key] = Classifier(*key)
    return _classifier_cache[key]


def get_gps_species_filter(data_path: Path, labels_path: Path) -> GPSBirdSpeciesFilter:
    key = (data_path.resolve(), labels_path.resolve())
    if key not in _gps_species_filter_cache:
        with _gps_species_filter_cache_lock:
            if key not in _gps_species_filter_cache:
                logger.info("Loading GPS species filter from {} using labels {}", key[0], key[1])
                _gps_species_filter_cache[key] = GPSBirdSpeciesFilter(*key)
    return _gps_species_filter_cache[key]


def to_square(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    box_width = x2 - x1
    box_height = y2 - y1
    size = max(box_width, box_height)
    center_x = x1 + box_width / 2
    center_y = y1 + box_height / 2
    half = size / 2

    new_x1 = max(0.0, center_x - half)
    new_y1 = max(0.0, center_y - half)
    new_x2 = min(float(width), center_x + half)
    new_y2 = min(float(height), center_y + half)
    return tuple(map(int, (new_x1, new_y1, new_x2, new_y2)))


def normalize_language_code(language_code: str | None) -> LanguageCode:
    normalized = (language_code or config.process.language).strip().upper()
    if normalized not in {"CN", "US"}:
        raise ValueError(f"Unsupported language code: {language_code!r}")
    return normalized  # type: ignore[return-value]


def load_taxonomy(taxonomy_path: Path, language: str) -> dict[str, str]:
    normalized_language = normalize_language_code(language)
    key = (taxonomy_path.resolve(), normalized_language)
    if key in _taxonomy_cache:
        return _taxonomy_cache[key]

    with _taxonomy_cache_lock:
        if key in _taxonomy_cache:
            return _taxonomy_cache[key]

        mapping: dict[str, str] = {}
        if not taxonomy_path.exists():
            logger.warning("common-name CSV not found at {} - labels will remain species codes", taxonomy_path)
            _taxonomy_cache[key] = mapping
            return mapping

        language_column = "COMMON_NAME_ZH_CN" if normalized_language == "CN" else "COMMON_NAME_US"

        with taxonomy_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = (row.get("SPECIES_CODE") or row.get("species_code") or "").strip()
                common_name = (row.get(language_column) or row.get(language_column.lower()) or "").strip()
                if code and common_name:
                    mapping[code.lower()] = common_name

        logger.info("Loaded {} common-name entries from {} using language {}", len(mapping), taxonomy_path, normalized_language)
        _taxonomy_cache[key] = mapping
        return mapping


def resolve_common_name(species_code: str, taxonomy: dict[str, str]) -> str:
    if not species_code:
        return species_code
    return taxonomy.get(species_code.lower(), species_code)


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    return Path(path).expanduser()


def _describe_image_source(image: ImageInput) -> str:
    if isinstance(image, (str, Path)):
        return str(image)
    if isinstance(image, (bytes, bytearray, memoryview)):
        return "<image bytes>"
    if hasattr(image, "read"):
        return "<file-like image>"
    return "<in-memory image>"


def _is_supported_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme.lower() in {"http", "https", "file"}


def _load_image_from_url(url: str) -> tuple[np.ndarray, Path | None]:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    if scheme == "file":
        file_path = Path(url2pathname(parsed.path)).expanduser()
        loaded = load_image(file_path)
        rgb = loaded.convert("RGB")
        loaded.close()
        try:
            return np.array(rgb), file_path
        finally:
            rgb.close()

    with urlopen(url, timeout=30) as response:
        payload = response.read()

    with Image.open(BytesIO(payload)) as loaded:
        rgb = loaded.convert("RGB")
        try:
            return np.array(rgb), None
        finally:
            rgb.close()


def _load_image_from_bytes(payload: bytes | bytearray | memoryview) -> tuple[np.ndarray, Path | None]:
    with Image.open(BytesIO(bytes(payload))) as loaded:
        rgb = loaded.convert("RGB")
        try:
            return np.array(rgb), None
        finally:
            rgb.close()


def _is_file_like_image(image: object) -> bool:
    return hasattr(image, "read")


def _load_image_from_file_like(image: IO[bytes] | object) -> tuple[np.ndarray, Path | None]:
    stream = image
    current_position = None
    if hasattr(stream, "tell"):
        try:
            current_position = stream.tell()
        except Exception:
            current_position = None

    payload = stream.read()
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ValueError(f"Expected file-like object to return bytes from read(), got {type(payload)!r}")

    if current_position is not None and hasattr(stream, "seek"):
        try:
            stream.seek(current_position)
        except Exception:
            pass

    return _load_image_from_bytes(payload)


def _coerce_image_input(image: ImageInput) -> tuple[np.ndarray, Path | None]:
    if isinstance(image, str) and _is_supported_url(image):
        return _load_image_from_url(image)

    if isinstance(image, (bytes, bytearray, memoryview)):
        return _load_image_from_bytes(image)

    if isinstance(image, (str, Path)):
        image_path = Path(image).expanduser()
        loaded = load_image(image_path)
        rgb = loaded.convert("RGB")
        loaded.close()
        try:
            return np.array(rgb), image_path
        finally:
            rgb.close()

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        try:
            return np.array(rgb), None
        finally:
            rgb.close()

    if _is_file_like_image(image):
        return _load_image_from_file_like(image)

    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(f"Expected an HWC RGB image array with 3 channels, got shape {image_array.shape}")
    if image_array.dtype != np.uint8:
        image_array = image_array.astype(np.uint8)
    return image_array, None


def _resolve_allowed_label_indices(
    image_path: Path | None,
    region_code: str | None,
    gps_filter: bool,
    gps_filter_data: Path,
    labels: Path,
) -> np.ndarray | None:
    if not gps_filter and not region_code:
        return None

    species_filter = get_gps_species_filter(gps_filter_data, labels)
    if not species_filter.is_available():
        logger.debug("GPS species filter data is unavailable at {}", gps_filter_data)
        return None

    if region_code:
        normalized_region_code = region_code.upper()
        logger.debug("Using manual GPS filter region override: region_code={}", normalized_region_code)
        allowed_label_indices, gps_filter_info = species_filter.get_allowed_indices_for_region(normalized_region_code)
        if gps_filter_info is not None:
            logger.debug(
                "Manual GPS filter info: region={}, species_count={}",
                gps_filter_info.region_code,
                gps_filter_info.species_count,
            )
        if allowed_label_indices is None or len(allowed_label_indices) == 0:
            logger.debug("Manual GPS filter region {} did not resolve any species codes", normalized_region_code)
            return None

        sample_species_codes = species_filter.describe_allowed_indices(allowed_label_indices, limit=10)
        logger.debug(
            "GPS filter will constrain classification to {} label indices; sample_labels={}",
            len(allowed_label_indices),
            sample_species_codes,
        )
        return allowed_label_indices

    if image_path is None:
        logger.debug("GPS filter requested for in-memory image without a manual region; classifier will use the full label set")
        return None

    allowed_label_indices, gps_filter_info = species_filter.get_allowed_indices(image_path)
    if gps_filter_info is not None:
        logger.debug(
            "GPS filter info for {}: lat={:.6f}, lon={:.6f}, region={}, species_count={}, fallback={}",
            image_path,
            gps_filter_info.latitude,
            gps_filter_info.longitude,
            gps_filter_info.region_code,
            gps_filter_info.species_count,
            gps_filter_info.used_fallback,
        )

    if allowed_label_indices is None or len(allowed_label_indices) == 0:
        logger.debug("GPS filter did not constrain classification for {}; classifier will use the full label set", image_path)
        return None

    sample_species_codes = species_filter.describe_allowed_indices(allowed_label_indices, limit=10)
    logger.debug(
        "GPS filter will constrain classification for {} to {} label indices; sample_labels={}",
        image_path,
        len(allowed_label_indices),
        sample_species_codes,
    )
    return allowed_label_indices


def detect_birds(
    image: ImageInput,
    *,
    region_code: str | None = None,
    language_code: str | None = None,
    conf: float | None = None,
    detector_model: str | Path | None = None,
    detector_merge_overlap_threshold: float | None = None,
    detector_min_box_area_ratio: float | None = None,
    gps_filter: bool = True,
    gps_filter_data: str | Path | None = None,
    classifier_model: str | Path | None = None,
    labels: str | Path | None = None,
    taxonomy: str | Path | None = None,
) -> list[BirdDetection]:
    process_config = config.process
    normalized_language = normalize_language_code(language_code)
    resolved_detector_model = _resolve_path(detector_model, process_config.detector_model)
    resolved_classifier_model = _resolve_path(classifier_model, process_config.classifier_model)
    resolved_labels = _resolve_path(labels, process_config.labels)
    resolved_taxonomy = _resolve_path(taxonomy, process_config.taxonomy)
    resolved_gps_filter_data = _resolve_path(gps_filter_data, process_config.gps_filter_data)
    resolved_conf = process_config.conf if conf is None else conf
    resolved_overlap_threshold = (
        process_config.detector_merge_overlap_threshold
        if detector_merge_overlap_threshold is None
        else detector_merge_overlap_threshold
    )
    resolved_min_box_area_ratio = (
        process_config.detector_min_box_area_ratio
        if detector_min_box_area_ratio is None
        else detector_min_box_area_ratio
    )

    detector = get_detector(
        resolved_detector_model,
        overlap_threshold=resolved_overlap_threshold,
        min_relative_area=resolved_min_box_area_ratio,
    )
    classifier = get_classifier(resolved_classifier_model, resolved_labels)
    taxonomy_map = load_taxonomy(resolved_taxonomy, normalized_language)
    image_array, image_path = _coerce_image_input(image)
    allowed_label_indices = _resolve_allowed_label_indices(
        image_path,
        region_code,
        gps_filter or region_code is not None,
        resolved_gps_filter_data,
        resolved_labels,
    )

    detections, (width, height) = detector.detect_birds(image_array, conf=resolved_conf)
    if not detections:
        logger.warning("No birds found in {}", _describe_image_source(image))
        return []

    rgb = Image.fromarray(image_array)
    try:
        results: list[BirdDetection] = []
        for bbox in detections:
            square = to_square(bbox, width, height)
            crop = rgb.crop(square)
            try:
                species_code, score = classifier.classify(crop, allowed_indices=allowed_label_indices)
            finally:
                crop.close()

            bird_name = resolve_common_name(species_code, taxonomy_map)
            logger.info(
                "Classified region {} in {} as '{}' -> '{}' (score={:.4f}) [box: {}]",
                len(results) + 1,
                _describe_image_source(image),
                species_code,
                bird_name,
                score,
                square,
            )
            results.append(
                BirdDetection(
                    species_code=species_code,
                    bird_name=bird_name,
                    score=score,
                    bbox=BoundingBox(*square),
                    image_width=width,
                    image_height=height,
                )
            )

        return results
    finally:
        rgb.close()


__all__ = ["BirdDetection", "BoundingBox", "ImageInput", "LanguageCode", "detect_birds"]
