from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from .onnxruntime_utils import create_inference_session
from .paths import DATA_DIR


BBox = tuple[float, float, float, float]


def resolve_detector_model_path(model_path: Path) -> Path:
    expanded_path = model_path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path

    if expanded_path.parent == Path("."):
        return DATA_DIR / expanded_path.name

    return expanded_path


def ensure_local_detector_model(model_path: Path) -> Path:
    resolved_path = resolve_detector_model_path(model_path).resolve()
    if resolved_path.suffix.lower() != ".onnx":
        raise ValueError(f"Detector model must be a local .onnx file: {resolved_path}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"Detector model not found at {resolved_path}. Provide a local ONNX file.")
    return resolved_path


def parse_model_names(raw_names: str | None) -> dict[int, str]:
    if not raw_names:
        return {}

    parsed = ast.literal_eval(raw_names)
    if isinstance(parsed, dict):
        return {int(key): str(value) for key, value in parsed.items()}
    if isinstance(parsed, list):
        return {index: str(value) for index, value in enumerate(parsed)}
    return {}


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    overlap_x1 = max(ax1, bx1)
    overlap_y1 = max(ay1, by1)
    overlap_x2 = min(ax2, bx2)
    overlap_y2 = min(ay2, by2)
    return bbox_area((overlap_x1, overlap_y1, overlap_x2, overlap_y2))


def drop_heavily_covered_smaller_boxes(detections: list[BBox], overlap_threshold: float = 0.7) -> list[BBox]:
    kept = list(detections)
    changed = True
    while changed:
        changed = False
        for left_idx in range(len(kept)):
            left_box = kept[left_idx]
            left_area = bbox_area(left_box)
            if left_area <= 0:
                continue

            for right_idx in range(len(kept)):
                if left_idx == right_idx:
                    continue

                right_box = kept[right_idx]
                right_area = bbox_area(right_box)
                if right_area <= left_area:
                    continue

                overlap = intersection_area(left_box, right_box)
                overlap_ratio = overlap / left_area
                logger.debug(
                    "Compare boxes for coverage drop: smaller_box={} area={:.2f}, larger_box={} area={:.2f}, overlap_area={:.2f}, overlap_ratio={:.6f}, threshold={:.6f}",
                    left_box,
                    left_area,
                    right_box,
                    right_area,
                    overlap,
                    overlap_ratio,
                    overlap_threshold,
                )
                if overlap_ratio >= overlap_threshold:
                    logger.debug(
                        "Dropping smaller box {} because {:.6f} of its area is covered by larger box {}",
                        left_box,
                        overlap_ratio,
                        right_box,
                    )
                    del kept[left_idx]
                    changed = True
                    break

            if changed:
                break

    return kept


def filter_tiny_boxes(detections: list[BBox], min_relative_area: float = 0.05) -> list[BBox]:
    if not detections:
        return detections

    max_area = max(bbox_area(bbox) for bbox in detections)
    if max_area <= 0:
        return []

    min_area = max_area * min_relative_area
    kept: list[BBox] = []
    for bbox in detections:
        area = bbox_area(bbox)
        if area >= min_area:
            kept.append(bbox)
            continue

        logger.debug(
            "Dropping tiny box {} because area {:.2f} is below minimum {:.2f} (ratio threshold {:.6f}, max_area {:.2f})",
            bbox,
            area,
            min_area,
            min_relative_area,
            max_area,
        )

    return kept


def deduplicate_bird_boxes(
    detections: list[BBox],
    overlap_threshold: float = 0.7,
    min_relative_area: float = 0.05,
) -> list[BBox]:
    logger.debug(
        "Starting bird box deduplication with {} boxes, overlap_threshold={:.6f}, min_relative_area={:.6f}",
        len(detections),
        overlap_threshold,
        min_relative_area,
    )
    for idx, bbox in enumerate(detections, 1):
        logger.debug("Raw bird box {}: {} area={:.2f}", idx, bbox, bbox_area(bbox))

    filtered = drop_heavily_covered_smaller_boxes(detections, overlap_threshold=overlap_threshold)
    logger.debug("After coverage-based drop: {} boxes remain", len(filtered))
    for idx, bbox in enumerate(filtered, 1):
        logger.debug("Coverage-filtered box {}: {} area={:.2f}", idx, bbox, bbox_area(bbox))

    final_boxes = filter_tiny_boxes(filtered, min_relative_area=min_relative_area)
    logger.debug("After tiny-box filter: {} boxes remain", len(final_boxes))
    for idx, bbox in enumerate(final_boxes, 1):
        logger.debug("Final bird box {}: {} area={:.2f}", idx, bbox, bbox_area(bbox))

    return final_boxes


class Detector:
    def __init__(self, model_path: Path, overlap_threshold: float = 0.7, min_relative_area: float = 0.05) -> None:
        self.model_path = ensure_local_detector_model(model_path)
        self.overlap_threshold = overlap_threshold
        self.min_relative_area = min_relative_area
        self.session = create_inference_session(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.names = parse_model_names(metadata.get("names"))
        self.bird_class_id = next((class_id for class_id, name in self.names.items() if name.lower() == "bird"), None)
        self.end2end = metadata.get("end2end", "False").lower() == "true"
        self.stride = int(metadata.get("stride", "32"))

        imgsz = metadata.get("imgsz", "[640, 640]")
        parsed_imgsz = ast.literal_eval(imgsz)
        if isinstance(parsed_imgsz, (list, tuple)) and len(parsed_imgsz) == 2:
            self.reference_height = int(parsed_imgsz[0])
            self.reference_width = int(parsed_imgsz[1])
        else:
            self.reference_height = 640
            self.reference_width = 640

        if not self.end2end:
            raise RuntimeError(
                f"Unsupported detector model at {self.model_path}: expected an end-to-end exported YOLO ONNX model."
            )
        if self.bird_class_id is None:
            raise RuntimeError(f"Detector model at {self.model_path} does not expose a 'bird' class in ONNX metadata.")

    def _load_image(self, image: Path | np.ndarray) -> np.ndarray:
        if isinstance(image, Path):
            with Image.open(image) as img:
                return np.asarray(img.convert("RGB"))

        if image.ndim != 3:
            raise ValueError(f"Expected HWC image array, got shape {image.shape}")
        if image.shape[2] == 3:
            return image
        raise ValueError(f"Expected 3 image channels, got shape {image.shape}")

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int], float, int, int]:
        height, width = image.shape[:2]
        scale = min(self.reference_width / width, self.reference_height / height)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))

        resized = np.asarray(Image.fromarray(image).resize((resized_width, resized_height), Image.BILINEAR))
        padded_width = int(math.ceil(resized_width / self.stride) * self.stride)
        padded_height = int(math.ceil(resized_height / self.stride) * self.stride)
        pad_left = (padded_width - resized_width) // 2
        pad_top = (padded_height - resized_height) // 2

        canvas = np.full((padded_height, padded_width, 3), 114, dtype=np.uint8)
        canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        return tensor, (width, height), scale, pad_left, pad_top

    def _scale_bbox(self, bbox: np.ndarray, original_size: tuple[int, int], scale: float, pad_left: int, pad_top: int) -> BBox:
        width, height = original_size
        x1 = max(0.0, min(float(width), (float(bbox[0]) - pad_left) / scale))
        y1 = max(0.0, min(float(height), (float(bbox[1]) - pad_top) / scale))
        x2 = max(0.0, min(float(width), (float(bbox[2]) - pad_left) / scale))
        y2 = max(0.0, min(float(height), (float(bbox[3]) - pad_top) / scale))
        return x1, y1, x2, y2

    def detect_birds(self, image: Path | np.ndarray, conf: float) -> tuple[list[BBox], tuple[int, int]]:
        image_array = self._load_image(image)
        input_tensor, original_size, scale, pad_left, pad_top = self._preprocess(image_array)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        if not outputs:
            return [], original_size

        predictions = outputs[0]
        if predictions.ndim != 3 or predictions.shape[2] < 6:
            raise RuntimeError(
                f"Unsupported detector output shape {predictions.shape} for model {self.model_path}."
            )

        rows = predictions[0]
        detections: list[BBox] = []
        passing_rows = 0
        logger.debug(
            "ONNX detector returned {} candidate rows for input with size {}x{} at conf={:.3f}",
            len(rows),
            original_size[0],
            original_size[1],
            conf,
        )

        for idx, row in enumerate(rows):
            score = float(row[4])
            class_id = int(round(float(row[5])))
            if score < conf or class_id != self.bird_class_id:
                continue

            passing_rows += 1
            name = self.names.get(class_id, str(class_id))
            candidate_bbox = self._scale_bbox(row[:4], original_size, scale, pad_left, pad_top)
            logger.debug(
                "ONNX bird box {}: class={} score={:.6f} bbox={}",
                idx + 1,
                name,
                score,
                candidate_bbox,
            )
            detections.append(candidate_bbox)

        logger.debug(
            "ONNX detector kept {} bird candidate rows after confidence/class filtering",
            passing_rows,
        )

        return deduplicate_bird_boxes(
            detections,
            overlap_threshold=self.overlap_threshold,
            min_relative_area=self.min_relative_area,
        ), original_size
