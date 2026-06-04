from __future__ import annotations

import concurrent.futures
import random
import stat
import shutil
import sys
import time
from pathlib import Path

from loguru import logger

from .config import config
from .image_io import collect_images
from .inference import detect_birds, get_classifier, get_detector, get_gps_species_filter, load_taxonomy
from .metadata import Region, write_metadata
from .progress import TerminalProgressReporter


GPU_RETRY_EXPONENTIAL_STEPS = 6
GPU_RETRY_BASE_DELAY_SECONDS = 0.5
MAX_TRANSIENT_GPU_RETRY_ATTEMPTS = 3
_GPU_ERROR_MARKERS = (
    "cuda",
    "cudnn",
    "culaunchkernel",
    "cudaexecutionprovider",
    "cudnn_fe",
    "onnxruntimeerror",
    "non-zero status code returned while running",
    "execution failed",
    "backend_api_failed",
)
_GPU_OVERLOAD_ERROR_MARKERS = (
    "too many resources requested for launch",
)


def copy_for_output(source: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return source
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / source.name
    shutil.copy2(source, target)
    target.chmod(target.stat().st_mode | stat.S_IWRITE)
    return target


def is_gpu_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _GPU_ERROR_MARKERS)


def is_infinite_retry_gpu_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _GPU_OVERLOAD_ERROR_MARKERS)


def compute_gpu_retry_delay(attempt: int) -> float:
    capped_attempt = max(1, min(attempt, GPU_RETRY_EXPONENTIAL_STEPS))
    base_delay = GPU_RETRY_BASE_DELAY_SECONDS * (2 ** (capped_attempt - 1))
    if attempt <= GPU_RETRY_EXPONENTIAL_STEPS:
        return base_delay
    return base_delay + (base_delay * 0.5 * random.random())


def _detect_regions_with_retry(image: Path) -> list[Region]:
    process_config = config.process

    attempt = 0
    while True:
        attempt += 1
        try:
            detections = detect_birds(
                image,
                region_code=process_config.region_filter,
                language_code=process_config.language,
                conf=process_config.conf,
                detector_model=process_config.detector_model,
                detector_merge_overlap_threshold=process_config.detector_merge_overlap_threshold,
                detector_min_box_area_ratio=process_config.detector_min_box_area_ratio,
                gps_filter=process_config.gps_filter,
                gps_filter_data=process_config.gps_filter_data,
                classifier_model=process_config.classifier_model,
                labels=process_config.labels,
                taxonomy=process_config.taxonomy,
            )
            return [
                Region(
                    label=detection.bird_name,
                    score=detection.score,
                    bbox=detection.bbox.as_tuple(),
                    image_size=(detection.image_width, detection.image_height),
                )
                for detection in detections
            ]
        except Exception as exc:
            if not is_gpu_error(exc):
                raise

            if not is_infinite_retry_gpu_error(exc) and attempt >= MAX_TRANSIENT_GPU_RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"GPU inference failed for {image} after {MAX_TRANSIENT_GPU_RETRY_ATTEMPTS} attempts: {exc}"
                ) from exc

            delay_seconds = compute_gpu_retry_delay(attempt)
            logger.warning(
                "Retryable GPU inference failure on {} (attempt {}): {}. Backing off for {:.1f}s before retrying.",
                image,
                attempt,
                exc,
                delay_seconds,
            )
            time.sleep(delay_seconds)


def process_image(image: Path) -> tuple[Path, list[Region]]:
    process_config = config.process
    regions = _detect_regions_with_retry(image)

    target = copy_for_output(image, process_config.output_dir)
    if process_config.dry_run:
        logger.info("Dry run: skipping metadata write for {}", target)
        return target, regions

    keywords = [region.label for region in regions]
    if process_config.bird_keyword:
        keywords.append(process_config.bird_keyword)
    write_metadata(target, regions, keywords)
    return target, regions


def run_process() -> None:
    process_config = config.process
    progress_enabled = process_config.log_level is None

    if process_config.source is None:
        _emit_process_error(progress_enabled, "Source path is not configured")
        sys.exit(1)

    if not process_config.source.exists():
        _emit_process_error(progress_enabled, "Source path does not exist: {}", process_config.source)
        sys.exit(1)

    images = collect_images(process_config.source)
    if not images:
        _emit_process_error(progress_enabled, "No supported images found in {}", process_config.source)
        sys.exit(1)

    load_taxonomy(process_config.taxonomy, process_config.language)
    logger.info("Found {} images to process", len(images))

    reporter: TerminalProgressReporter | None = None

    try:
        get_detector(
            process_config.detector_model,
            overlap_threshold=process_config.detector_merge_overlap_threshold,
            min_relative_area=process_config.detector_min_box_area_ratio,
        )
        get_classifier(process_config.classifier_model, process_config.labels)
        if process_config.gps_filter:
            get_gps_species_filter(process_config.gps_filter_data, process_config.labels)

        if progress_enabled:
            reporter = TerminalProgressReporter(total=len(images), stream=sys.stderr)
            reporter.render(image=images[0], status="Queued")

        future_to_image: dict[concurrent.futures.Future[tuple[Path, list[Region]]], Path] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=process_config.workers) as executor:
            for image in images:
                future_to_image[executor.submit(process_image, image)] = image

            for future in concurrent.futures.as_completed(future_to_image):
                image = future_to_image[future]
                try:
                    target, regions = future.result()
                except Exception as exc:  # pragma: no cover - log only
                    if reporter is not None:
                        reporter.advance(image, (), status=f"Failed: {type(exc).__name__}")
                    else:
                        logger.exception("Failed to process image: {}", exc)
                    continue

                if reporter is not None:
                    reporter.advance(target, [region.label for region in regions])

                if not regions:
                    continue

                logger.info("Wrote {} regions to {}", len(regions), target)
    finally:
        if reporter is not None:
            reporter.close()


def _emit_process_error(progress_enabled: bool, message: str, *args: object) -> None:
    if progress_enabled:
        print(message.format(*args), file=sys.stderr)
        return
    logger.error(message, *args)
