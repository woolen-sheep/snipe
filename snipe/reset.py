from __future__ import annotations

import concurrent.futures
import sys
from pathlib import Path

from loguru import logger

from .image_io import collect_images
from .metadata import reset_metadata


def run_reset(source: Path, workers: int) -> None:
    if not source.exists():
        logger.error("Source path does not exist: {}", source)
        sys.exit(1)

    if not source.is_dir() and not source.is_file():
        logger.error("Reset source must be an image file or directory: {}", source)
        sys.exit(1)

    images = collect_images(source)
    if not images:
        logger.error("No supported images found in {}", source)
        sys.exit(1)

    logger.info("Resetting Snipe-managed XMP metadata for {} images under {} with {} workers", len(images), source, workers)

    future_to_image: dict[concurrent.futures.Future[None], Path] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        for image in images:
            future_to_image[executor.submit(reset_metadata, image)] = image

        for future in concurrent.futures.as_completed(future_to_image):
            image = future_to_image[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - log only
                logger.exception("Failed to reset image {}: {}", image, exc)

    logger.info("Reset completed for {} images under {}", len(images), source)
