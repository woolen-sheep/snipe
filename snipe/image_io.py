from __future__ import annotations

from io import BytesIO
from pathlib import Path

import rawpy
from loguru import logger
from PIL import Image

RAW_SUFFIXES = {
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
SUPPORTED_SUFFIXES = {".jpg", ".jpeg"} | RAW_SUFFIXES


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() in SUPPORTED_SUFFIXES else []

    images: list[Path] = []
    for suffix in SUPPORTED_SUFFIXES:
        images.extend(source.rglob(f"*{suffix}"))
    return sorted(images)


def load_image(path: Path) -> Image.Image:
    """Load an image from disk with RAW fallbacks for camera formats."""
    suffix = path.suffix.lower()
    try:
        if suffix in RAW_SUFFIXES:
            try:
                with rawpy.imread(str(path)) as raw:
                    rgb = raw.postprocess(
                        use_camera_wb=True,
                        output_bps=8,
                        no_auto_bright=False,
                        auto_bright_thr=0.01,
                    )
                return Image.fromarray(rgb)
            except Exception as exc:
                logger.warning("rawpy decode failed for {}: {}", path, exc)

        try:
            return Image.open(path)
        except Exception as exc:
            logger.warning("PIL.Image.open failed for {}: {}", path, exc)

        try:
            with rawpy.imread(str(path)) as raw:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    return Image.open(BytesIO(thumb.data))
                if thumb.format == rawpy.ThumbFormat.BITMAP:
                    return Image.fromarray(thumb.data)
                raise RuntimeError("Unknown thumbnail format")
        except Exception as exc:
            logger.error("rawpy thumbnail extraction failed for {}: {}", path, exc)

        raise RuntimeError(f"Failed to load image: {path}")
    except Exception as exc:
        logger.error("All attempts to load image failed for {}: {}", path, exc)
        raise
