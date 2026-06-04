from __future__ import annotations

from pathlib import Path

import pytest

from snipe.config import config
from snipe.paths import DATA_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ASSET_DIR = REPO_ROOT / "tests" / "assets"
SAMPLE_JPEG_NAMES = (
    "DSC_4626.jpg",
    "DSC_4768.jpg",
    "DSC_4853.jpg",
)
LOW_QUALITY_JPEG_NAME = "low_quality.jpg"
SAMPLE_RAW_NAMES = (
    "DSC_7655.NEF",
)
SAMPLE_TEST_ASSET_NAMES = SAMPLE_JPEG_NAMES + (LOW_QUALITY_JPEG_NAME,) + SAMPLE_RAW_NAMES
SAMPLE_IMAGE_NAMES = SAMPLE_JPEG_NAMES + SAMPLE_RAW_NAMES
EXPECTED_DETECTIONS: dict[str, dict[str, object]] = {
    "DSC_4626.jpg": {
        "species_code": "plapri1",
        "bird_name": "纯色山鹪莺",
        "bbox": (327, 897, 942, 1512),
    },
    "DSC_4768.jpg": {
        "species_code": "bkfbun1",
        "bird_name": "灰头鹀",
        "bbox": (859, 725, 1080, 946),
    },
    "DSC_4853.jpg": {
        "species_code": "lotshr1",
        "bird_name": "棕背伯劳",
        "bbox": (209, 684, 847, 1322),
    },
    "DSC_7655.NEF": {
        "species_code": "rbbmag",
        "bird_name": "红嘴蓝鹊",
        "bbox": (2453, 1494, 4287, 3328),
    },
}
BBOX_TOLERANCE_RATIO = 0.05
METADATA_REGION_TOLERANCE_RATIO = 0.05
REQUIRED_INFERENCE_ASSETS = (
    DATA_DIR / "yolo26x-seg.onnx",
    DATA_DIR / "classifier.onnx",
    DATA_DIR / "labels.txt",
    DATA_DIR / "common_name.csv",
)


@pytest.fixture(autouse=True)
def restore_config_state():
    snapshot = config.model_copy(deep=True)
    config.process.detector_model = DATA_DIR / "yolo26x-seg.onnx"
    config.process.classifier_model = DATA_DIR / "classifier.onnx"
    yield
    config.replace_with(snapshot)


@pytest.fixture(scope="session")
def sample_image_paths() -> list[Path]:
    missing = [SAMPLE_ASSET_DIR / name for name in SAMPLE_IMAGE_NAMES if not (SAMPLE_ASSET_DIR / name).exists()]
    if missing:
        pytest.skip(f"Sample images are missing: {missing}")
    return [SAMPLE_ASSET_DIR / name for name in SAMPLE_IMAGE_NAMES]


@pytest.fixture(scope="session")
def sample_jpeg_paths() -> list[Path]:
    missing = [SAMPLE_ASSET_DIR / name for name in SAMPLE_JPEG_NAMES if not (SAMPLE_ASSET_DIR / name).exists()]
    if missing:
        pytest.skip(f"Sample JPEG images are missing: {missing}")
    return [SAMPLE_ASSET_DIR / name for name in SAMPLE_JPEG_NAMES]


@pytest.fixture(scope="session")
def sample_test_asset_paths() -> list[Path]:
    missing = [SAMPLE_ASSET_DIR / name for name in SAMPLE_TEST_ASSET_NAMES if not (SAMPLE_ASSET_DIR / name).exists()]
    if missing:
        pytest.skip(f"Sample test assets are missing: {missing}")
    return [SAMPLE_ASSET_DIR / name for name in SAMPLE_TEST_ASSET_NAMES]


@pytest.fixture(scope="session")
def low_quality_image_path() -> Path:
    image_path = SAMPLE_ASSET_DIR / LOW_QUALITY_JPEG_NAME
    if not image_path.exists():
        pytest.skip(f"Low-quality JPEG image is missing: {image_path}")
    return image_path


@pytest.fixture(scope="session")
def sample_image_dir(sample_image_paths: list[Path]) -> Path:
    return sample_image_paths[0].parent


@pytest.fixture(scope="session")
def inference_ready(sample_image_paths: list[Path]) -> None:
    missing = [path for path in REQUIRED_INFERENCE_ASSETS if not path.exists()]
    if missing:
        pytest.skip(f"Inference runtime assets are missing: {missing}")
