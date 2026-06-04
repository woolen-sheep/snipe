from __future__ import annotations

import os
from importlib import metadata
from importlib.metadata import PackageNotFoundError

from loguru import logger

try:
    import onnxruntime as ort
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError(
        "ONNX Runtime is not installed. Install Snipe with `snipe[cpu]` or `snipe[gpu]` to enable inference."
    ) from exc


def detect_onnxruntime_package() -> str:
    try:
        metadata.version("onnxruntime-gpu")
    except PackageNotFoundError:
        pass
    else:
        return "gpu"

    try:
        metadata.version("onnxruntime")
    except PackageNotFoundError:
        return "unknown"

    return "cpu"


def get_inference_providers() -> list[str]:
    runtime_package = detect_onnxruntime_package()
    available = ort.get_available_providers()

    if runtime_package == "gpu":
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info(
                "Using ONNX Runtime GPU package with providers {}",
                providers,
            )
            return providers

        logger.warning(
            "Detected onnxruntime-gpu package, but CUDAExecutionProvider is unavailable. Falling back to CPUExecutionProvider. Available providers: {}",
            available,
        )
        return ["CPUExecutionProvider"]

    if runtime_package == "cpu":
        logger.info("Using ONNX Runtime CPU package with CPUExecutionProvider")
        return ["CPUExecutionProvider"]

    logger.warning(
        "Could not determine whether the installed ONNX Runtime package is CPU or GPU. Falling back to CPUExecutionProvider. Available providers: {}",
        available,
    )
    return ["CPUExecutionProvider"]


def create_session_options() -> ort.SessionOptions:
    session_options = ort.SessionOptions()
    raw_level = os.environ.get("SNIPE_ORT_LOG_SEVERITY_LEVEL", "3")

    try:
        session_options.log_severity_level = int(raw_level)
    except ValueError:
        logger.warning(
            "Invalid SNIPE_ORT_LOG_SEVERITY_LEVEL={!r}; expected integer severity level. Using warning-suppressed default level 3.",
            raw_level,
        )
        session_options.log_severity_level = 3
    return session_options


def create_inference_session(model_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(
        model_path,
        sess_options=create_session_options(),
        providers=get_inference_providers(),
    )
