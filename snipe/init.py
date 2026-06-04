from __future__ import annotations

from pathlib import Path

from loguru import logger

from .model_manager import initialize_runtime


def run_init(config_path: Path | None = None) -> None:
    resolved_config_path, created_config, resolved_models = initialize_runtime(config_path)

    if created_config:
        logger.info("Wrote default config to {}", resolved_config_path)
    else:
        logger.info("Config already exists at {}; leaving it unchanged", resolved_config_path)

    for logical_name, model_path in resolved_models.items():
        logger.info("{} model ready at {}", logical_name, model_path)
