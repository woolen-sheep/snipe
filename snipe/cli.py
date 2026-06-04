from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError
from loguru import logger

from .config import DEFAULT_CONFIG_PATH, DEFAULT_WORKERS, config
from .init import run_init
from .preview import run_preview
from .processing import run_process
from .reset import run_reset


@dataclass
class ResetCommand:
    command: str
    source: Path
    workers: int = DEFAULT_WORKERS
    active_log_level: str | None = "INFO"


@dataclass
class InitCommand:
    command: str
    config_path: Path
    active_log_level: str | None = "INFO"


def configure_logging(level: str | None) -> None:
    logger.remove()
    logger.enable("snipe")
    if level is None:
        logger.disable("snipe")
        return
    logger.add(sys.stderr, level=level.upper())


def build_parser() -> argparse.ArgumentParser:
    config_parent = argparse.ArgumentParser(add_help=False)
    config_parent.add_argument(
        "--config",
        type=Path,
        default=argparse.SUPPRESS,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG_PATH})",
    )

    parser = argparse.ArgumentParser(description="Detect birds, classify crops, write EXIF regions.", parents=[config_parent])
    subparsers = parser.add_subparsers(dest="command")

    process_parser = subparsers.add_parser(
        "process",
        help="Detect birds, classify, and write EXIF regions",
        parents=[config_parent],
    )
    process_parser.add_argument("source", nargs="?", type=Path, default=argparse.SUPPRESS, help="File or directory containing JPG images")
    process_parser.add_argument("-o", "--output-dir", type=Path, default=argparse.SUPPRESS, help="Optional directory to write new files")
    process_parser.add_argument("--conf", type=float, default=argparse.SUPPRESS, help="Confidence threshold for bird detection")
    process_parser.add_argument("--workers", type=int, default=argparse.SUPPRESS, help="Number of parallel workers")
    process_parser.add_argument(
        "--detector-model",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to a local YOLO ONNX detector model",
    )
    process_parser.add_argument(
        "--detector-merge-overlap-threshold",
        type=float,
        default=argparse.SUPPRESS,
        help="Merge boxes when either box has at least this overlap ratio covered by another box",
    )
    process_parser.add_argument(
        "--detector-min-box-area-ratio",
        type=float,
        default=argparse.SUPPRESS,
        help="Drop boxes smaller than this fraction of the largest detected box",
    )
    process_parser.add_argument(
        "--gps-filter",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable GPS-based bird species filtering when GPS metadata or a manual region filter is available",
    )
    process_parser.add_argument(
        "--gps-filter-data",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to the GPS filter JSON data used for in-memory bird species filtering",
    )
    process_parser.add_argument(
        "--gps-filter-db",
        dest="gps_filter_data",
        type=Path,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    process_parser.add_argument(
        "--region-filter",
        type=str,
        default=argparse.SUPPRESS,
        help="Force filtering by a specific region code such as CN or CN-11",
    )
    process_parser.add_argument(
        "--language",
        choices=("CN", "US"),
        default=argparse.SUPPRESS,
        type=str.upper,
        help="Display common names in the selected language (CN or US)",
    )
    process_parser.add_argument(
        "--classifier-model",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to ONNX classifier model",
    )
    process_parser.add_argument(
        "--labels",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to classifier labels file",
    )
    process_parser.add_argument(
        "--taxonomy",
        type=Path,
        default=argparse.SUPPRESS,
        help="Path to common-name CSV (SPECIES_CODE,COMMON_NAME_ZH_CN,COMMON_NAME_US)",
    )
    process_parser.add_argument(
        "--bird-keyword",
        default=argparse.SUPPRESS,
        help="Keyword to add when a bird is detected; use --no-bird-keyword to disable",
    )
    process_parser.add_argument(
        "--no-bird-keyword",
        action="store_const",
        const=None,
        dest="bird_keyword",
        help="Disable writing the generic bird keyword",
    )
    process_parser.add_argument(
        "--log-level",
        default=argparse.SUPPRESS,
        help="Logging level (DEBUG, INFO, WARN, ERROR). Use null/none for the cross-worker progress bar.",
    )
    process_parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS, help="Run without writing metadata")

    preview_parser = subparsers.add_parser("preview", help="Preview images and their XMP regions", parents=[config_parent])
    preview_parser.add_argument("source", nargs="?", type=Path, default=argparse.SUPPRESS, help="File or directory containing images")
    preview_parser.add_argument(
        "--log-level",
        default=argparse.SUPPRESS,
        help="Logging level (DEBUG, INFO, WARN, ERROR). Use null/none to suppress preview logs.",
    )
    preview_parser.add_argument("--debug", action="store_true", default=argparse.SUPPRESS, help="Enable webview debug/devtools")

    subparsers.add_parser("init", help="Create the default config and pre-download managed model files", parents=[config_parent])

    clear_parser = subparsers.add_parser("clear", help="Clear Snipe-managed XMP metadata from an image file or directory")
    clear_parser.add_argument("source", type=Path, help="Image file or directory containing images to clear")
    clear_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel workers")

    return parser


def parse_args(argv: Sequence[str] | None = None):
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    if not _has_explicit_command(argv_list):
        argv_list = ["process"] + argv_list

    explicit_no_gps_filter = "--no-gps-filter" in argv_list
    parser = build_parser()
    namespace = parser.parse_args(argv_list)

    if getattr(namespace, "command", None) == "clear":
        return ResetCommand(command="clear", source=namespace.source, workers=namespace.workers)

    if getattr(namespace, "command", None) == "init":
        return InitCommand(command="init", config_path=getattr(namespace, "config", DEFAULT_CONFIG_PATH))

    config_path = getattr(namespace, "config", DEFAULT_CONFIG_PATH)

    try:
        config.replace_with(config.from_file(config_path))
        resolved_config = config.replace_with(
            config.apply_cli_overrides(getattr(namespace, "command", "process"), _namespace_to_overrides(namespace))
        )
    except (OSError, ValueError, ValidationError) as exc:
        parser.error(f"Failed to load config from {config_path}: {exc}")

    if resolved_config.command == "process" and resolved_config.process.region_filter:
        if explicit_no_gps_filter:
            parser.error("--region-filter cannot be combined with --no-gps-filter")
        resolved_config.process.gps_filter = True

    if resolved_config.command == "process" and resolved_config.process.source is None:
        parser.error("the following arguments are required: source")

    if resolved_config.command == "preview" and resolved_config.preview.source is None:
        parser.error("the following arguments are required: source")

    return resolved_config


def _namespace_to_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides = vars(args).copy()
    overrides.pop("command", None)
    overrides.pop("config", None)
    return {
        key: value
        for key, value in overrides.items()
        if not _is_suppressed_value(value)
    }


def _has_explicit_command(argv_list: Sequence[str]) -> bool:
    skip_next = False
    for token in argv_list:
        if skip_next:
            skip_next = False
            continue
        if token == "--config":
            skip_next = True
            continue
        if token.startswith("--config="):
            continue
        if token in {"process", "preview", "clear", "init"}:
            return True
        if not token.startswith("-"):
            return False
    return False


def _is_suppressed_value(value: object) -> bool:
    return value == argparse.SUPPRESS or str(value) == argparse.SUPPRESS


def main(argv: Sequence[str] | None = None) -> None:
    resolved_config = parse_args(argv)
    configure_logging(resolved_config.active_log_level)

    if resolved_config.command == "preview":
        run_preview()
        return

    if resolved_config.command == "init":
        run_init(resolved_config.config_path)
        return

    if resolved_config.command == "clear":
        run_reset(resolved_config.source, resolved_config.workers)
        return

    run_process()
