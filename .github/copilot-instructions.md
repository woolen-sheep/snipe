# Snipe Project Context

## What This Project Does
Snipe is a Python CLI tool for bird photo annotation.
It processes JPEG and common RAW photo formats, detects birds with YOLO, classifies each detected crop with an ONNX model, and writes metadata back into image files.

Primary outcomes:
- Bird species labels written to XMP keyword fields.
- Detected bird regions written as MWG region metadata (normalized coordinates).
- Optional desktop preview UI to inspect image regions/labels.

## Main Workflows

### 1) Process workflow (`snipe process`)
1. Collect supported images from a file or directory.
2. Load image data (RAW decode via rawpy when needed).
3. Detect bird bounding boxes using a local YOLO ONNX model via `onnxruntime` (`Detector`).
4. Convert each box to a square crop and classify with ONNX Runtime (`Classifier`).
5. Resolve species code to common name via `snipe/resources/common_name.csv` using the selected language column when available.
6. Write labels and MWG regions via `exifmwg`.

Notes:
- Supports parallel processing with thread pool workers.
- Classifier inference is guarded by a lock so the shared runtime session stays thread-safe.
- `--dry-run` runs detection/classification without metadata writes.

### 2) Preview workflow (`snipe preview`)
- Launches a pywebview app that loads `snipe/web/preview/index.html`.
- Converts images to cached JPEGs under the user config cache directory such as `~/.config/.snipe/cache/` or `%APPDATA%\\.snipe\\cache\\` for display.
- Reads XMP MWG regions from metadata and overlays boxes in the UI.

## Key Modules
- `snipe/cli.py`: CLI entrypoint and command parsing.
- `snipe/processing.py`: processing pipeline orchestration and worker dispatch.
- `snipe/detector.py`: YOLO-based bird detection (keeps only class name `bird`).
- `snipe/classifier.py`: ONNX classifier inference and label lookup.

## Data/Model Files
Expected under `snipe/resources/` at runtime:
- `yolo26x-seg.onnx` (detector, must exist locally)
- `classifier.onnx` (classifier)
- `labels.txt` (classifier labels)
- `common_name.csv` (species_code -> common_name for `COMMON_NAME_ZH_CN` and `COMMON_NAME_US`)

## Dependencies and Runtime
- Python >= 3.12
- Core libs: onnxruntime, rawpy, pillow, numpy, exifmwg, pywebview, loguru
- Requires `exiftool` available on PATH for metadata writes via exifmwg
- Detector models are local ONNX assets under `snipe/resources/`; runtime should not download detector weights.
- GPS-based species filtering uses an uncompressed JSON artifact in `snipe/resources/location_species_filter.json` and preloads it into memory at runtime; keep the filtering logic isolated from the main pipeline so the backend can still be replaced later.

## Typical Commands
- `uv sync`
- `uv run snipe <path> --conf 0.35 --workers 4`
- `uv run snipe preview <path> --debug`

## Copilot Guidance for This Repo
- Treat this as a metadata-centric photo pipeline, not a generic web app.
- Preserve EXIF/XMP write behavior and MWG region schema when refactoring.
- Be careful with image I/O fallbacks (RAW decode -> PIL -> RAW thumbnail).
- Keep defaults aligned with packaged `snipe/resources/` assets unless user overrides CLI options.
- Avoid introducing changes that break preview cache path assumptions (`~/.config/.snipe/cache` on Unix-like systems, `%APPDATA%\\.snipe\\cache` on Windows).
- Auxiliary reference folders are not part of this project's implementation, architecture, or requirements.
