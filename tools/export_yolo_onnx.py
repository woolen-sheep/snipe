from __future__ import annotations

import argparse
from pathlib import Path

try:
    import onnx
except ImportError:  # pragma: no cover - depends on local tool environment
    onnx = None


def find_pt_models(data_dir: Path) -> list[Path]:
    return sorted(path for path in data_dir.glob("*.pt") if path.is_file())


def export_model(model_path: Path, output_dir: Path, imgsz: int, opset: int, simplify: bool) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - tool-only dependency
        raise RuntimeError(
            "The export tool requires ultralytics. Run it with `uv run --with ultralytics --with onnx python tools/export_yolo_onnx.py`."
        ) from exc

    model = YOLO(str(model_path))
    exported_path = Path(
        model.export(
            format="onnx",
            imgsz=imgsz,
            dynamic=True,
            opset=opset,
            simplify=simplify,
        )
    ).resolve()

    target_path = (output_dir / f"{model_path.stem}.onnx").resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != target_path:
        target_path.write_bytes(exported_path.read_bytes())
    return target_path


def _read_input_dims(onnx_path: Path) -> list[str]:
    if onnx is None:
        raise RuntimeError(
            "The 'onnx' package is required to validate exported models. "
            "Run this script with `uv run --with onnx python tools/export_yolo_onnx.py`."
        )

    model = onnx.load(str(onnx_path))
    if not model.graph.input:
        raise RuntimeError(f"Exported ONNX has no graph inputs: {onnx_path}")

    dims: list[str] = []
    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.dim_value:
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return dims


def validate_dynamic_hw(onnx_path: Path) -> None:
    dims = _read_input_dims(onnx_path)
    if len(dims) != 4:
        raise RuntimeError(f"Expected 4D input for {onnx_path}, got dims={dims}")

    height_dim = dims[2]
    width_dim = dims[3]
    if height_dim.isdigit() or width_dim.isdigit():
        raise RuntimeError(
            f"Expected dynamic height/width for {onnx_path}, got dims={dims}. "
            "Re-export with dynamic=True and verify the model supports dynamic spatial axes."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO .pt models under data/ to ONNX with dynamic height and width axes"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to scan for YOLO .pt models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for exported ONNX files; defaults to the data directory",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Reference export size used by Ultralytics while keeping H/W dynamic in the ONNX graph",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version to use for export",
    )
    parser.add_argument(
        "--simplify",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run Ultralytics ONNX graph simplification after export",
    )
    parser.add_argument(
        "--model",
        type=Path,
        action="append",
        help="Specific .pt model(s) to export; may be passed multiple times",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.data_dir
    models = [path.resolve() for path in args.model] if args.model else find_pt_models(args.data_dir)
    if not models:
        raise SystemExit(f"No .pt models found under {args.data_dir}")

    for model_path in models:
        if model_path.suffix.lower() != ".pt":
            raise SystemExit(f"Expected a .pt model, got: {model_path}")

        exported_path = export_model(
            model_path=model_path,
            output_dir=output_dir,
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=args.simplify,
        )
        validate_dynamic_hw(exported_path)
        dims = _read_input_dims(exported_path)
        print(f"Exported {model_path} -> {exported_path} with input dims {dims}")


if __name__ == "__main__":
    main()
