from __future__ import annotations

import argparse
from pathlib import Path

import tf2onnx.convert as tf2onnx_convert


def convert_tflite_to_onnx(tflite_path: Path, onnx_path: Path, opset: int) -> None:
    """Convert a TFLite model to ONNX using tf2onnx."""
    model_proto, _ = tf2onnx_convert.from_tflite(str(tflite_path), opset=opset)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_path.write_bytes(model_proto.SerializeToString())


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert TFLite to ONNX with tf2onnx")
    parser.add_argument(
        "--tflite",
        default="snipe/resources/classifier.tflite",
        type=Path,
        help="Path to input .tflite file",
    )
    parser.add_argument(
        "--out",
        default="snipe/resources/classifier.onnx",
        type=Path,
        help="Path to output .onnx file",
    )
    parser.add_argument(
        "--opset",
        default=13,
        type=int,
        help="Target ONNX opset version",
    )
    args = parser.parse_args()

    convert_tflite_to_onnx(args.tflite, args.out, args.opset)


if __name__ == "__main__":
    main()
