from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ModuleNotFoundError:
    from tensorflow.lite.python.interpreter import Interpreter

from snipe.config import config
from snipe.detector import Detector
from snipe.image_io import load_image
from snipe.inference import get_gps_species_filter, to_square
from snipe.onnxruntime_utils import create_inference_session


def load_labels(labels_path: Path) -> list[str]:
    with labels_path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def top_k(labels: list[str], values: np.ndarray, limit: int = 5) -> list[dict[str, float | str]]:
    indices = np.argsort(values)[-limit:][::-1]
    return [{"label": labels[int(index)], "score": float(values[int(index)])} for index in indices]


def top_k_allowed(
    labels: list[str], values: np.ndarray, allowed: np.ndarray, limit: int = 5
) -> list[dict[str, float | str]]:
    subset = allowed[np.argsort(values[allowed])[-limit:][::-1]]
    return [{"label": labels[int(index)], "score": float(values[int(index)])} for index in subset]


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]

    image_paths: list[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_file() and child.suffix.lower() in {".jpg", ".jpeg"}:
            image_paths.append(child)
    return image_paths


def analyze_image(
    image_path: Path,
    detector: Detector,
    onnx_session,
    tflite: Interpreter,
    labels: list[str],
    allowed_array: np.ndarray,
    input_name: str,
    input_size: tuple[int, int],
    conf: float,
) -> dict[str, object]:
    loaded = load_image(image_path)
    rgb = loaded.convert("RGB")
    loaded.close()
    image_array = np.array(rgb)
    rgb.close()

    detections, (width, height) = detector.detect_birds(image_array, conf=conf)
    if not detections:
        raise ValueError("No detections found for image")

    square = to_square(detections[0], width, height)
    image = Image.fromarray(image_array)
    crop = image.crop(square)
    image.close()

    input_width, input_height = input_size
    resized = crop.resize((input_width, input_height), Image.NEAREST)
    input_data = np.expand_dims(np.asarray(resized, dtype=np.float32), axis=0)
    resized.close()
    crop.close()

    onnx_values = np.array(onnx_session.run(None, {input_name: input_data})[0][0], copy=True)

    input_index = int(tflite.get_input_details()[0]["index"])
    output_index = int(tflite.get_output_details()[0]["index"])
    tflite.set_tensor(input_index, input_data)
    tflite.invoke()
    tflite_values = np.array(tflite.get_tensor(output_index)[0], copy=True)

    onnx_allowed_index = int(allowed_array[np.argmax(onnx_values[allowed_array])])
    tflite_allowed_index = int(allowed_array[np.argmax(tflite_values[allowed_array])])
    diff = np.abs(onnx_values - tflite_values)

    return {
        "image": str(image_path),
        "bbox": [int(value) for value in square],
        "onnx_top5_all": top_k(labels, onnx_values),
        "tflite_top5_all": top_k(labels, tflite_values),
        "onnx_top5_region": top_k_allowed(labels, onnx_values, allowed_array),
        "tflite_top5_region": top_k_allowed(labels, tflite_values, allowed_array),
        "onnx_top1_region": {
            "label": labels[onnx_allowed_index],
            "score": float(onnx_values[onnx_allowed_index]),
        },
        "tflite_top1_region": {
            "label": labels[tflite_allowed_index],
            "score": float(tflite_values[tflite_allowed_index]),
        },
        "top1_region_match": labels[onnx_allowed_index] == labels[tflite_allowed_index],
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare TFLite and ONNX classifier outputs for one image or a JPG directory")
    parser.add_argument("image", type=Path, help="Path to the source image or directory of JPG images")
    parser.add_argument("--region-filter", default="CN", help="Region code used to constrain labels")
    parser.add_argument("--detector-model", type=Path, default=Path("snipe/resources/yolo26x-seg.onnx"))
    parser.add_argument("--onnx-model", type=Path, default=Path("snipe/resources/classifier.onnx"))
    parser.add_argument("--tflite-model", type=Path, default=Path("snipe/resources/classifier.tflite"))
    parser.add_argument("--labels", type=Path, default=Path("snipe/resources/labels.txt"))
    parser.add_argument(
        "--gps-filter-data",
        type=Path,
        default=Path("snipe/resources/location_species_filter.json"),
    )
    parser.add_argument("--report-out", type=Path, help="Optional path to write a JSON report")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    image_paths = iter_images(args.image)
    if not image_paths:
        raise SystemExit(f"No JPG images found at {args.image}")

    detector = Detector(
        args.detector_model,
        overlap_threshold=config.process.detector_merge_overlap_threshold,
        min_relative_area=config.process.detector_min_box_area_ratio,
    )

    onnx_session = create_inference_session(str(args.onnx_model))
    input_name = onnx_session.get_inputs()[0].name
    input_shape = onnx_session.get_inputs()[0].shape
    _, input_height, input_width, _ = input_shape

    species_filter = get_gps_species_filter(args.gps_filter_data, args.labels)
    allowed, _ = species_filter.get_allowed_indices_for_region(args.region_filter)
    if allowed is None or len(allowed) == 0:
        raise SystemExit(f"No allowed labels found for region {args.region_filter}")
    allowed_array = np.asarray(allowed, dtype=np.intp)

    tflite = Interpreter(model_path=str(args.tflite_model))
    tflite.allocate_tensors()

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for image_path in image_paths:
        try:
            result = analyze_image(
                image_path=image_path,
                detector=detector,
                onnx_session=onnx_session,
                tflite=tflite,
                labels=labels,
                allowed_array=allowed_array,
                input_name=input_name,
                input_size=(int(input_width), int(input_height)),
                conf=config.process.conf,
            )
        except Exception as exc:
            failures.append({"image": str(image_path), "error": str(exc)})
            continue
        results.append(result)

    report = {
        "region_filter": args.region_filter,
        "image_root": str(args.image),
        "total_images": len(image_paths),
        "processed_images": len(results),
        "failed_images": len(failures),
        "top1_region_match_count": sum(1 for result in results if result["top1_region_match"]),
        "results": results,
        "failures": failures,
    }

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("images", len(image_paths))
    print("processed", len(results))
    print("failed", len(failures))
    print("top1 region matches", report["top1_region_match_count"])
    if args.report_out is not None:
        print("report", args.report_out)

    if len(results) == 1:
        result = results[0]
        print("bbox", tuple(result["bbox"]))
        print("onnx top5 all", result["onnx_top5_all"])
        print("tflite top5 all", result["tflite_top5_all"])
        print("onnx top5 region", result["onnx_top5_region"])
        print("tflite top5 region", result["tflite_top5_region"])
        print("onnx top1 region", result["onnx_top1_region"])
        print("tflite top1 region", result["tflite_top1_region"])
        print("max abs diff", result["max_abs_diff"])
        print("mean abs diff", result["mean_abs_diff"])


if __name__ == "__main__":
    main()
