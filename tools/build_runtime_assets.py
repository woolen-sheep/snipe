from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


LOCAL_MODEL_DIR = Path("local_models")


def build_common_name_csv(taxonomy_cn_path: Path, taxonomy_us_path: Path, output_path: Path) -> None:
    merged: dict[str, dict[str, str]] = {}
    order: list[str] = []

    for source_path, output_column in (
        (taxonomy_cn_path, "COMMON_NAME_ZH_CN"),
        (taxonomy_us_path, "COMMON_NAME_US"),
    ):
        with source_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                species_code = (row.get("SPECIES_CODE") or row.get("species_code") or "").strip()
                common_name = (row.get("COMMON_NAME") or row.get("common_name") or "").strip()
                if not species_code:
                    continue

                if species_code not in merged:
                    merged[species_code] = {
                        "SPECIES_CODE": species_code,
                        "COMMON_NAME_ZH_CN": "",
                        "COMMON_NAME_US": "",
                    }
                    order.append(species_code)

                merged[species_code][output_column] = common_name

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["SPECIES_CODE", "COMMON_NAME_ZH_CN", "COMMON_NAME_US"])
        writer.writeheader()
        for species_code in order:
            writer.writerow(merged[species_code])


def copy_labels_file(labels_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if labels_path.resolve() == output_path.resolve():
        return
    shutil.copy2(labels_path, output_path)


def copy_classifier_model(model_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.resolve() == output_path.resolve():
        return
    shutil.copy2(model_path, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime support assets for Snipe")
    parser.add_argument(
        "--taxonomy-cn",
        type=Path,
        default=Path("data/taxonomy.csv"),
        help="Path to the CN taxonomy source CSV",
    )
    parser.add_argument(
        "--taxonomy-us",
        type=Path,
        default=Path("data/taxonomy_us.csv"),
        help="Path to the US taxonomy source CSV",
    )
    parser.add_argument(
        "--common-name-output",
        type=Path,
        default=Path("snipe/resources/common_name.csv"),
        help="Output path for the merged bilingual common-name CSV",
    )
    parser.add_argument(
        "--labels-input",
        type=Path,
        default=Path("data/labels.txt"),
        help="Path to the plain-text classifier labels file",
    )
    parser.add_argument(
        "--labels-output",
        type=Path,
        default=Path("snipe/resources/labels.txt"),
        help="Output path for the classifier labels file",
    )
    parser.add_argument(
        "--classifier-input",
        type=Path,
        default=LOCAL_MODEL_DIR / "classifier.onnx",
        help="Path to the source ONNX classifier model",
    )
    parser.add_argument(
        "--classifier-output",
        type=Path,
        default=LOCAL_MODEL_DIR / "classifier.onnx",
        help="Output path for the local ONNX classifier model artifact",
    )
    args = parser.parse_args()

    build_common_name_csv(args.taxonomy_cn, args.taxonomy_us, args.common_name_output)
    copy_labels_file(args.labels_input, args.labels_output)
    copy_classifier_model(args.classifier_input, args.classifier_output)

    print(f"Wrote common names: {args.common_name_output}")
    print(f"Wrote labels: {args.labels_output}")
    print(f"Wrote classifier: {args.classifier_output}")


if __name__ == "__main__":
    main()
