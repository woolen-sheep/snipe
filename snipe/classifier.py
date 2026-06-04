from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from .onnxruntime_utils import create_inference_session


class Classifier:
    def __init__(self, model_path: Path, labels_path: Path) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.labels = self._load_labels(labels_path)
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.session = create_inference_session(str(model_path))
        input_details = self.session.get_inputs()[0]
        output_details = self.session.get_outputs()[0]
        self.input_name = input_details.name
        self.output_name = output_details.name

        if len(input_details.shape) != 4:
            raise RuntimeError(f"Unsupported classifier input shape {input_details.shape} for model {model_path}.")

        _, self.input_height, self.input_width, channels = input_details.shape
        if int(channels) != 3:
            raise RuntimeError(f"Expected classifier input with 3 channels, got {input_details.shape} for {model_path}.")

        self.input_height = int(self.input_height)
        self.input_width = int(self.input_width)
        self._lock = threading.Lock()

    @staticmethod
    def _load_labels(labels_path: Path) -> list[str]:
        with labels_path.open("r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]

    def classify(self, image: Image.Image, allowed_indices: Optional[Sequence[int] | np.ndarray] = None) -> tuple[str, float]:
        resized = image.resize((self.input_width, self.input_height), Image.NEAREST)
        input_data = np.asarray(resized, dtype=np.float32)
        input_data = np.expand_dims(input_data, axis=0)
        resized.close()

        with self._lock:
            output = np.array(self.session.run([self.output_name], {self.input_name: input_data})[0][0], copy=True)

        if allowed_indices is not None and len(allowed_indices) > 0:
            allowed_indices_array = np.asarray(allowed_indices, dtype=np.intp)
            if allowed_indices_array.size > 0:
                best_allowed = int(np.argmax(output[allowed_indices_array]))
                index = int(allowed_indices_array[best_allowed])
            else:
                index = int(np.argmax(output))
        else:
            index = int(np.argmax(output))

        score = float(output[index])
        label = self.labels[index] if index < len(self.labels) else str(index)
        return label, score
