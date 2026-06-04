from __future__ import annotations

import shutil
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Sequence, TextIO

from tqdm import tqdm
from tqdm.utils import disp_len, disp_trim


class TerminalProgressReporter:
    def __init__(self, total: int, stream: TextIO | None = None) -> None:
        self._total = total
        self._stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._processed = 0
        self._latest_image = "Waiting for first result..."
        self._latest_status = "Starting"
        self._accumulated_bird_count = 0
        self._accumulated_species: Counter[str] = Counter()
        self._rendered = False
        self._render_locked()

    def render(self, *, image: Path | None = None, status: str | None = None) -> None:
        with self._lock:
            if image is not None:
                self._latest_image = image.name
            if status is not None:
                self._latest_status = status
            self._render_locked()

    def advance(self, image: Path, labels: Sequence[str], *, status: str | None = None) -> None:
        with self._lock:
            self._processed += 1
            self._latest_image = image.name
            self._latest_status = status or self._summarize_labels(labels)
            self._accumulated_bird_count += len(labels)
            self._accumulated_species.update(label for label in labels if label)
            self._render_locked()

    def close(self) -> None:
        with self._lock:
            self._render_locked()
            self._stream.write("\n")
            self._stream.flush()

    def _render_locked(self) -> None:
        width = shutil.get_terminal_size(fallback=(100, 20)).columns
        progress_line = self._build_progress_line(width)
        status_line = self._build_status_line(width)

        if self._rendered:
            self._stream.write("\x1b[2F")

        for line in (progress_line, status_line):
            self._stream.write("\x1b[2K")
            self._stream.write(line)
            self._stream.write("\n")

        self._stream.flush()
        self._rendered = True

    def _build_progress_line(self, width: int) -> str:
        elapsed = max(0.0, time.monotonic() - self._started_at)
        line = tqdm.format_meter(
            n=self._processed,
            total=self._total,
            elapsed=elapsed,
            prefix="Processing",
            ncols=width,
        )
        return self._fit_line(line, width)

    def _build_status_line(self, width: int) -> str:
        left = f"{self._latest_image} | {self._latest_status}"
        right = f"birds {self._accumulated_bird_count} | species {len(self._accumulated_species)}"
        right_width = disp_len(right)
        if right_width >= width:
            return self._fit_line(right, width)
        available_left = max(0, width - right_width - 1)
        left = self._fit_line(left, available_left)
        if not left:
            return right.rjust(width)
        padding = max(1, width - disp_len(left) - right_width)
        return f"{left}{' ' * padding}{right}"

    @staticmethod
    def _summarize_labels(labels: Sequence[str]) -> str:
        unique_labels = list(dict.fromkeys(labels))
        if not unique_labels:
            return "No birds"
        visible_labels = unique_labels[:3]
        summary = ", ".join(visible_labels)
        remaining = len(unique_labels) - len(visible_labels)
        if remaining > 0:
            summary = f"{summary} +{remaining} more"
        return summary

    @staticmethod
    def _fit_line(text: str, width: int) -> str:
        if width <= 0:
            return ""
        if disp_len(text) <= width:
            return text
        if width <= 3:
            return disp_trim(text, width)
        return disp_trim(text, width - 3) + "..."
