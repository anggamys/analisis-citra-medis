"""Logger CSV sederhana untuk training."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path


class CSVLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=self.fieldnames,
                extrasaction="ignore",
            )
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)
