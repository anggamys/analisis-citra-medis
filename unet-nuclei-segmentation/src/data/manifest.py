"""Baca/tulis manifest CSV."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

REQUIRED_COLUMNS = {"image_id", "image_path"}


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest tidak ditemukan: {manifest_path}")
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest kosong: {manifest_path}")
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {sorted(missing)}")
    return rows


def write_manifest(
    path: str | Path,
    rows: Iterable[dict],
    fieldnames: list[str] | None = None,
) -> Path:
    row_list = list(rows)
    if not row_list:
        raise ValueError("Tidak ada baris untuk ditulis")
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or list(row_list[0].keys())
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(row_list)
    return manifest_path


def validate_manifest(rows: list[dict[str, str]], require_mask: bool = True) -> None:
    if not rows:
        raise ValueError("Manifest kosong")
    missing = REQUIRED_COLUMNS - set(rows[0].keys())
    if missing:
        raise ValueError(f"Kolom wajib tidak ada: {sorted(missing)}")
    if require_mask and "mask_path" not in rows[0]:
        raise ValueError("Manifest evaluasi harus memiliki kolom mask_path")
    image_ids = [row["image_id"] for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("Manifest mengandung image_id duplikat")
