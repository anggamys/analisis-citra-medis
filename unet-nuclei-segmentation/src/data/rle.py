"""Encoder/decoder RLE format Kaggle (kolom-mayor)."""

from __future__ import annotations

import numpy as np


def rle_decode(rle: str, height: int, width: int) -> np.ndarray:
    """Decode RLE menjadi mask biner uint8 (0/1)."""
    if not rle or not str(rle).strip() or str(rle).strip() == "-1":
        return np.zeros((height, width), dtype=np.uint8)

    parts = str(rle).split()
    if len(parts) % 2 != 0:
        raise ValueError("RLE harus berisi pasangan start-length")

    starts = np.asarray(parts[0::2], dtype=np.int64) - 1
    lengths = np.asarray(parts[1::2], dtype=np.int64)
    total = height * width
    if starts.min() < 0 or lengths.min() < 0:
        raise ValueError("RLE mengandung nilai negatif")
    if (starts + lengths).max() > total:
        raise ValueError("RLE melebihi ukuran mask")

    flat = np.zeros(total, dtype=np.uint8)
    for start, length in zip(starts, lengths):
        flat[start : start + length] = 1
    return flat.reshape((height, width), order="F")


def rle_encode(mask: np.ndarray) -> str:
    """Encode mask biner menjadi RLE format Kaggle."""
    pixels = (np.asarray(mask).reshape(-1, order="F") > 0).astype(np.uint8)
    if not pixels.any():
        return ""
    padded = np.concatenate(([0], pixels, [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0]
    lengths = ends - starts + 1
    return " ".join(f"{start} {length}" for start, length in zip(starts, lengths))
