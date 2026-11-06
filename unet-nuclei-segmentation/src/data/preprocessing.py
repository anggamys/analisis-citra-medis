"""Pemuatan citra/mask dan penggabungan mask instance."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import cv2
import numpy as np


def _to_uint8(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    low, high = float(arr.min()), float(arr.max())
    if high <= low:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)


def _uint16_to_uint8(image: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image, (1, 99))
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.uint8)
    scaled = (image.astype(np.float32) - low) / (high - low) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def resized_size(height: int, width: int, longest_side: int) -> tuple[int, int]:
    """Ukuran citra setelah sisi terpanjang di-resize (aspek rasio tetap).

    Meniru perilaku ``LongestMaxSize`` di Albumentations sehingga hasil
    prediksi bisa dipotong dari kanvas persegi sebelum dikembalikan ke
    ukuran asli.
    """
    scale = float(longest_side) / max(height, width)
    return max(1, round(height * scale)), max(1, round(width * scale))


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Baca citra apa pun menjadi RGB uint8 (alpha dibuang)."""
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Gagal membaca citra: {image_path}")

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            image = np.repeat(image, 3, axis=2)
        elif channels == 2:
            image = np.repeat(image[:, :, :1], 3, axis=2)
        elif channels >= 4:
            image = image[:, :, :3]
    else:
        raise ValueError(f"Shape citra tidak didukung: {image.shape}")

    if image.dtype == np.uint16:
        image = _uint16_to_uint8(image)
    elif image.dtype != np.uint8:
        image = _to_uint8(image)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_mask(path: str | Path) -> np.ndarray:
    """Baca mask menjadi biner uint8 0/1."""
    mask_path = Path(path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Gagal membaca mask: {mask_path}")
    return (mask > 0).astype(np.uint8)


def merge_instance_masks(
    mask_paths: Iterable[str | Path],
    shape: Sequence[int] | None = None,
) -> np.ndarray:
    """Gabungkan seluruh mask instance menjadi satu mask biner 0/255."""
    masks: list[np.ndarray] = []
    expected = tuple(shape) if shape is not None else None

    for mask_path in mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Gagal membaca mask: {mask_path}")
        if expected is None:
            expected = mask.shape
        if mask.shape != expected:
            raise ValueError(f"Ukuran mask {mask_path} {mask.shape} != {expected}")
        masks.append(mask > 0)

    if not masks:
        raise ValueError("Daftar mask instance kosong")

    merged = np.logical_or.reduce(masks).astype(np.uint8) * 255
    return merged
