"""Visualisasi prediksi dan kurva training."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def tensor_to_image(
    tensor: np.ndarray | torch.Tensor,
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        tensor_value = cast(torch.Tensor, tensor)
        array = tensor_value.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    if array.ndim == 3:
        array = np.transpose(array, (1, 2, 0))
    array = array * np.asarray(std).reshape(1, 1, -1)
    array = array + np.asarray(mean).reshape(1, 1, -1)
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> np.ndarray:
    output = image.copy()
    active = np.asarray(mask) > 0
    if active.any():
        color_array = np.asarray(color, dtype=np.float32)
        blended = (
            output[active].astype(np.float32) * (1.0 - alpha) + color_array * alpha
        )
        output[active] = np.clip(blended, 0, 255).astype(np.uint8)
    return output


def _dice(prob: np.ndarray, target: np.ndarray, threshold: float) -> float:
    pred = np.asarray(prob) >= threshold
    truth = np.asarray(target) >= 0.5
    intersection = np.logical_and(pred, truth).sum()
    denominator = pred.sum() + truth.sum()
    if denominator == 0:
        return 1.0
    return float(2.0 * intersection / denominator)


def save_prediction_grid(
    predictions: list[dict[str, Any]],
    output_path: str | Path,
    threshold: float = 0.5,
    max_samples: int = 8,
    mean: Sequence[float] = (0.5, 0.5, 0.5),
    std: Sequence[float] = (0.5, 0.5, 0.5),
    sort_by_dice: bool = True,
) -> Path:
    """Simpan grid: citra, GT, prediksi, overlay."""
    if not predictions:
        raise ValueError("Tidak ada prediksi untuk divisualisasikan")

    samples = predictions
    if sort_by_dice:
        samples = sorted(
            samples,
            key=lambda item: _dice(item["prob"], item["target"], threshold),
        )
    samples = samples[:max_samples]

    rows = len(samples)
    figure, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    axes = np.atleast_2d(axes)

    for row_index, item in enumerate(samples):
        image = tensor_to_image(item["image"], mean=mean, std=std)
        target = (np.asarray(item["target"]) >= 0.5).astype(np.uint8)
        prediction = (np.asarray(item["prob"]) >= threshold).astype(np.uint8)
        combined = overlay_mask(image, target, color=(0, 255, 0))
        combined = overlay_mask(combined, prediction, color=(255, 0, 0))

        dice = _dice(item["prob"], item["target"], threshold)
        panels = [
            (image, f"{item['image_id']} (Dice={dice:.3f})"),
            (target, "Ground truth"),
            (prediction, f"Prediksi (t={threshold})"),
            (combined, "GT hijau / prediksi merah"),
        ]
        for column, (panel, title) in enumerate(panels):
            axis = axes[row_index, column]
            if panel.ndim == 2:
                axis.imshow(panel, cmap="gray", vmin=0, vmax=1)
            else:
                axis.imshow(panel)
            axis.set_title(title, fontsize=9)
            axis.axis("off")

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output


def save_binary_mask(mask: np.ndarray, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), (np.asarray(mask) > 0).astype(np.uint8) * 255)
    return output


def save_overlay_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.45,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    overlay = overlay_mask(image_rgb, mask, color=color, alpha=alpha)
    cv2.imwrite(str(output), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return output


def plot_training_curves(
    history: list[dict[str, Any]] | str | Path,
    output_path: str | Path,
) -> Path:
    if isinstance(history, (str, Path)):
        with Path(history).open("r", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        rows = history
    if not rows:
        raise ValueError("History training kosong")

    epochs = [int(row["epoch"]) + 1 for row in rows]

    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, values("train_loss"), label="train")
    axes[0].plot(epochs, values("val_loss"), label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, values("val_dice"), label="Dice")
    axes[1].plot(epochs, values("val_iou"), label="IoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output
