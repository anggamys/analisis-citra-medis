"""Metrik segmentasi biner: Dice, IoU, precision, recall."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch


def _check_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(f"Shape pred {pred.shape} != target {target.shape}")


def _flatten(
    pred: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    _check_shapes(pred, target)
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
        target = target[:, 0]
    if pred.ndim == 3:
        pred = pred.reshape(pred.shape[0], -1)
        target = target.reshape(target.shape[0], -1)
    return pred.float(), target.float()


def compute_metrics(
    prob: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, torch.Tensor]:
    """Hitung metrik per image dari probabilitas [0,1]."""
    pred, target = _flatten(prob, target)
    pred = (pred >= threshold).float()
    target = (target >= 0.5).float()

    intersection = (pred * target).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    target_sum = target.sum(dim=1)
    union = pred_sum + target_sum - intersection

    both_empty = (pred_sum + target_sum) == 0
    dice = torch.where(
        both_empty,
        torch.ones_like(intersection),
        (2.0 * intersection + eps) / (pred_sum + target_sum + eps),
    )
    iou = torch.where(
        union == 0,
        torch.ones_like(intersection),
        (intersection + eps) / (union + eps),
    )
    precision = torch.where(
        pred_sum > 0,
        (intersection + eps) / (pred_sum + eps),
        torch.where(
            target_sum > 0,
            torch.zeros_like(intersection),
            torch.ones_like(intersection),
        ),
    )
    recall = torch.where(
        target_sum > 0,
        (intersection + eps) / (target_sum + eps),
        torch.ones_like(intersection),
    )
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def summarize_metrics(
    per_image: Iterable[dict],
    keys: tuple[str, ...] = ("dice", "iou", "precision", "recall"),
) -> dict[str, dict[str, float]]:
    rows = list(per_image)
    if not rows:
        raise ValueError("Tidak ada metrik untuk diringkas")

    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "n": int(len(values)),
        }
    return summary


def evaluate_predictions(
    predictions: list[dict],
    threshold: float = 0.5,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    """Hitung metrik per image dari daftar {prob, target, image_id}."""
    rows: list[dict] = []
    for item in predictions:
        prob = torch.from_numpy(np.asarray(item["prob"], dtype=np.float32)).unsqueeze(0)
        target = torch.from_numpy(
            np.asarray(item["target"], dtype=np.float32)
        ).unsqueeze(0)
        metrics = compute_metrics(prob, target, threshold=threshold)

        target_np = np.asarray(item["target"]) > 0.5
        pred_np = np.asarray(item["prob"]) >= threshold
        row = {
            "image_id": str(item["image_id"]),
            "threshold": float(threshold),
            "target_foreground": float(target_np.mean()),
            "pred_foreground": float(pred_np.mean()),
        }
        row.update({key: float(value[0]) for key, value in metrics.items()})
        rows.append(row)

    return rows, summarize_metrics(rows)


def tune_threshold(
    predictions: list[dict],
    candidates: Iterable[float],
) -> tuple[float, list[dict[str, float]]]:
    """Pilih threshold terbaik berdasarkan mean Dice.

    Hanya boleh dipanggil pada validation set.
    """
    trials: list[dict[str, float]] = []
    best_threshold = 0.5
    best_dice = -1.0

    for threshold in sorted({float(value) for value in candidates}):
        _, summary = evaluate_predictions(predictions, threshold=threshold)
        mean_dice = summary["dice"]["mean"]
        trials.append(
            {
                "threshold": float(threshold),
                "dice": float(mean_dice),
                "iou": float(summary["iou"]["mean"]),
                "precision": float(summary["precision"]["mean"]),
                "recall": float(summary["recall"]["mean"]),
            }
        )
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_threshold = float(threshold)

    return best_threshold, trials
