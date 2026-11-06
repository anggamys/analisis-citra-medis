"""Evaluasi model dan penyimpanan hasil."""

from __future__ import annotations

import csv
from collections.abc import Iterable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import evaluate_predictions, tune_threshold
from src.utils.io import save_json


def _autocast(enabled: bool):
    if enabled:
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    max_samples: int | None = None,
    include_images: bool = True,
) -> list[dict[str, Any]]:
    model.eval()
    results: list[dict[str, Any]] = []
    amp_enabled = use_amp and device.type == "cuda"

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True).float()
        with _autocast(amp_enabled):
            logits = model(images)
        probabilities = torch.sigmoid(logits).float().cpu()
        targets = masks.float().cpu()

        for index, image_id in enumerate(batch["image_id"]):
            item: dict[str, Any] = {
                "image_id": str(image_id),
                "prob": probabilities[index, 0].numpy(),
                "target": targets[index, 0].numpy(),
                "original_size": batch["original_size"][index].tolist(),
            }
            if include_images:
                item["image"] = images[index].detach().cpu().numpy()
            results.append(item)
            if max_samples is not None and len(results) >= max_samples:
                return results

    return results


def find_best_threshold(
    model: torch.nn.Module,
    val_loader: DataLoader,
    candidates: Iterable[float],
    device: torch.device,
    use_amp: bool = False,
) -> tuple[float, list[dict[str, float]]]:
    predictions = collect_predictions(
        model,
        val_loader,
        device,
        use_amp=use_amp,
        include_images=False,
    )
    return tune_threshold(predictions, candidates)


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    use_amp: bool = False,
    max_samples: int | None = None,
) -> tuple[list[dict], dict[str, dict[str, float]], list[dict[str, Any]]]:
    predictions = collect_predictions(
        model,
        loader,
        device,
        use_amp=use_amp,
        max_samples=max_samples,
        include_images=True,
    )
    rows, summary = evaluate_predictions(predictions, threshold=threshold)
    return rows, summary, predictions


def evaluate_predictions_direct(
    predictions: list[dict],
    threshold: float,
) -> tuple[list[dict], dict[str, dict[str, float]]]:
    rows, summary = evaluate_predictions(predictions, threshold=threshold)
    return rows, summary


def save_evaluation(
    output_dir: str | Path,
    split: str,
    rows: list[dict],
    summary: dict[str, dict[str, float]],
    threshold: float,
    trials: list[dict[str, float]] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    per_image_path = output_path / f"per_image_{split}.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload: dict[str, Any] = {
        "split": split,
        "threshold": float(threshold),
        "num_images": len(rows),
        "metrics": summary,
    }
    if trials is not None:
        payload["threshold_trials"] = trials
    save_json(payload, output_path / f"metrics_{split}.json")


def subset_summary(predictions: list[dict], threshold: float) -> dict:
    rows, summary = evaluate_predictions(predictions, threshold=threshold)
    return {"rows": rows, "summary": summary}


def predictions_to_arrays(predictions: list[dict]) -> dict[str, np.ndarray]:
    return {
        "prob": np.stack([item["prob"] for item in predictions]),
        "target": np.stack([item["target"] for item in predictions]),
    }
