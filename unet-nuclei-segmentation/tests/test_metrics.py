from __future__ import annotations

import torch
from src.evaluation.metrics import compute_metrics, summarize_metrics


def test_perfect_prediction() -> None:
    target = torch.zeros((2, 1, 8, 8))
    target[:, :, 2:6, 2:6] = 1
    metrics = compute_metrics(target, target, threshold=0.5)
    assert torch.allclose(metrics["dice"], torch.ones(2))
    assert torch.allclose(metrics["iou"], torch.ones(2))
    assert torch.allclose(metrics["precision"], torch.ones(2))
    assert torch.allclose(metrics["recall"], torch.ones(2))


def test_empty_prediction_and_target() -> None:
    target = torch.zeros((1, 1, 4, 4))
    metrics = compute_metrics(target, target, threshold=0.5)
    assert float(metrics["dice"][0]) == 1.0
    assert float(metrics["iou"][0]) == 1.0


def test_empty_prediction_nonempty_target() -> None:
    prob = torch.zeros((1, 1, 4, 4))
    target = torch.zeros((1, 1, 4, 4))
    target[:, :, 1:3, 1:3] = 1
    metrics = compute_metrics(prob, target, threshold=0.5)
    # eps pada metrik membuat hasilnya sangat kecil, bukan tepat nol.
    assert abs(float(metrics["dice"][0])) < 1e-6
    assert abs(float(metrics["recall"][0])) < 1e-6


def test_summary_mean() -> None:
    rows = [
        {"dice": 0.5, "iou": 0.5, "precision": 0.5, "recall": 0.5},
        {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0},
    ]
    summary = summarize_metrics(rows)
    assert summary["dice"]["mean"] == 0.75
    assert summary["dice"]["n"] == 2
