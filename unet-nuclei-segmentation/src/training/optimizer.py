"""Optimizer dan scheduler."""

from __future__ import annotations

from typing import Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> Optimizer:
    training_cfg = config["training"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )


def build_scheduler(
    optimizer: Optimizer,
    config: dict[str, Any],
) -> tuple[Any | None, str]:
    training_cfg = config["training"]
    name = str(training_cfg.get("scheduler", "plateau")).lower()

    if name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(training_cfg.get("scheduler_factor", 0.5)),
            patience=int(training_cfg.get("scheduler_patience", 3)),
        )
        return scheduler, name
    if name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=int(training_cfg.get("epochs", 50)),
        )
        return scheduler, name
    if name in {"none", ""}:
        return None, "none"
    raise ValueError(f"Scheduler tidak dikenal: {name}")
