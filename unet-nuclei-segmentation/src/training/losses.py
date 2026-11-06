"""Fungsi loss untuk segmentasi biner."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-7, from_logits: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.from_logits = from_logits

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(prediction.shape[0], -1).float()
        target = target.reshape(target.shape[0], -1).float()

        intersection = (prediction * target).sum(dim=1)
        denominator = prediction.sum(dim=1) + target.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        pos_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Bobot loss tidak boleh negatif")
        if bce_weight == 0 and dice_weight == 0:
            raise ValueError("Minimal satu bobot loss harus > 0")

        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([float(pos_weight)], dtype=torch.float32)
        )
        self.dice = DiceLoss(from_logits=True)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        loss = prediction.new_tensor(0.0)
        if self.bce_weight > 0:
            loss = loss + self.bce_weight * self.bce(prediction, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(prediction, target)
        return loss


def build_loss(config: dict[str, Any]) -> BCEDiceLoss:
    loss_cfg = config.get("loss", {})
    return BCEDiceLoss(
        bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
        dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
        pos_weight=float(loss_cfg.get("bce_pos_weight", 1.0)),
    )
