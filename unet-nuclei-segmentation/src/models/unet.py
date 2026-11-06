"""Arsitektur U-Net untuk segmentasi biner."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.blocks import DoubleConv, Down, Up


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        bilinear: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth minimal 1")

        self.depth = depth
        self.inc = DoubleConv(in_channels, base_channels, dropout=dropout)

        encoder_channels = [base_channels * (2**level) for level in range(depth + 1)]
        self.downs = nn.ModuleList(
            [
                Down(encoder_channels[level], encoder_channels[level + 1])
                for level in range(depth)
            ]
        )
        self.ups = nn.ModuleList(
            [
                Up(
                    encoder_channels[level + 1],
                    encoder_channels[level],
                    bilinear=bilinear,
                )
                for level in range(depth - 1, -1, -1)
            ]
        )
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = [self.inc(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        x = skips[-1]
        for up, skip in zip(self.ups, reversed(skips[:-1])):
            x = up(x, skip)
        return self.outc(x)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_unet(config: dict[str, Any]) -> UNet:
    model_cfg = config["model"]
    return UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        depth=int(model_cfg.get("depth", 4)),
        bilinear=bool(model_cfg.get("bilinear", False)),
        dropout=float(model_cfg.get("dropout", 0.0)),
    )
