from __future__ import annotations

import torch
from src.models.unet import UNet


def test_output_shape() -> None:
    model = UNet(in_channels=3, out_channels=1, base_channels=8, depth=3)
    inputs = torch.randn(2, 3, 64, 64)
    outputs = model(inputs)
    assert outputs.shape == (2, 1, 64, 64)


def test_forward_backward() -> None:
    model = UNet(in_channels=3, out_channels=1, base_channels=8, depth=2)
    inputs = torch.randn(1, 3, 32, 32)
    target = torch.zeros((1, 1, 32, 32))
    target[:, :, 8:24, 8:24] = 1

    logits = model(inputs)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    loss.backward()

    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(gradient is not None for gradient in gradients)


def test_non_square_input() -> None:
    model = UNet(in_channels=3, out_channels=1, base_channels=8, depth=3)
    outputs = model(torch.randn(1, 3, 60, 92))
    assert outputs.shape == (1, 1, 60, 92)
