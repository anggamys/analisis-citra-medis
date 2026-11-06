from __future__ import annotations

from src.data.preprocessing import resized_size


def test_resized_size_downscale() -> None:
    assert resized_size(96, 80, 64) == (64, 53)


def test_resized_size_upscale_landscape() -> None:
    assert resized_size(256, 320, 256) == (205, 256)


def test_resized_size_extreme_aspect_ratio() -> None:
    assert resized_size(100, 400, 256) == (64, 256)
