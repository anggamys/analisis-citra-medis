from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from src.data.augmentation import build_eval_transforms
from src.data.dataset import NucleiDataset


def _write_sample(root: Path) -> tuple[Path, Path]:
    image = np.zeros((96, 80, 4), dtype=np.uint8)
    image[..., 0] = 40
    image[..., 1] = 80
    image[..., 2] = 120
    image[..., 3] = 255
    mask = np.zeros((96, 80), dtype=np.uint8)
    mask[20:70, 10:60] = 255

    image_path = root / "image.png"
    mask_path = root / "mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    return image_path, mask_path


def _write_manifest(root: Path, image_path: Path, mask_path: Path | None) -> Path:
    manifest = root / "manifest.csv"
    row = {
        "image_id": "sample",
        "image_path": str(image_path),
    }
    if mask_path is not None:
        row["mask_path"] = str(mask_path)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    return manifest


def test_dataset_shapes_and_mask_range(tmp_path: Path) -> None:
    image_path, mask_path = _write_sample(tmp_path)
    manifest = _write_manifest(tmp_path, image_path, mask_path)
    dataset = NucleiDataset(
        manifest,
        transform=build_eval_transforms(64),
        has_masks=True,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["mask"].shape == (1, 64, 64)
    assert set(torch.unique(sample["mask"]).tolist()).issubset({0.0, 1.0})
    assert sample["original_size"].tolist() == [96, 80]


def test_inference_dataset_without_mask(tmp_path: Path) -> None:
    image_path, _ = _write_sample(tmp_path)
    manifest = _write_manifest(tmp_path, image_path, None)
    dataset = NucleiDataset(
        manifest,
        transform=build_eval_transforms(64),
        has_masks=False,
    )
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert "mask" not in sample
