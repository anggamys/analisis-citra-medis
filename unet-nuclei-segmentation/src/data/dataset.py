"""Dataset PyTorch untuk citra nukleus dan mask biner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.manifest import read_manifest, validate_manifest
from src.data.preprocessing import load_image_rgb, load_mask
from src.utils.config import resolve_path


def _resolve(path_value: str | Path) -> Path:
    return resolve_path(path_value)


class NucleiDataset(Dataset):
    """Dataset supervised (image + mask) atau inference (image saja)."""

    def __init__(
        self,
        manifest: str | Path | list[dict[str, str]],
        transform: Any | None = None,
        has_masks: bool = True,
    ) -> None:
        self.rows = (
            read_manifest(manifest) if not isinstance(manifest, list) else manifest
        )
        validate_manifest(self.rows, require_mask=has_masks)
        self.transform = transform
        self.has_masks = has_masks

        for index, row in enumerate(self.rows):
            image_path = _resolve(row["image_path"])
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Baris {index}: citra tidak ditemukan: {image_path}"
                )
            if has_masks:
                mask_path = _resolve(row["mask_path"])
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"Baris {index}: mask tidak ditemukan: {mask_path}"
                    )

    def __len__(self) -> int:
        return len(self.rows)

    def _prepare_tensor(self, array: np.ndarray, is_mask: bool = False) -> torch.Tensor:
        if is_mask:
            tensor = torch.from_numpy(array)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            return (tensor.float() > 0).float()
        tensor = torch.from_numpy(array.transpose(2, 0, 1))
        return tensor.float() / 255.0

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = load_image_rgb(_resolve(row["image_path"]))
        height, width = image.shape[:2]

        if self.has_masks:
            mask = load_mask(_resolve(row["mask_path"]))
            if mask.shape != (height, width):
                raise ValueError(
                    f"Ukuran mask {row['image_id']} {mask.shape} "
                    f"!= citra {(height, width)}"
                )
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image_tensor = transformed["image"]
                mask_tensor = transformed["mask"]
            else:
                image_tensor = self._prepare_tensor(image)
                mask_tensor = self._prepare_tensor(mask, is_mask=True)

            if not torch.is_tensor(image_tensor):
                image_tensor = self._prepare_tensor(np.asarray(image_tensor))
            if not torch.is_tensor(mask_tensor):
                mask_tensor = self._prepare_tensor(
                    np.asarray(mask_tensor), is_mask=True
                )
            if mask_tensor.ndim == 2:
                mask_tensor = mask_tensor.unsqueeze(0)
            mask_tensor = (mask_tensor.float() > 0).float()

            return {
                "image": image_tensor.float(),
                "mask": mask_tensor,
                "image_id": str(row["image_id"]),
                "image_path": str(_resolve(row["image_path"])),
                "original_size": torch.tensor([height, width], dtype=torch.long),
            }

        if self.transform is not None:
            image_tensor = self.transform(image=image)["image"]
        else:
            image_tensor = self._prepare_tensor(image)
        if not torch.is_tensor(image_tensor):
            image_tensor = self._prepare_tensor(np.asarray(image_tensor))

        return {
            "image": image_tensor.float(),
            "image_id": str(row["image_id"]),
            "image_path": str(_resolve(row["image_path"])),
            "original_size": torch.tensor([height, width], dtype=torch.long),
        }
