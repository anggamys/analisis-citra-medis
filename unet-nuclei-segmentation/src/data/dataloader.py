"""Pembuat DataLoader."""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from src.utils.seed import worker_init_fn


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
