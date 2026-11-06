"""Pembagian ID citra menjadi train/val tanpa leakage."""

from __future__ import annotations

import random
from collections.abc import Iterable


def split_ids(
    ids: Iterable[str],
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Split ID secara acak dan deterministik."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio harus di rentang [0, 1)")

    unique_ids = sorted({str(image_id) for image_id in ids})
    if len(unique_ids) < 2 or val_ratio == 0.0:
        return unique_ids, []

    shuffled = unique_ids.copy()
    random.Random(seed).shuffle(shuffled)

    n_val = max(1, round(len(shuffled) * val_ratio))
    n_val = min(n_val, len(shuffled) - 1)

    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    if set(train_ids) & set(val_ids):
        raise RuntimeError("Terjadi overlap antara train dan val")
    return train_ids, val_ids
