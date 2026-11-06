from __future__ import annotations

from src.data.split import split_ids


def test_split_is_disjoint() -> None:
    ids = [f"image{i:03d}" for i in range(100)]
    train_ids, val_ids = split_ids(ids, val_ratio=0.15, seed=42)
    assert set(train_ids).isdisjoint(val_ids)
    assert set(train_ids) | set(val_ids) == set(ids)


def test_split_is_deterministic() -> None:
    ids = [f"image{i:03d}" for i in range(50)]
    first = split_ids(ids, val_ratio=0.2, seed=7)
    second = split_ids(list(reversed(ids)), val_ratio=0.2, seed=7)
    assert first == second


def test_split_no_duplicates() -> None:
    ids = ["a", "a", "b", "c", "d"]
    train_ids, val_ids = split_ids(ids, val_ratio=0.25, seed=1)
    assert sorted(train_ids + val_ids) == ["a", "b", "c", "d"]
