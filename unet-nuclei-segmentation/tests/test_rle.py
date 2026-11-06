from __future__ import annotations

import numpy as np
from src.data.rle import rle_decode, rle_encode


def test_roundtrip_simple_mask() -> None:
    mask = np.zeros((5, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, height=5, width=4)
    assert np.array_equal(decoded, mask)


def test_column_major_order() -> None:
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    mask[2, 0] = 1
    assert rle_encode(mask) == "1 1 3 1"


def test_empty_mask() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    assert rle_encode(mask) == ""
    assert rle_decode("", 4, 4).sum() == 0


def test_decode_known_rle() -> None:
    decoded = rle_decode("1 1 3 1", height=3, width=3)
    expected = np.zeros((3, 3), dtype=np.uint8)
    expected[0, 0] = 1
    expected[2, 0] = 1
    assert np.array_equal(decoded, expected)
