"""
Region Growing untuk segmentasi berbasis kemiripan intensitas.

Kriteria tumbuh: |f(p) - mean(region)| <= thresh
8-connectivity, BFS dari seed point.

Referensi: https://www.sciencedirect.com/topics/computer-science/region-growing
"""

from collections import deque

import cv2
import numpy as np


def region_growing(
    image: np.ndarray,
    seeds: list[tuple[int, int]] | None = None,
    thresh: int = 10,
) -> np.ndarray:
    """
    Region Growing dari seed point berdasarkan toleransi intensitas.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    seeds : list of (row, col) atau None
        Titik awal tumbuh. None = seed otomatis di tengah citra
        (default: None)
    thresh : int
        Toleransi selisih intensitas terhadap mean region (default: 10)

    Returns:
    --------
    np.ndarray
        Mask biner uint8 (0 / 255)
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    h, w = image.shape
    if seeds is None:
        seeds = [(h // 2, w // 2)]

    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    queue: deque[tuple[int, int]] = deque()

    region_sum = 0.0
    region_count = 0
    for r, c in seeds:
        if 0 <= r < h and 0 <= c < w and not visited[r, c]:
            visited[r, c] = True
            queue.append((r, c))
            region_sum += float(image[r, c])
            region_count += 1

    if region_count == 0:
        raise ValueError("Seed di luar citra")

    while queue:
        r, c = queue.popleft()
        region_mean = region_sum / region_count
        if abs(float(image[r, c]) - region_mean) <= thresh:
            mask[r, c] = 255
            region_sum += float(image[r, c])
            region_count += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

    return mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python region_growing.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    for thresh in [5, 10, 20]:
        result = region_growing(image, thresh=thresh)
        filename = f"hasil_region_growing_t{thresh}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
