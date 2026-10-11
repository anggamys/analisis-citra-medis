"""
Growing (Region Growing) untuk segmentasi berbasis kemiripan intensitas.

Kriteria tumbuh: |f(p) - mean(region)| <= thresh
8-connectivity, BFS dari seed point.

Referensi: https://www.sciencedirect.com/topics/computer-science/region-growing
"""

from collections import deque

import cv2
import numpy as np


def _auto_seed_candidates(
    work: np.ndarray,
    margin_ratio: float = 0.15,
    sat_thresh: float = 250.0,
    n: int = 4,
    min_dist: int = 15,
) -> list[tuple[int, int]]:
    """Kandidat seed otomatis terurut skor tertinggi.

    Skor = mean_lokal - std_lokal (box 15x15): terang DAN homogen.
    Margin tepi + piksel tersaturasi dikecualikan karena artefak USG
    (bar/strip putih, border hitam) menempel di tepi dan jenuh di 255.
    Kandidat dipisah min_dist agar tersebar.

    Args:
        work        : np.ndarray - citra kerja float (sudah di-blur)
        margin_ratio: float - proporsi tepi diabaikan per sisi (default: 0.15)
        sat_thresh  : float - piksel di atas ini bukan kandidat (default: 250.0)
        n           : int - jumlah kandidat (default: 4)
        min_dist    : int - jarak minimum antar kandidat px (default: 15)

    Returns:
        list of (row, col) terurut skor menurun
    """
    h, w = work.shape
    mr, mc = int(h * margin_ratio), int(w * margin_ratio)

    mean_local = cv2.boxFilter(work, -1, (15, 15))
    sqr_local = cv2.boxFilter(work * work, -1, (15, 15))
    std_local = np.sqrt(np.maximum(sqr_local - mean_local * mean_local, 0))

    score = mean_local - std_local
    score[:mr, :] = -np.inf
    score[-mr:, :] = -np.inf
    score[:, :mc] = -np.inf
    score[:, -mc:] = -np.inf
    score[work > sat_thresh] = -np.inf

    flat = score.ravel()
    order = np.argsort(flat)[::-1]
    candidates = []
    for idx in order:
        r, c = int(idx // w), int(idx % w)
        if not np.isfinite(flat[idx]):
            break
        if all(abs(r - pr) + abs(c - pc) >= min_dist for pr, pc in candidates):
            candidates.append((r, c))
        if len(candidates) >= n:
            break

    return candidates


def _grow_from_seed(work: np.ndarray, seed: tuple[int, int], thresh: int) -> np.ndarray:
    """BFS 8-connectivity dari satu seed dengan toleransi mean region.

    Args:
        work  : np.ndarray - citra kerja float
        seed  : (row, col) titik awal
        thresh: int toleransi selisih terhadap mean region

    Returns:
        np.ndarray mask biner uint8 (0 / 255)
    """
    h, w = work.shape
    visited = np.zeros((h, w), dtype=bool)
    mask = np.zeros((h, w), dtype=np.uint8)
    queue: deque[tuple[int, int]] = deque()

    r, c = seed
    if not (0 <= r < h and 0 <= c < w):
        raise ValueError("Seed di luar citra")

    visited[r, c] = True
    queue.append((r, c))
    region_sum = float(work[r, c])
    region_count = 1

    while queue:
        r, c = queue.popleft()
        region_mean = region_sum / region_count
        if abs(float(work[r, c]) - region_mean) <= thresh:
            mask[r, c] = 255
            region_sum += float(work[r, c])
            region_count += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc]:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

    return mask


def growing_segmentation(
    image: np.ndarray,
    seeds: list[tuple[int, int]] | None = None,
    thresh: int = 40,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Region Growing dari seed point berdasarkan toleransi intensitas.

    Citra dihaluskan dulu (Gaussian) agar speckle USG tidak mengunci
    pertumbuhan; mask hasil berukuran sama dengan citra asli.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    seeds : list of (row, col) atau None
        Titik awal tumbuh. None = seed otomatis via skor
        intensitas-homogenitas di ROI tengah (terang dan homogen,
        artefak tepi diabaikan), dengan fallback kandidat berikutnya
        bila region tumbuh < 0,1% citra (default: None)
    thresh : int
        Toleransi selisih intensitas terhadap mean region (default: 40)
    blur_ksize : int
        Ukuran kernel Gaussian blur awal, harus ganjil; 0 = tanpa blur
        (default: 5)

    Returns:
    --------
    np.ndarray
        Mask biner uint8 (0 / 255)
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    if blur_ksize > 0:
        if blur_ksize % 2 == 0:
            raise ValueError("blur_ksize harus ganjil atau 0")
        work = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), 0)
    else:
        work = image
    work = work.astype(np.float64)

    if seeds is not None:
        return _grow_from_seed(work, seeds[0], thresh)

    # Seed otomatis + fallback bila region terlalu kecil
    mask = np.zeros(work.shape, dtype=np.uint8)
    for seed in _auto_seed_candidates(work):
        mask = _grow_from_seed(work, seed, thresh)
        if (mask > 0).mean() >= 0.001:
            return mask

    return mask


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python growing.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    for thresh in [20, 40, 60]:
        result = growing_segmentation(image, thresh=thresh)
        filename = f"hasil_growing_t{thresh}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
