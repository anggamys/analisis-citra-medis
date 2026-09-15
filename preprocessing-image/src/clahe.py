"""
Contrast Limited Adaptive Histogram Equalization (CLAHE).

Formula: T(i) = (L-1)/N * sum_{j=0}^{i} H_c(j)

Referensi: https://link.springer.com/article/10.1007/s11554-024-01465-1
"""

import cv2
import numpy as np


def clahe(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE).

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    clip_limit : float
        Batas clipping untuk histogram (default: 2.0)
    tile_grid_size : Tuple[int, int]
        Ukuran grid tile (default: (8, 8))

    Returns:
    --------
    np.ndarray
        Citra hasil CLAHE
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    return clahe_obj.apply(image)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python clahe.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    clip_limits = [1.0, 2.0, 4.0]
    for clip_limit in clip_limits:
        result = clahe(image, clip_limit)
        filename = f"hasil_clahe_clip{clip_limit}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
