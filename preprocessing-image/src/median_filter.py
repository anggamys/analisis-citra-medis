"""
Median Filter untuk reduksi noise.

Formula: g(x,y) = median{f(s,t) : (s,t) ∈ S_xy}

Referensi: https://www.sciencedirect.com/topics/computer-science/median-filtering
"""

import cv2
import numpy as np


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Median Filter untuk reduksi noise.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    kernel_size : int
        Ukuran kernel/window (harus ganjil, default: 3)

    Returns:
    --------
    np.ndarray
        Citra hasil median filtering
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size harus ganjil")

    return cv2.medianBlur(image, kernel_size)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python median_filter.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes:
        result = median_filter(image, kernel_size)
        filename = f"hasil_median_{kernel_size}x{kernel_size}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
