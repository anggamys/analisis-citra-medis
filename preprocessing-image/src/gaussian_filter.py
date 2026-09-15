"""
Gaussian Filter menggunakan convolution dengan Gaussian kernel.

Formula: G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))

Referensi: https://link.springer.com/article/10.1007/s10851-024-01196-9
"""

import cv2
import numpy as np


def gaussian_filter(
    image: np.ndarray, kernel_size: tuple[int, int] = (3, 3), sigma: float = 0
) -> np.ndarray:
    """
    Gaussian Filter menggunakan convolution dengan Gaussian kernel.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    kernel_size : Tuple[int, int]
        Ukuran kernel (default: (3, 3))
    sigma : float
        Standard deviation Gaussian (default: 0, dihitung otomatis)

    Returns:
    --------
    np.ndarray
        Citra hasil Gaussian filtering
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    return cv2.GaussianBlur(image, kernel_size, sigma)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python gaussian_filter.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    for kernel_size in kernel_sizes:
        result = gaussian_filter(image, kernel_size)
        filename = f"hasil_gaussian_{kernel_size[0]}x{kernel_size[1]}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
