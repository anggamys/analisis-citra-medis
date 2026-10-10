"""
Thresholding untuk segmentasi citra.

Formula (global): g(x,y) = 255 jika f(x,y) > T, else 0
Otsu: T* = argmax w0(T)*w1(T)*(mu0-mu1)^2

Referensi: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
"""

import cv2
import numpy as np


def thresholding(
    image: np.ndarray,
    method: str = "otsu",
    thresh_value: int = 127,
    block_size: int = 11,
    c: int = 2,
) -> np.ndarray:
    """
    Thresholding untuk segmentasi citra.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    method : str
        'otsu' | 'fixed' | 'adaptive' (default: 'otsu')
    thresh_value : int
        Nilai threshold untuk method 'fixed' (default: 127)
    block_size : int
        Ukuran blok tetangga untuk method 'adaptive', harus ganjil (default: 11)
    c : int
        Konstanta pengurang untuk method 'adaptive' (default: 2)

    Returns:
    --------
    np.ndarray
        Mask biner uint8 (0 / 255)
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    if method == "otsu":
        _, result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "fixed":
        _, result = cv2.threshold(image, thresh_value, 255, cv2.THRESH_BINARY)
    elif method == "adaptive":
        if block_size % 2 == 0:
            raise ValueError("block_size harus ganjil")
        result = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )
    else:
        raise ValueError("method harus 'otsu', 'fixed', atau 'adaptive'")

    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python thresholding.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    for method in ["otsu", "fixed", "adaptive"]:
        result = thresholding(image, method)
        filename = f"hasil_thresholding_{method}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
