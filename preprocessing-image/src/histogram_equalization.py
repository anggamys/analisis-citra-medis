"""
Histogram Equalization (HE).

Formula: s_k = (L-1) * C(r_k)
dimana C(r_k) = sum_{j=0}^{k} p(r_j)

Referensi: https://pmc.ncbi.nlm.nih.gov/articles/PMC10635952/
"""

import numpy as np


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Histogram Equalization menggunakan CDF sebagai fungsi transformasi.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8

    Returns:
    --------
    np.ndarray
        Citra hasil histogram equalization
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    # Hitung histogram
    hist, _ = np.histogram(image.flatten(), 256, (0, 256))

    # Hitung CDF
    cdf = hist.cumsum()

    # Normalisasi CDF
    cdf_normalized = cdf * 255 / cdf.max()

    # Map intensitas menggunakan CDF
    result = cdf_normalized[image]

    return result.astype(np.uint8)


if __name__ == "__main__":
    import sys

    import cv2

    if len(sys.argv) < 2:
        print("Usage: python histogram_equalization.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    result = histogram_equalization(image)
    cv2.imwrite("hasil_histogram_equalization.png", result)
    print("Disimpan: hasil_histogram_equalization.png")
