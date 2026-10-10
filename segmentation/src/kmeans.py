"""
Clustering K-Means untuk segmentasi citra.

Tujuan: min sum_{k} sum_{x in Ck} ||x - mu_k||^2

Referensi: https://docs.opencv.org/4.x/d1/d5c/tutorial_py_kmeans_opencv.html
"""

import cv2
import numpy as np


def kmeans_segmentation(
    image: np.ndarray, k: int = 2, attempts: int = 10
) -> np.ndarray:
    """
    Segmentasi K-Means pada intensitas piksel.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    k : int
        Jumlah cluster (default: 2)
    attempts : int
        Jumlah percobaan dengan pusat awal berbeda (default: 10)

    Returns:
    --------
    np.ndarray
        Peta label uint8, nilai 0..k-1 diskalakan ke 0..255
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    if k < 2:
        raise ValueError("k minimal 2")

    samples = image.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    best_labels = np.empty((samples.shape[0],), dtype=np.int32)
    _, labels, _ = cv2.kmeans(
        samples, k, best_labels, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    labels = labels.reshape(image.shape).astype(np.uint8)
    if k > 1:
        labels = (labels * (255 // (k - 1))).astype(np.uint8)

    return labels


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kmeans.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    for k in [2, 3, 4]:
        result = kmeans_segmentation(image, k)
        filename = f"hasil_kmeans_k{k}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
