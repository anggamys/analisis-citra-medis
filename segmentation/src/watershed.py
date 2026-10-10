"""
Watershed untuk segmentasi berbasis topografi gradien.

Pipeline: Otsu -> opening -> sure bg (dilate) ->
distance transform (sure fg) -> unknown -> markers -> cv2.watershed.

Referensi: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
"""

import cv2
import numpy as np


def watershed_segmentation(
    image: np.ndarray, dist_ratio: float = 0.7, kernel_size: int = 3
) -> np.ndarray:
    """
    Segmentasi Watershed dengan marker otomatis.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    dist_ratio : float
        Rasio threshold distance transform untuk sure foreground
        (0-1, default: 0.7)
    kernel_size : int
        Ukuran kernel opening (default: 3)

    Returns:
    --------
    np.ndarray
        Marker int32 hasil watershed, batas = -1
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    if not 0 < dist_ratio < 1:
        raise ValueError("dist_ratio harus di antara 0 dan 1")

    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    threshold_value = float(dist_ratio * float(dist.max()))
    _, sure_fg = cv2.threshold(dist, threshold_value, 255.0, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(bgr, markers)

    return markers.astype(np.int32)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python watershed.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    for ratio in [0.5, 0.7]:
        markers = watershed_segmentation(image, ratio)
        result = np.zeros_like(image)
        result[markers == -1] = 255
        filename = f"hasil_watershed_r{ratio}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
