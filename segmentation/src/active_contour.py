"""
Active Contour (Snakes) untuk segmentasi berbasis energi kontur.

Energi: E = E_internal (alpha, beta) + E_external (w_line, w_edge)
Kontur awal lingkaran otomatis di tengah citra.

Referensi: https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.active_contour
"""

import cv2
import numpy as np


def active_contour_seg(
    image: np.ndarray,
    alpha: float = 0.015,
    beta: float = 10.0,
    gamma: float = 0.001,
    n_points: int = 200,
    radius_ratio: float = 0.3,
) -> np.ndarray:
    """
    Active Contour dengan inisialisasi lingkaran otomatis.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    alpha : float
        Bobot elastisitas kontur (default: 0.015)
    beta : float
        Bobot kekakuan kontur (default: 10.0)
    gamma : float
        Ukuran langkah optimasi (default: 0.001)
    n_points : int
        Jumlah titik kontur awal (default: 200)
    radius_ratio : float
        Radius lingkaran awal relatif terhadap min(h, w) (default: 0.3)

    Returns:
    --------
    np.ndarray
        Array (n_points, 2) koordinat (row, col) kontur akhir
    """
    from skimage.segmentation import active_contour

    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    h, w = image.shape
    cr, cc = h / 2, w / 2
    radius = min(h, w) * radius_ratio
    s = np.linspace(0, 2 * np.pi, n_points)
    init = np.stack([cr + radius * np.sin(s), cc + radius * np.cos(s)], axis=1)

    normalized = image.astype(np.float64) / 255.0
    snake = active_contour(normalized, init, alpha=alpha, beta=beta, gamma=gamma)

    return np.asarray(snake)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python active_contour.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    from skimage.draw import polygon2mask

    for alpha, beta in [(0.015, 10.0), (0.05, 5.0), (0.01, 20.0)]:
        snake = active_contour_seg(image, alpha=alpha, beta=beta)
        mask = (polygon2mask(image.shape, snake) * 255).astype(np.uint8)
        filename = f"hasil_snake_a{alpha}_b{beta}.png"
        cv2.imwrite(filename, mask)
        print(f"Disimpan: {filename}")
