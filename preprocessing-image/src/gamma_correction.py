"""
Gamma Correction (Power-Law Transformation).

Formula: s = c * r^gamma
dimana r = intensitas input (normalized 0-1)

Referensi: https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/iet-ipr.2019.0992
"""

import numpy as np


def gamma_correction(image: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """
    Gamma Correction menggunakan power-law transformation.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    gamma : float
        Parameter gamma (< 1 = lebih terang, > 1 = lebih gelap)
    c : float
        Konstanta (default: 1.0)

    Returns:
    --------
    np.ndarray
        Citra hasil gamma correction
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    # Normalisasi ke range [0, 1]
    normalized = image / 255.0

    # Apply gamma correction: s = c * r^gamma
    corrected = c * np.power(normalized, gamma)

    # Kembali ke range [0, 255]
    return np.asarray(corrected * 255, dtype=np.uint8)


if __name__ == "__main__":
    import sys

    import cv2

    if len(sys.argv) < 2:
        print("Usage: python gamma_correction.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    gamma_values = [0.5, 1.0, 2.0]
    for gamma in gamma_values:
        result = gamma_correction(image, gamma)
        filename = f"hasil_gamma_{gamma}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
