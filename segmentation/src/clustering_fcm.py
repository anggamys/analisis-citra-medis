"""
Clustering (Fuzzy C-Means) untuk segmentasi clustering.

Tujuan: J = sum_i sum_k u_ik^m * ||x_k - v_i||^2
Update: v_i = sum(u_ik^m * x_k) / sum(u_ik^m)
        u_ik = 1 / sum_j (||x_k - v_i|| / ||x_k - v_j||)^(2/(m-1))

Berbeda dari K-Means: tiap piksel punya derajat keanggotaan di semua cluster.

Referensi: https://en.wikipedia.org/wiki/Fuzzy_clustering#Fuzzy_C-means
"""

import cv2
import numpy as np


def clustering_fcm(
    image: np.ndarray,
    c: int = 2,
    m: float = 2.0,
    max_iter: int = 100,
    epsilon: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """
    Segmentasi Fuzzy C-Means pada intensitas piksel.

    Parameters:
    -----------
    image : np.ndarray
        Citra grayscale dalam format uint8
    c : int
        Jumlah cluster (default: 2)
    m : float
        Eksponen fuzziness > 1 (default: 2.0)
    max_iter : int
        Iterasi maksimum (default: 100)
    epsilon : float
        Batas perubahan membership untuk berhenti (default: 1e-3)
    seed : int
        Seed RNG agar hasil reproduksibel (default: 0)

    Returns:
    --------
    np.ndarray
        Peta label uint8 (argmax membership), diskalakan ke 0..255
    """
    if len(image.shape) != 2:
        raise ValueError("Image harus grayscale")

    if c < 2:
        raise ValueError("c minimal 2")

    if m <= 1:
        raise ValueError("m harus > 1")

    rng = np.random.default_rng(seed)
    x = image.reshape(-1).astype(np.float64)
    n = x.size

    # Inisialisasi centroid dari piksel acak distinct
    centers = rng.choice(x, size=c, replace=False)

    power = 2.0 / (m - 1)
    u = np.zeros((c, n))
    for _ in range(max_iter):
        dist = np.abs(x[None, :] - centers[:, None])
        dist = np.maximum(dist, 1e-10)
        denom = (dist[:, None, :] / dist[None, :, :]) ** power
        u_new = 1.0 / denom.sum(axis=1)

        um = u_new**m
        centers = (um * x).sum(axis=1) / um.sum(axis=1)

        if np.linalg.norm(u_new - u) < epsilon:
            u = u_new
            break
        u = u_new

    labels = u.argmax(axis=0).reshape(image.shape).astype(np.uint8)
    if c > 1:
        labels = (labels * (255 // (c - 1))).astype(np.uint8)

    return labels


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python clustering_fcm.py <image_path>")
        sys.exit(1)

    image = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Tidak dapat membuka citra {sys.argv[1]}")
        sys.exit(1)

    for c in [2, 3, 4]:
        result = clustering_fcm(image, c)
        filename = f"hasil_clustering_fcm_c{c}.png"
        cv2.imwrite(filename, result)
        print(f"Disimpan: {filename}")
