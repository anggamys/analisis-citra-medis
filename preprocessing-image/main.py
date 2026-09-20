"""
Main script untuk mendemonstrasikan semua teknik preprocessing.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

from src import (
    clahe,
    gamma_correction,
    gaussian_filter,
    histogram_equalization,
    median_filter,
)


def plot_histograms(images, titles, filename, figsize=(15, 10)):
    """Plot citra beserta histogramnya.

    Args:
        images  : list of np.ndarray - daftar citra grayscale
        titles  : list of str       - judul untuk setiap citra
        filename: str               - nama file output
        figsize : tuple             - ukuran figure (default: (15, 10))
    """
    n_images = len(images)
    _, axes = plt.subplots(2, n_images, figsize=figsize)

    for i, (title, image) in enumerate(zip(titles, images)):
        axes[0, i].imshow(image, cmap="gray")
        axes[0, i].set_title(title)
        axes[0, i].axis("off")

        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        axes[1, i].plot(hist)
        axes[1, i].set_title(f"Histogram {title}")
        axes[1, i].set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def demonstrate_histogram_equalization(image, name):
    """Histogram Equalization - menyebar distribusi intensitas agar kontras lebih merata.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        np.ndarray - citra hasil HE
    """
    print("  Histogram Equalization")

    he_result = histogram_equalization(image)

    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    axes[1].imshow(he_result, cmap="gray")
    axes[1].set_title("Hasil HE")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_histogram_equalization.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    return he_result


def demonstrate_clahe(image, name):
    """CLAHE - HE lokal dengan batas clipping untuk hindari noise berlebih.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"clip_1.0": np.ndarray, "clip_2.0": np.ndarray, "clip_4.0": np.ndarray}
    """
    print("  CLAHE")

    clip_limits = [1.0, 2.0, 4.0]
    results = {}

    _, axes = plt.subplots(1, len(clip_limits) + 1, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    for i, clip_limit in enumerate(clip_limits):
        result = clahe(image, clip_limit)
        results[f"clip_{clip_limit}"] = result

        axes[i + 1].imshow(result, cmap="gray")
        axes[i + 1].set_title(f"CLAHE (clip={clip_limit})")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_clahe_comparison.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    return results


def demonstrate_gamma_correction(image, name):
    """Gamma Correction - transformasi power-law untuk brighten/darken citra.

    Gamma < 1 = lebih terang, gamma > 1 = lebih gelap.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"gamma_0.5": np.ndarray, "gamma_1.0": np.ndarray, "gamma_2.0": np.ndarray}
    """
    print("  Gamma Correction")

    gamma_values = [0.5, 1.0, 2.0]
    results = {}

    _, axes = plt.subplots(1, len(gamma_values) + 1, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    for i, gamma in enumerate(gamma_values):
        result = gamma_correction(image, gamma)
        results[f"gamma_{gamma}"] = result

        axes[i + 1].imshow(result, cmap="gray")
        axes[i + 1].set_title(f"Gamma={gamma}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_gamma_correction.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    return results


def demonstrate_median_filter(image, name):
    """Median Filter - reduksi noise salt-and-pepper dengan ambil median tetangga.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"kernel_3": np.ndarray, "kernel_5": np.ndarray, "kernel_7": np.ndarray}
    """
    print("  Median Filter")

    kernel_sizes = [3, 5, 7]
    results = {}

    _, axes = plt.subplots(1, len(kernel_sizes) + 1, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    for i, kernel_size in enumerate(kernel_sizes):
        result = median_filter(image, kernel_size)
        results[f"kernel_{kernel_size}"] = result

        axes[i + 1].imshow(result, cmap="gray")
        axes[i + 1].set_title(f"Median (kernel={kernel_size})")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_median_filter.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def demonstrate_gaussian_filter(image, name):
    """Gaussian Filter - penghalusan citra dengan konvolusi Gaussian kernel.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"kernel_3x3": np.ndarray, "kernel_5x5": np.ndarray, "kernel_7x7": np.ndarray}
    """
    print("  Gaussian Filter")

    kernel_sizes = [(3, 3), (5, 5), (7, 7)]
    results = {}

    _, axes = plt.subplots(1, len(kernel_sizes) + 1, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    for i, kernel_size in enumerate(kernel_sizes):
        result = gaussian_filter(image, kernel_size)
        results[f"kernel_{kernel_size[0]}x{kernel_size[1]}"] = result

        axes[i + 1].imshow(result, cmap="gray")
        axes[i + 1].set_title(f"Gaussian (kernel={kernel_size})")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_gaussian_filter.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    return results


def process_image(image_path):
    """Proses satu citra dengan semua teknik preprocessing.

    Args:
        image_path: Path - path ke file citra
    """
    name = image_path.stem
    print(f"\n[{name}]")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"  Error: Gagal membaca citra {image_path}")
        return

    he_result = demonstrate_histogram_equalization(image, name)
    clahe_results = demonstrate_clahe(image, name)
    gamma_results = demonstrate_gamma_correction(image, name)
    median_results = demonstrate_median_filter(image, name)
    gaussian_results = demonstrate_gaussian_filter(image, name)

    images_to_compare = {
        "Asli": image,
        "HE": he_result,
        "CLAHE (clip=2.0)": clahe_results["clip_2.0"],
        "Gamma=0.5": gamma_results["gamma_0.5"],
        "Median (3x3)": median_results["kernel_3"],
        "Gaussian (3x3)": gaussian_results["kernel_3x3"],
    }

    plot_histograms(
        images_to_compare.values(),
        images_to_compare.keys(),
        f"{name}_histograms_comparison.png",
        figsize=(18, 10),
    )


def main():
    print("Memulai demonstrasi preprocessing citra...")

    image_paths = sorted(
        p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        print(f"Tidak ditemukan citra .png di {DATA_DIR}")
        return

    print(f"Ditemukan {len(image_paths)} citra")

    for image_path in image_paths:
        process_image(image_path)

    print(f"\nSelesai! Semua hasil disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
