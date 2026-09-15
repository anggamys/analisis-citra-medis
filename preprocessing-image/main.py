"""
Main script untuk mendemonstrasikan semua teknik preprocessing.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src import (
    clahe,
    gamma_correction,
    gaussian_filter,
    histogram_equalization,
    median_filter,
)


def create_test_image(size: tuple = (256, 256)) -> np.ndarray:
    """
    Membuat citra test dengan berbagai intensitas.
    """
    x = np.linspace(0, 255, size[1])
    y = np.linspace(0, 255, size[0])
    X, Y = np.meshgrid(x, y)

    image = np.uint8((X + Y) / 2)

    noise = np.random.normal(0, 25, size)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)

    return image


def plot_histograms(images, titles, figsize=(15, 10)):
    """
    Memplot histogram untuk beberapa citra.
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
    plt.savefig("histograms_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def demonstrate_histogram_equalization(image):
    print("=== Histogram Equalization ===")

    he_result = histogram_equalization(image)

    _, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")

    axes[1].imshow(he_result, cmap="gray")
    axes[1].set_title("Hasil HE")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("histogram_equalization.png", dpi=150, bbox_inches="tight")
    plt.show()

    return he_result


def demonstrate_clahe(image):
    print("=== CLAHE ===")

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
    plt.savefig("clahe_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    return results


def demonstrate_gamma_correction(image):
    print("=== Gamma Correction ===")

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
    plt.savefig("gamma_correction.png", dpi=150, bbox_inches="tight")
    plt.show()

    return results


def demonstrate_median_filter(image):
    print("=== Median Filter ===")

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
    plt.savefig("median_filter.png", dpi=150, bbox_inches="tight")
    plt.show()

    return results


def demonstrate_gaussian_filter(image):
    print("=== Gaussian Filter ===")

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
    plt.savefig("gaussian_filter.png", dpi=150, bbox_inches="tight")
    plt.show()

    return results


def main():
    print("Memulai demonstrasi preprocessing citra...")

    image = create_test_image()

    cv2.imwrite("test_image.png", image)
    print("Citra test disimpan: test_image.png")

    he_result = demonstrate_histogram_equalization(image)
    clahe_results = demonstrate_clahe(image)
    gamma_results = demonstrate_gamma_correction(image)
    median_results = demonstrate_median_filter(image)
    gaussian_results = demonstrate_gaussian_filter(image)

    images_to_compare = {
        "Asli": image,
        "HE": he_result,
        "CLAHE (clip=2.0)": clahe_results["clip_2.0"],
        "Gamma=0.5": gamma_results["gamma_0.5"],
        "Median (3x3)": median_results["kernel_3"],
        "Gaussian (3x3)": gaussian_results["kernel_3x3"],
    }

    plot_histograms(
        images_to_compare.values(), images_to_compare.keys(), figsize=(18, 10)
    )

    print("\nSelesai! Semua hasil telah disimpan.")


if __name__ == "__main__":
    main()
