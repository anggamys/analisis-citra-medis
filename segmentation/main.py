"""
Main script untuk mendemonstrasikan semua teknik segmentasi.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "preprocessing-image" / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

from src import (
    active_contour_seg,
    kmeans_segmentation,
    region_growing,
    thresholding,
    watershed_segmentation,
)

SNAKE_PARAMS = [(0.015, 10.0), (0.05, 5.0), (0.01, 20.0)]


def _overlay_boundaries(image, markers):
    """Overlay batas watershed (marker -1) merah di atas citra grayscale.

    Args:
        image  : np.ndarray - citra grayscale uint8
        markers: np.ndarray - marker int32 hasil watershed

    Returns:
        np.ndarray - citra RGB dengan batas merah
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb[markers == -1] = (255, 0, 0)
    return rgb


def _overlay_snake(image, snake):
    """Overlay kontur snake merah di atas citra grayscale.

    Args:
        image: np.ndarray - citra grayscale uint8
        snake: np.ndarray - array (N, 2) koordinat (row, col)

    Returns:
        np.ndarray - citra RGB dengan kontur merah
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    pts = snake[:, ::-1].astype(int)
    cv2.polylines(rgb, [pts], True, (255, 0, 0), 2)
    return rgb


def demonstrate_thresholding(image, name):
    """Thresholding - pisah foreground/background via ambang intensitas.

    Varian: Otsu, fixed=127, adaptive-Gaussian.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"otsu": mask, "fixed": mask, "adaptive": mask}
    """
    print("  Thresholding")

    variants = {
        "Otsu": thresholding(image, "otsu"),
        "Fixed=127": thresholding(image, "fixed", thresh_value=127),
        "Adaptive": thresholding(image, "adaptive"),
    }

    _, axes = plt.subplots(1, len(variants) + 1, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, (title, mask) in enumerate(variants.items()):
        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(title)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_thresholding.png", dpi=150, bbox_inches="tight")
    plt.close()

    return variants


def demonstrate_region_growing(image, name):
    """Region Growing - tumbuhkan region dari seed selama mirip intensitas.

    Varian: thresh 5, 10, 20 (seed otomatis di tengah citra).

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"t5": mask, "t10": mask, "t20": mask}
    """
    print("  Region Growing")

    thresh_values = [5, 10, 20]
    results = {}

    _, axes = plt.subplots(1, len(thresh_values) + 1, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, thresh in enumerate(thresh_values):
        mask = region_growing(image, thresh=thresh)
        results[f"t{thresh}"] = mask
        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(f"thresh={thresh}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_region_growing.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def demonstrate_kmeans(image, name):
    """Clustering K-Means - kelompokkan piksel berdasar intensitas.

    Varian: k=2, 3, 4.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"k2": label, "k3": label, "k4": label}
    """
    print("  K-Means")

    k_values = [2, 3, 4]
    results = {}

    _, axes = plt.subplots(1, len(k_values) + 1, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, k in enumerate(k_values):
        labels = kmeans_segmentation(image, k)
        results[f"k{k}"] = labels
        axes[i + 1].imshow(labels, cmap="jet")
        axes[i + 1].set_title(f"k={k}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_kmeans.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def demonstrate_watershed(image, name):
    """Watershed - segmentasi berbasis marker dari distance transform.

    Varian: dist_ratio 0.5, 0.7.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"r0.5": markers, "r0.7": markers}
    """
    print("  Watershed")

    ratios = [0.5, 0.7]
    results = {}

    _, axes = plt.subplots(1, len(ratios) + 1, figsize=(12, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, ratio in enumerate(ratios):
        markers = watershed_segmentation(image, ratio)
        results[f"r{ratio}"] = markers
        axes[i + 1].imshow(_overlay_boundaries(image, markers))
        axes[i + 1].set_title(f"dist={ratio}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_watershed.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def demonstrate_active_contour(image, name):
    """Active Contour (Snakes) - evolusi kontur minimalkan energi.

    Varian: 3 set (alpha, beta), init lingkaran otomatis.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"a.._b..": snake array}
    """
    print("  Active Contour")

    results = {}

    _, axes = plt.subplots(1, len(SNAKE_PARAMS) + 1, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, (alpha, beta) in enumerate(SNAKE_PARAMS):
        snake = active_contour_seg(image, alpha=alpha, beta=beta)
        results[f"a{alpha}_b{beta}"] = snake
        axes[i + 1].imshow(_overlay_snake(image, snake))
        axes[i + 1].set_title(f"a={alpha} b={beta}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_active_contour.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def plot_summary(image, name, representatives):
    """Plot ringkasan 1 varian wakil tiap metode.

    Args:
        image          : np.ndarray - citra grayscale uint8
        name           : str        - nama/identifier citra
        representatives: dict - {"Thresholding": img, ...} citra siap tampil
    """
    titles = ["Asli"] + list(representatives.keys())
    panels = [image] + list(representatives.values())

    _, axes = plt.subplots(1, len(panels), figsize=(18, 5))
    for ax, title, panel in zip(axes, titles, panels):
        if panel.ndim == 2:
            ax.imshow(panel, cmap="gray")
        else:
            ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()


def process_image(image_path):
    """Proses satu citra dengan semua teknik segmentasi.

    Args:
        image_path: Path - path ke file citra
    """
    name = image_path.stem
    print(f"\n[{name}]")

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"  Error: Gagal membaca citra {image_path}")
        return

    thresh = demonstrate_thresholding(image, name)
    region = demonstrate_region_growing(image, name)
    km = demonstrate_kmeans(image, name)
    ws = demonstrate_watershed(image, name)
    snake = demonstrate_active_contour(image, name)

    default_snake = snake[f"a{SNAKE_PARAMS[0][0]}_b{SNAKE_PARAMS[0][1]}"]
    plot_summary(
        image,
        name,
        {
            "Thresholding": thresh["Otsu"],
            "Region": region["t10"],
            "KMeans": km["k2"],
            "Watershed": _overlay_boundaries(image, ws["r0.7"]),
            "Snakes": _overlay_snake(image, default_snake),
        },
    )


def main():
    print("Memulai demonstrasi segmentasi citra...")

    image_paths = sorted(
        p for p in DATA_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not image_paths:
        print(f"Tidak ditemukan citra di {DATA_DIR}")
        return

    print(f"Ditemukan {len(image_paths)} citra")

    for image_path in image_paths:
        process_image(image_path)

    print(f"\nSelesai! Semua hasil disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
