"""
Main script untuk mendemonstrasikan semua teknik segmentasi.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "infected"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Sama dengan Soal 2: 3 citra unik pertama (file dataset berpasangan duplikat)
SAMPLES = ["Image_001.jpg", "Image_003.jpg", "Image_005.jpg"]

from src import (
    clustering_fcm,
    clustering_kmeans,
    growing_segmentation,
    snakes_segmentation,
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


def demonstrate_growing(image, name):
    """Growing (Region Growing) - tumbuhkan region dari seed selama mirip intensitas.

    Varian: thresh 20, 40, 60 (seed otomatis piksel terang + blur 5x5).

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"t20": mask, "t40": mask, "t60": mask}
    """
    print("  Growing")

    thresh_values = [20, 40, 60]
    results = {}

    _, axes = plt.subplots(1, len(thresh_values) + 1, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, thresh in enumerate(thresh_values):
        mask = growing_segmentation(image, thresh=thresh)
        results[f"t{thresh}"] = mask
        axes[i + 1].imshow(mask, cmap="gray")
        axes[i + 1].set_title(f"thresh={thresh}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_growing.png", dpi=150, bbox_inches="tight")
    plt.close()

    return results


def demonstrate_clustering_kmeans(image, name):
    """Clustering (K-Means) - kelompokkan piksel berdasar intensitas.

    Varian: k=2, 3, 4.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"k2": label, "k3": label, "k4": label}
    """
    print("  Clustering (K-Means)")

    k_values = [2, 3, 4]
    results = {}

    _, axes = plt.subplots(1, len(k_values) + 1, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, k in enumerate(k_values):
        labels = clustering_kmeans(image, k)
        results[f"k{k}"] = labels
        axes[i + 1].imshow(labels, cmap="jet")
        axes[i + 1].set_title(f"k={k}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_clustering_kmeans.png", dpi=150, bbox_inches="tight"
    )
    plt.close()

    return results


def demonstrate_clustering_fcm(image, name):
    """Clustering (Fuzzy C-Means) - tiap piksel punya derajat keanggotaan semua cluster.

    Varian: c=2, 3, 4 (m=2.0, seed fixed).

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"c2": label, "c3": label, "c4": label}
    """
    print("  Clustering (FCM)")

    c_values = [2, 3, 4]
    results = {}

    _, axes = plt.subplots(1, len(c_values) + 1, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, c in enumerate(c_values):
        labels = clustering_fcm(image, c)
        results[f"c{c}"] = labels
        axes[i + 1].imshow(labels, cmap="jet")
        axes[i + 1].set_title(f"c={c}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_clustering_fcm.png", dpi=150, bbox_inches="tight")
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


def demonstrate_snakes(image, name):
    """Snakes (Active Contour) - evolusi kontur minimalkan energi.

    Varian: 3 set (alpha, beta), init lingkaran otomatis.

    Args:
        image: np.ndarray - citra grayscale uint8
        name : str        - nama/identifier citra

    Returns:
        dict - {"a.._b..": snake array}
    """
    print("  Snakes")

    results = {}

    _, axes = plt.subplots(1, len(SNAKE_PARAMS) + 1, figsize=(15, 5))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Citra Asli")
    axes[0].axis("off")
    for i, (alpha, beta) in enumerate(SNAKE_PARAMS):
        snake = snakes_segmentation(image, alpha=alpha, beta=beta)
        results[f"a{alpha}_b{beta}"] = snake
        axes[i + 1].imshow(_overlay_snake(image, snake))
        axes[i + 1].set_title(f"a={alpha} b={beta}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_snakes.png", dpi=150, bbox_inches="tight")
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

    _, axes = plt.subplots(1, len(panels), figsize=(21, 5))
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
    growing = demonstrate_growing(image, name)
    ckm = demonstrate_clustering_kmeans(image, name)
    cfcm = demonstrate_clustering_fcm(image, name)
    ws = demonstrate_watershed(image, name)
    snake = demonstrate_snakes(image, name)

    default_snake = snake[f"a{SNAKE_PARAMS[0][0]}_b{SNAKE_PARAMS[0][1]}"]
    plot_summary(
        image,
        name,
        {
            "Thresholding": thresh["Otsu"],
            "Growing": growing["t40"],
            "Clustering (K-Means)": ckm["k2"],
            "Clustering (FCM)": cfcm["c2"],
            "Watershed": _overlay_boundaries(image, ws["r0.7"]),
            "Snakes": _overlay_snake(image, default_snake),
        },
    )


def main():
    print("Memulai demonstrasi segmentasi citra...")

    image_paths = [DATA_DIR / f for f in SAMPLES]
    missing = [str(p) for p in image_paths if not p.exists()]
    if missing:
        print(f"Citra tidak ditemukan: {missing}")
        return

    print(f"Ditemukan {len(image_paths)} citra")

    for image_path in image_paths:
        process_image(image_path)

    print(f"\nSelesai! Semua hasil disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
