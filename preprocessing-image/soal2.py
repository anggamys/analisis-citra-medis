"""
Soal 2: Enhancement 3 citra USG infected (HE, CLAHE, Gamma Correction).

Sampel fixed: Image_001.jpg, Image_003.jpg, Image_005.jpg dari dataset/infected/.
Catatan: file dataset berpasangan duplikat (001=002, 003=004, ...),
sehingga dipilih 3 citra unik pertama.
Output: visualisasi sebelum-sesudah + histogram + metrik kuantitatif (CSV).
"""

import csv
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "dataset" / "infected"
OUTPUT_DIR = SCRIPT_DIR / "output" / "soal2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from src import clahe, gamma_correction, histogram_equalization

SAMPLES = ["Image_001.jpg", "Image_003.jpg", "Image_005.jpg"]
GAMMA_VALUES = [0.5, 1.0, 2.0]


def compute_metrics(image):
    """Hitung metrik kuantitatif citra grayscale.

    Args:
        image: np.ndarray - citra grayscale uint8

    Returns:
        dict - {"mean": float, "std": float, "entropy": float}
    """
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)))
    return {
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "entropy": entropy,
    }


def enhance(image):
    """Terapkan 3 metode enhancement ke satu citra.

    Args:
        image: np.ndarray - citra grayscale uint8

    Returns:
        dict - {"Asli": img, "HE": img, "CLAHE": img,
                "Gamma=0.5": img, "Gamma=1.0": img, "Gamma=2.0": img}
    """
    results = {
        "Asli": image,
        "HE": histogram_equalization(image),
        "CLAHE": clahe(image, clip_limit=2.0),
    }
    for gamma in GAMMA_VALUES:
        results[f"Gamma={gamma}"] = gamma_correction(image, gamma)
    return results


def plot_before_after(results, name):
    """Plot sebelum-sesudah + histogram tiap varian.

    Args:
        results: dict - nama metode -> citra grayscale
        name   : str  - nama/identifier citra
    """
    titles = list(results.keys())
    _, axes = plt.subplots(len(titles), 2, figsize=(10, 4 * len(titles)))

    for i, title in enumerate(titles):
        img = results[title]
        axes[i, 0].imshow(img, cmap="gray")
        axes[i, 0].set_title(title)
        axes[i, 0].axis("off")

        hist, _ = np.histogram(img.flatten(), bins=256, range=(0, 256))
        axes[i, 1].plot(hist)
        axes[i, 1].set_title(f"Histogram {title}")
        axes[i, 1].set_xlim([0, 256])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_before_after.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_comparison_all(all_results):
    """Plot ringkasan semua citra dengan metode wakil.

    Args:
        all_results: dict - nama citra -> dict hasil enhance()
    """
    methods = ["Asli", "HE", "CLAHE", "Gamma=0.5"]
    names = list(all_results.keys())

    panels = [(name, method) for name in names for method in methods]
    nrows = (len(panels) + 1) // 2
    _, axes = plt.subplots(nrows, 2, figsize=(10, 5 * nrows))
    for ax, (name, method) in zip(axes.flatten(), panels):
        ax.imshow(all_results[name][method], cmap="gray")
        ax.set_title(f"{name} - {method}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "comparison_all.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("Soal 2: enhancement 3 citra infected...")

    all_results = {}
    rows = []

    for filename in SAMPLES:
        name = Path(filename).stem
        print(f"\n[{name}]")

        image = cv2.imread(str(DATA_DIR / filename), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"  Error: Gagal membaca citra {filename}")
            continue

        results = enhance(image)
        all_results[name] = results
        plot_before_after(results, name)

        for method, img in results.items():
            m = compute_metrics(img)
            rows.append(
                {
                    "image": filename,
                    "method": method,
                    "mean": round(m["mean"], 2),
                    "std": round(m["std"], 2),
                    "entropy": round(m["entropy"], 3),
                }
            )
            print(
                f"  {method}: mean={m['mean']:.2f} "
                f"std={m['std']:.2f} entropy={m['entropy']:.3f}"
            )

    if not all_results:
        print("Tidak ada citra berhasil diproses.")
        return

    plot_comparison_all(all_results)

    csv_path = OUTPUT_DIR / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "method", "mean", "std", "entropy"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSelesai! {len(all_results)} citra, {len(rows)} baris metrik.")
    print(f"Hasil disimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
