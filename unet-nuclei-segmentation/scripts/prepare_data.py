"""Gabungkan mask instance, validasi dataset, dan buat split train/val."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from src.data.manifest import write_manifest
from src.data.preprocessing import merge_instance_masks
from src.data.split import split_ids
from src.utils.config import PROJECT_ROOT, load_config, resolve_path
from src.utils.io import save_json
from tqdm import tqdm

STAGE1_TRAIN_URL = "https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip"


def download_stage1_train(train_raw: Path) -> None:
    """Unduh dan ekstrak stage1_train resmi ke ``train_raw``."""
    train_raw.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
        archive_path = Path(handle.name)

    try:
        print(f"Mengunduh dataset: {STAGE1_TRAIN_URL}")
        urllib.request.urlretrieve(STAGE1_TRAIN_URL, archive_path)
        print(f"Mengekstrak ke: {train_raw}")
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                parts = Path(member.filename).parts
                if parts and parts[0] == "stage1_train":
                    parts = parts[1:]
                if not parts:
                    continue

                target = train_raw.joinpath(*parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    finally:
        archive_path.unlink(missing_ok=True)


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def discover_images(train_raw: Path) -> tuple[dict[str, dict], list[str]]:
    entries: dict[str, dict] = {}
    errors: list[str] = []

    for folder in sorted(train_raw.iterdir()):
        if not folder.is_dir():
            continue
        images = sorted((folder / "images").glob("*.png"))
        masks = sorted((folder / "masks").glob("*.png"))
        if len(images) != 1:
            errors.append(f"{folder.name}: jumlah image={len(images)}")
            continue
        if not masks:
            errors.append(f"{folder.name}: tidak ada mask")
            continue
        entries[folder.name] = {"image": images[0], "masks": masks}

    return entries, errors


def read_shape(image_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Gagal membaca citra: {image_path}")
    return int(image.shape[0]), int(image.shape[1])


def build_rows(
    entries: dict[str, dict],
    mask_dir: Path,
    force: bool,
) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    sizes: Counter = Counter()
    foreground_ratios: list[float] = []
    instance_counts: list[int] = []

    for image_id, entry in tqdm(sorted(entries.items()), desc="Menyiapkan mask"):
        image_path = entry["image"]
        height, width = read_shape(image_path)
        sizes[(height, width)] += 1

        mask_path = mask_dir / f"{image_id}.png"
        if force or not mask_path.exists():
            merged = merge_instance_masks(entry["masks"], shape=(height, width))
            if not np.any(merged):
                raise ValueError(f"{image_id}: mask gabungan kosong")
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(mask_path), merged):
                raise OSError(f"Gagal menyimpan mask: {mask_path}")
        else:
            merged = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if merged is None or merged.shape != (height, width):
                raise ValueError(f"{image_id}: mask lama tidak valid")

        foreground = float((merged > 0).mean())
        foreground_ratios.append(foreground)
        instance_counts.append(len(entry["masks"]))

        rows.append(
            {
                "image_id": image_id,
                "image_path": relative_path(image_path),
                "mask_path": relative_path(mask_path),
                "height": height,
                "width": width,
                "foreground_ratio": round(foreground, 6),
                "num_instances": len(entry["masks"]),
            }
        )

    size_list = [
        {"height": height, "width": width, "count": count}
        for (height, width), count in sizes.most_common()
    ]
    stats = {
        "image_sizes": size_list,
        "foreground_ratio": {
            "min": float(np.min(foreground_ratios)),
            "max": float(np.max(foreground_ratios)),
            "mean": float(np.mean(foreground_ratios)),
            "median": float(np.median(foreground_ratios)),
        },
        "instances_per_image": {
            "min": int(np.min(instance_counts)),
            "max": int(np.max(instance_counts)),
            "mean": float(np.mean(instance_counts)),
            "total": int(np.sum(instance_counts)),
        },
    }
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Unduh stage1_train resmi jika data-train masih kosong.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config["paths"]
    data_cfg = config["data"]

    train_raw = resolve_path(paths["train_raw"])
    processed_dir = resolve_path(paths["processed_dir"])
    splits_dir = resolve_path(paths["splits_dir"])
    mask_dir = processed_dir / "labeled" / "masks"
    splits_dir.mkdir(parents=True, exist_ok=True)

    val_ratio = (
        float(args.val_ratio)
        if args.val_ratio is not None
        else float(data_cfg.get("val_ratio", 0.15))
    )
    seed = int(args.seed if args.seed is not None else config["project"]["seed"])

    has_raw_data = train_raw.is_dir() and any(
        path.is_dir() for path in train_raw.iterdir()
    )
    if args.download and not has_raw_data:
        download_stage1_train(train_raw)
    elif args.download:
        print(f"data-train sudah berisi, unduhan dilewati: {train_raw}")

    entries, discovery_errors = discover_images(train_raw)
    if args.limit is not None:
        entries = dict(sorted(entries.items())[: args.limit])
    if not entries:
        raise SystemExit(f"Tidak ada citra train di {train_raw}")

    rows, stats = build_rows(entries, mask_dir, force=args.force)
    train_ids, val_ids = split_ids(
        [row["image_id"] for row in rows],
        val_ratio=val_ratio,
        seed=seed,
    )
    row_by_id = {row["image_id"]: row for row in rows}

    for name, ids in (("train", train_ids), ("val", val_ids)):
        split_rows = []
        for image_id in ids:
            row = dict(row_by_id[image_id])
            row["split"] = name
            split_rows.append(row)
        save_json(
            {
                "split": name,
                "num_images": len(split_rows),
                "image_ids": ids,
            },
            processed_dir / f"split_{name}.json",
        )
        write_manifest(
            splits_dir / f"{name}.csv",
            split_rows,
            fieldnames=[
                "image_id",
                "image_path",
                "mask_path",
                "split",
                "height",
                "width",
                "foreground_ratio",
                "num_instances",
            ],
        )

    report = {
        "train_raw": str(train_raw),
        "num_images": len(rows),
        "num_train": len(train_ids),
        "num_val": len(val_ids),
        "val_ratio": val_ratio,
        "seed": seed,
        "discovery_errors": discovery_errors,
        **stats,
    }
    save_json(report, processed_dir / "dataset_report.json")

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Total: {len(rows)}")
    print(f"Manifest: {splits_dir}")
    print(f"Report  : {processed_dir / 'dataset_report.json'}")
    if discovery_errors:
        print("Ditemukan error pada dataset:", file=sys.stderr)
        for message in discovery_errors:
            print(f"  - {message}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
