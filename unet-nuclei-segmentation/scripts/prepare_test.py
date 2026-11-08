"""Decode ground truth RLE resmi BBBC038 untuk stage1 dan stage2."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
from src.data.manifest import write_manifest
from src.data.rle import rle_decode
from src.utils.config import PROJECT_ROOT, load_config, resolve_path
from src.utils.io import save_json
from tqdm import tqdm

SOLUTIONS = {
    "stage1": {
        "csv": "stage1_solution.csv",
        "url": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_solution.csv",
        "raw_key": "stage1_test_raw",
        "raw_url": "https://data.broadinstitute.org/bbbc/BBBC038/stage1_test.zip",
        "raw_prefix": "stage1_test",
        "split": "test_stage1",
        "output": "stage1",
        "predict_manifest": "predict_stage1_all.csv",
    },
    "stage2": {
        "csv": "stage2_solution_final.csv",
        "url": (
            "https://data.broadinstitute.org/bbbc/BBBC038/stage2_solution_final.csv"
        ),
        "raw_key": "stage2_test_raw",
        "raw_url": "https://data.broadinstitute.org/bbbc/BBBC038/stage2_test_final.zip",
        "raw_prefix": "stage2_test_final",
        "split": "test_stage2_scored",
        "output": "stage2_scored",
        "predict_manifest": "predict_stage2_all.csv",
    },
}


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def has_raw_images(raw_dir: Path) -> bool:
    return raw_dir.is_dir() and any(
        (folder / "images").is_dir() for folder in raw_dir.iterdir() if folder.is_dir()
    )


def download_and_extract(url: str, destination: Path, prefix: str) -> None:
    """Unduh ZIP dataset resmi lalu ekstrak tanpa folder pembungkus."""
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
        archive_path = Path(handle.name)

    try:
        print(f"Mengunduh dataset: {url}")
        urllib.request.urlretrieve(url, archive_path)
        print(f"Mengekstrak ke: {destination}")
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                parts = Path(member.filename).parts
                if parts and parts[0] == prefix:
                    parts = parts[1:]
                if not parts:
                    continue

                target = destination.joinpath(*parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    finally:
        archive_path.unlink(missing_ok=True)


def ensure_solution(path: Path, url: str, download: bool) -> None:
    if path.is_file():
        return
    if not download:
        raise FileNotFoundError(
            f"File solusi tidak ditemukan: {path}\n"
            f"Jalankan ulang dengan --download atau unduh manual: {url}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Mengunduh {url}")
    urllib.request.urlretrieve(url, path)


def group_rows(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["ImageId"]].append(row)
    return dict(grouped)


def decode_group(image_id: str, rows: list[dict[str, str]]) -> np.ndarray:
    height = int(rows[0]["Height"])
    width = int(rows[0]["Width"])
    for row in rows:
        if int(row["Height"]) != height or int(row["Width"]) != width:
            raise ValueError(f"{image_id}: dimensi RLE tidak konsisten")

    mask = np.zeros((height, width), dtype=np.uint8)
    for row in rows:
        try:
            instance = rle_decode(row.get("EncodedPixels", ""), height, width)
        except ValueError as error:
            raise ValueError(f"{image_id}: RLE tidak valid: {error}") from error
        mask = np.maximum(mask, instance)
    return (mask * 255).astype(np.uint8)


def find_image(raw_dir: Path, image_id: str) -> Path:
    candidates = sorted((raw_dir / image_id / "images").glob("*.png"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"{image_id}: jumlah image={len(candidates)} di {raw_dir}"
        )
    return candidates[0]


def process_stage(
    name: str,
    config: dict[str, Any],
    download: bool,
    force: bool,
    limit: int | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    spec = SOLUTIONS[name]
    paths = config["paths"]
    solutions_dir = resolve_path(paths["solutions_dir"])
    processed_dir = resolve_path(paths["processed_dir"])
    splits_dir = resolve_path(paths["splits_dir"])
    raw_dir = resolve_path(paths[spec["raw_key"]])

    raw_ready = has_raw_images(raw_dir)
    if not raw_ready and download:
        download_and_extract(spec["raw_url"], raw_dir, spec["raw_prefix"])
        raw_ready = has_raw_images(raw_dir)
    if not raw_ready:
        message = f"Data mentah {name} tidak ditemukan: {raw_dir}"
        print(
            f"Peringatan: {message}\n"
            "  Jalankan ulang dengan --download atau sediakan dataset test.",
            file=sys.stderr,
        )
        return [], {
            "stage": name,
            "raw_dir": str(raw_dir),
            "num_solution_images": 0,
            "num_scored": 0,
            "num_ignored": 0,
            "manifest": str(splits_dir / f"{spec['split']}.csv"),
            "errors": [message],
            "skipped": True,
        }

    solution_path = solutions_dir / spec["csv"]
    ensure_solution(solution_path, spec["url"], download=download)

    with solution_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = group_rows(rows)

    mask_dir = processed_dir / "official_test" / spec["output"] / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    ignored = 0
    errors: list[str] = []

    items = sorted(grouped.items())
    if limit is not None:
        items = items[:limit]

    for image_id, group in tqdm(items, desc=f"Memproses {name}"):
        usage = group[0].get("Usage", "")
        if usage == "Ignored":
            ignored += 1
            continue

        try:
            image_path = find_image(raw_dir, image_id)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(f"Gagal membaca {image_path}")
            height, width = int(image.shape[0]), int(image.shape[1])
            expected = (int(group[0]["Height"]), int(group[0]["Width"]))
            if (height, width) != expected:
                raise ValueError(
                    f"{image_id}: ukuran image {(height, width)} != RLE {expected}"
                )

            mask_path = mask_dir / f"{image_id}.png"
            if force or not mask_path.exists():
                mask = decode_group(image_id, group)
                if not cv2.imwrite(str(mask_path), mask):
                    raise OSError(f"Gagal menyimpan mask {mask_path}")
            else:
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != (height, width):
                    raise ValueError(f"{image_id}: mask lama tidak valid")

            manifest_rows.append(
                {
                    "image_id": image_id,
                    "image_path": relative_path(image_path),
                    "mask_path": relative_path(mask_path),
                    "split": spec["split"],
                    "height": height,
                    "width": width,
                    "usage": usage,
                    "num_instances": len(group),
                    "foreground_ratio": round(float((mask > 0).mean()), 6),
                }
            )
        except Exception as error:  # noqa: BLE001
            errors.append(f"{image_id}: {error}")

    manifest_path = splits_dir / f"{spec['split']}.csv"
    if manifest_rows:
        write_manifest(
            manifest_path,
            manifest_rows,
            fieldnames=[
                "image_id",
                "image_path",
                "mask_path",
                "split",
                "height",
                "width",
                "usage",
                "num_instances",
                "foreground_ratio",
            ],
        )

    report = {
        "stage": name,
        "solution": str(solution_path),
        "raw_dir": str(raw_dir),
        "num_solution_images": len(grouped),
        "num_scored": len(manifest_rows),
        "num_ignored": ignored,
        "manifest": str(manifest_path),
        "errors": errors,
    }
    return manifest_rows, report


def write_predict_manifest(
    name: str,
    config: dict[str, Any],
    split_name: str,
) -> tuple[Path, int]:
    spec = SOLUTIONS[name]
    paths = config["paths"]
    raw_dir = resolve_path(paths[spec["raw_key"]])
    splits_dir = resolve_path(paths["splits_dir"])

    if not raw_dir.is_dir():
        print(
            f"Peringatan {name}: folder {raw_dir} tidak ada; "
            "manifest prediksi dilewati.",
            file=sys.stderr,
        )
        return splits_dir / spec["predict_manifest"], 0

    rows: list[dict] = []
    errors: list[str] = []
    for folder in sorted(raw_dir.iterdir()):
        if not folder.is_dir():
            continue
        try:
            image_path = find_image(raw_dir, folder.name)
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(str(image_path))
            rows.append(
                {
                    "image_id": folder.name,
                    "image_path": relative_path(image_path),
                    "split": split_name,
                    "height": int(image.shape[0]),
                    "width": int(image.shape[1]),
                }
            )
        except Exception as error:  # noqa: BLE001
            errors.append(f"{folder.name}: {error}")

    output_path = splits_dir / spec["predict_manifest"]
    if rows:
        write_manifest(
            output_path,
            rows,
            fieldnames=["image_id", "image_path", "split", "height", "width"],
        )
    if errors:
        print(f"Peringatan {name}: {len(errors)} image gagal dibaca", file=sys.stderr)
    return output_path, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--stage", choices=["all", "stage1", "stage2"], default="all")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    stages = ["stage1", "stage2"] if args.stage == "all" else [args.stage]
    reports: dict[str, Any] = {}

    for name in stages:
        _, report = process_stage(
            name,
            config,
            download=args.download,
            force=args.force,
            limit=args.limit,
        )
        reports[name] = report
        if report.get("skipped"):
            predict_path = (
                resolve_path(config["paths"]["splits_dir"])
                / SOLUTIONS[name]["predict_manifest"]
            )
            predict_count = 0
        else:
            predict_path, predict_count = write_predict_manifest(
                name,
                config,
                split_name=f"predict_{name}",
            )
        report["predict_manifest"] = str(predict_path)
        report["predict_images"] = predict_count
        print(
            f"{name}: scored={report['num_scored']} | "
            f"ignored={report['num_ignored']} | "
            f"predict_all={predict_count}"
        )
        if report["errors"]:
            print(f"  error: {len(report['errors'])}", file=sys.stderr)
            for message in report["errors"][:5]:
                print(f"    - {message}", file=sys.stderr)

    processed_dir = resolve_path(config["paths"]["processed_dir"])
    save_json(reports, processed_dir / "official_test" / "report.json")
    errors = sum(len(report.get("errors", [])) for report in reports.values())
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
