"""Evaluasi checkpoint pada test set resmi stage1/stage2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.data.augmentation import build_eval_transforms
from src.data.dataloader import create_dataloader
from src.data.dataset import NucleiDataset
from src.evaluation.evaluator import (
    collect_predictions,
    evaluate_split,
    save_evaluation,
)
from src.evaluation.metrics import tune_threshold
from src.models.unet import build_unet
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_path
from src.utils.visualization import save_prediction_grid

SPLIT_FILES = {
    "stage1": ("test_stage1.csv", "test_stage1"),
    "stage2": ("test_stage2_scored.csv", "test_stage2_scored"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="stage1")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--tune-on-val", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    experiment_dir = (
        resolve_path(paths["experiments_dir"]) / config["project"]["experiment"]
    )
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint
        else experiment_dir / "checkpoints" / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    device = choose_device(args.device)

    # Arsitektur dan preprocessing harus mengikuti config saat training,
    # bukan config default yang mungkin sudah diubah.
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    train_config = checkpoint.get("config") or config
    use_amp = bool(train_config["training"].get("amp", False)) and device.type == "cuda"
    model = build_unet(train_config).to(device)
    model.load_state_dict(checkpoint["model_state"])

    threshold = args.threshold
    if threshold is None:
        threshold = float(
            checkpoint.get(
                "threshold",
                train_config["training"].get("threshold", 0.5),
            )
        )

    data_cfg = train_config["data"]
    eval_cfg = train_config["evaluation"]
    mean_values = tuple(float(value) for value in data_cfg["normalize_mean"])
    std_values = tuple(float(value) for value in data_cfg["normalize_std"])
    if len(mean_values) != 3 or len(std_values) != 3:
        raise ValueError(
            "normalize_mean dan normalize_std harus masing-masing memiliki "
            "tepat 3 nilai."
        )
    mean: tuple[float, float, float] = (
        mean_values[0],
        mean_values[1],
        mean_values[2],
    )
    std: tuple[float, float, float] = (
        std_values[0],
        std_values[1],
        std_values[2],
    )
    transforms = build_eval_transforms(
        int(data_cfg["image_size"]),
        mean=mean,
        std=std,
    )

    splits_dir = resolve_path(paths["splits_dir"])
    manifest_name, split_name = SPLIT_FILES[args.split]
    test_manifest = splits_dir / manifest_name
    if not test_manifest.is_file():
        raise SystemExit(
            f"Manifest test tidak ditemukan: {test_manifest}\n"
            f"Jalankan: python scripts/prepare_test.py --config {args.config}"
        )

    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"

    trials = None
    if args.tune_on_val:
        val_manifest = splits_dir / "val.csv"
        val_dataset = NucleiDataset(val_manifest, transform=transforms, has_masks=True)
        val_loader = create_dataloader(
            val_dataset,
            batch_size=int(eval_cfg.get("batch_size", 8)),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_predictions = collect_predictions(
            model,
            val_loader,
            device,
            use_amp=use_amp,
            include_images=False,
        )
        threshold, trials = tune_threshold(
            val_predictions,
            eval_cfg.get("threshold_candidates", [0.5]),
        )
        print(f"Threshold terbaik dari validation: {threshold:.2f}")

    test_dataset = NucleiDataset(test_manifest, transform=transforms, has_masks=True)
    test_loader = create_dataloader(
        test_dataset,
        batch_size=int(eval_cfg.get("batch_size", 8)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    rows, summary, predictions = evaluate_split(
        model,
        test_loader,
        device,
        threshold=threshold,
        use_amp=use_amp,
        max_samples=args.max_samples,
    )

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else experiment_dir / "evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    save_evaluation(
        output_dir=output_dir,
        split=split_name,
        rows=rows,
        summary=summary,
        threshold=threshold,
        trials=trials,
    )

    num_visualizations = int(eval_cfg.get("num_visualizations", 8))
    save_prediction_grid(
        predictions,
        output_dir / f"predictions_{split_name}.png",
        threshold=threshold,
        max_samples=num_visualizations,
        mean=mean,
        std=std,
        sort_by_dice=False,
    )
    save_prediction_grid(
        predictions,
        output_dir / f"failure_cases_{split_name}.png",
        threshold=threshold,
        max_samples=int(eval_cfg.get("num_failure_cases", 8)),
        mean=mean,
        std=std,
        sort_by_dice=True,
    )

    print(f"Split   : {split_name} ({len(rows)} image)")
    print(f"Threshold: {threshold:.2f}")
    for key in ("dice", "iou", "precision", "recall"):
        metric = summary[key]
        print(f"{key:>9}: {metric['mean']:.4f} ± {metric['std']:.4f}")
    print(f"Output  : {output_dir}")


if __name__ == "__main__":
    main()
