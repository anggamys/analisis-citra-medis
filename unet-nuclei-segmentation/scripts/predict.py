"""Prediksi mask pada himpunan citra tanpa ground truth."""

from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.data.augmentation import build_eval_transforms
from src.data.dataloader import create_dataloader
from src.data.dataset import NucleiDataset
from src.data.preprocessing import load_image_rgb, resized_size
from src.models.unet import build_unet
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config, resolve_path
from src.utils.io import save_json
from src.utils.visualization import save_binary_mask, save_overlay_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default="data/splits/predict_stage2_all.csv")
    parser.add_argument("--split-name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    experiment_dir = resolve_path(paths["experiments_dir"]) / config["project"]["experiment"]
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint
        else experiment_dir / "checkpoints" / "best.pt"
    )
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint tidak ditemukan: {checkpoint_path}")

    manifest_path = resolve_path(args.manifest)
    if not manifest_path.is_file():
        raise SystemExit(
            f"Manifest tidak ditemukan: {manifest_path}\n"
            f"Jalankan: python scripts/prepare_test.py --config {args.config}"
        )

    device = choose_device(args.device)

    # Gunakan arsitektur dan preprocessing dari checkpoint agar cocok
    # walaupun config default diubah setelah training.
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
    image_size = int(data_cfg["image_size"])
    mean = tuple(float(value) for value in data_cfg["normalize_mean"])
    std = tuple(float(value) for value in data_cfg["normalize_std"])
    transforms = build_eval_transforms(
        image_size,
        mean=mean,
        std=std,
    )

    split_name = args.split_name or manifest_path.stem
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else experiment_dir / "predictions" / split_name
    )
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    dataset = NucleiDataset(manifest_path, transform=transforms, has_masks=False)
    loader = create_dataloader(
        dataset,
        batch_size=int(train_config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)) and device.type == "cuda",
    )

    saved = 0
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Prediksi {split_name}"):
            images = batch["image"].to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast(enabled=True):
                    logits = model(images)
            else:
                logits = model(images)
            probabilities = torch.sigmoid(logits).float().cpu().numpy()

            for index, image_id in enumerate(batch["image_id"]):
                height, width = [int(value) for value in batch["original_size"][index]]
                # Kanvas model berbentuk persegi (hasil resize + pad), jadi
                # potong area citra yang valid sebelum kembali ke ukuran asli.
                content_height, content_width = resized_size(
                    height,
                    width,
                    image_size,
                )
                probability = probabilities[index, 0][
                    :content_height, :content_width
                ]
                probability = np.ascontiguousarray(probability)
                probability = cv2.resize(
                    probability,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                )
                mask = (probability >= threshold).astype(np.uint8) * 255
                save_binary_mask(mask, mask_dir / f"{image_id}.png")

                if args.save_probabilities:
                    probability_dir = output_dir / "probabilities"
                    probability_dir.mkdir(parents=True, exist_ok=True)
                    probability_uint8 = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
                    cv2.imwrite(
                        str(probability_dir / f"{image_id}.png"),
                        probability_uint8,
                    )

                if args.save_overlays:
                    image_rgb = load_image_rgb(batch["image_path"][index])
                    save_overlay_image(
                        image_rgb,
                        mask,
                        output_dir / "overlays" / f"{image_id}.png",
                    )

                saved += 1
                if args.max_samples is not None and saved >= args.max_samples:
                    break
            if args.max_samples is not None and saved >= args.max_samples:
                break

    save_json(
        {
            "manifest": str(manifest_path),
            "split": split_name,
            "threshold": float(threshold),
            "num_predictions": saved,
            "output_dir": str(output_dir),
        },
        output_dir / "prediction_report.json",
    )
    print(f"Selesai: {saved} mask -> {mask_dir}")


if __name__ == "__main__":
    main()
