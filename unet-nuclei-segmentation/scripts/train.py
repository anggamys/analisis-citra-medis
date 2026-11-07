"""Training U-Net nuclei segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.data.augmentation import build_eval_transforms, build_train_transforms
from src.data.dataloader import create_dataloader
from src.data.dataset import NucleiDataset
from src.data.manifest import read_manifest
from src.models.unet import build_unet
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.checkpoint import environment_info, load_checkpoint
from src.utils.config import load_config, resolve_path, save_config
from src.utils.io import save_json
from src.utils.seed import seed_everything
from src.utils.visualization import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None, help="auto | cpu | cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def choose_device(requested: str | None) -> torch.device:
    if requested is None or requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.experiment is not None:
        config["project"]["experiment"] = args.experiment
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr
    if args.no_amp:
        config["training"]["amp"] = False
    if args.smoke_test:
        config["training"]["epochs"] = 1
        config["training"]["batch_size"] = 2
        config["data"]["num_workers"] = 0
        config["evaluation"]["batch_size"] = 2
        config["training"]["early_stopping_patience"] = 1
        args.max_samples = args.max_samples or 8

    seed = int(args.seed if args.seed is not None else config["project"]["seed"])
    seed_everything(
        seed, deterministic=bool(config["project"].get("deterministic", False))
    )

    paths = config["paths"]
    splits_dir = resolve_path(paths["splits_dir"])
    experiments_dir = resolve_path(paths["experiments_dir"])
    experiment_dir = experiments_dir / config["project"]["experiment"]
    experiment_dir.mkdir(parents=True, exist_ok=True)

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    if not train_csv.is_file() or not val_csv.is_file():
        raise SystemExit(
            f"Manifest train/val tidak ditemukan. Jalankan dulu:\n"
            f"  python scripts/prepare_data.py --config {args.config}"
        )

    train_rows = read_manifest(train_csv)
    val_rows = read_manifest(val_csv)
    if args.max_samples is not None:
        train_rows = train_rows[: args.max_samples]
        val_rows = val_rows[: args.max_samples]

    device = choose_device(args.device)
    if device.type != "cuda":
        config["training"]["amp"] = False
        if args.device not in {None, "auto", "cpu"}:
            print("Peringatan: CUDA tidak tersedia, memakai CPU.")
        print("Device: CPU (AMP dinonaktifkan)")

    data_cfg = config["data"]
    mean = (
        float(data_cfg["normalize_mean"][0]),
        float(data_cfg["normalize_mean"][1]),
        float(data_cfg["normalize_mean"][2]),
    )
    std = (
        float(data_cfg["normalize_std"][0]),
        float(data_cfg["normalize_std"][1]),
        float(data_cfg["normalize_std"][2]),
    )
    image_size = int(data_cfg["image_size"])

    train_dataset = NucleiDataset(
        train_rows,
        transform=build_train_transforms(image_size, mean=mean, std=std),
        has_masks=True,
    )
    val_dataset = NucleiDataset(
        val_rows,
        transform=build_eval_transforms(image_size, mean=mean, std=std),
        has_masks=True,
    )

    num_workers = int(data_cfg.get("num_workers", 0))
    pin_memory = bool(data_cfg.get("pin_memory", False)) and device.type == "cuda"
    train_loader = create_dataloader(
        train_dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=int(config["evaluation"]["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    model = build_unet(config).to(device)
    print(f"Model parameter: {model.num_parameters:,}")

    loss_fn = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler, scheduler_name = build_scheduler(optimizer, config)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_name=scheduler_name,
        device=device,
        config=config,
        experiment_dir=experiment_dir,
        threshold=float(config["training"].get("threshold", 0.5)),
    )

    start_epoch = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(
            resolve_path(args.resume),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        trainer.best_score = float(checkpoint.get("best_score", float("-inf")))
        print(f"Resume dari epoch {start_epoch} (best={trainer.best_score:.4f})")

    save_config(config, experiment_dir / "config.yaml")
    save_json(environment_info(), experiment_dir / "environment.json")

    trainer.fit(train_loader, val_loader, start_epoch=start_epoch)
    if trainer.history:
        plot_training_curves(
            trainer.history,
            experiment_dir / "training_curves.png",
        )
    else:
        print("Tidak ada epoch baru; kurva training tidak diperbarui.")

    print(f"Best Dice: {trainer.best_score:.4f}")
    print(f"Checkpoint: {trainer.best_checkpoint}")
    print(f"Log: {experiment_dir / 'logs' / 'train_log.csv'}")


if __name__ == "__main__":
    main()
