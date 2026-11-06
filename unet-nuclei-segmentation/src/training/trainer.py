"""Training loop dan validasi."""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import compute_metrics, summarize_metrics
from src.utils.checkpoint import save_checkpoint
from src.utils.io import save_json
from src.utils.logger import CSVLogger


class Trainer:
    LOG_FIELDS: ClassVar[list[str]] = [
        "epoch",
        "lr",
        "train_loss",
        "val_loss",
        "val_dice",
        "val_iou",
        "val_precision",
        "val_recall",
        "epoch_seconds",
        "best_dice",
    ]

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None,
        scheduler_name: str,
        device: torch.device,
        config: dict[str, Any],
        experiment_dir: str | Path,
        threshold: float = 0.5,
    ) -> None:
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_name = scheduler_name
        self.device = device
        self.config = config
        self.experiment_dir = Path(experiment_dir)
        self.threshold = float(threshold)

        training_cfg = config["training"]
        self.epochs = int(training_cfg.get("epochs", 50))
        self.grad_clip = float(training_cfg.get("grad_clip", 0.0))
        self.min_delta = float(training_cfg.get("min_delta", 0.0))
        self.patience = int(training_cfg.get("early_stopping_patience", 10))

        self.use_amp = bool(training_cfg.get("amp", False)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.best_checkpoint = self.checkpoint_dir / "best.pt"
        self.last_checkpoint = self.checkpoint_dir / "last.pt"
        self.logger = CSVLogger(
            self.experiment_dir / "logs" / "train_log.csv",
            fieldnames=self.LOG_FIELDS,
        )

        self.best_score = float("-inf")
        self.epochs_without_improvement = 0
        self.history: list[dict[str, Any]] = []

    def _autocast(self):
        if self.use_amp:
            return torch.cuda.amp.autocast(enabled=True)
        return nullcontext()

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True).float()

            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

            self.scaler.scale(loss).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip,
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        return total_loss / max(total_samples, 1)

    @torch.no_grad()
    def _validate(
        self,
        loader: DataLoader,
    ) -> tuple[float, dict[str, dict[str, float]], list[dict[str, Any]]]:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        per_image: list[dict[str, Any]] = []

        for batch in loader:
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True).float()

            with self._autocast():
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)

            probabilities = torch.sigmoid(logits).float().cpu()
            targets = masks.float().cpu()
            batch_metrics = compute_metrics(
                probabilities,
                targets,
                threshold=self.threshold,
            )

            image_ids = batch["image_id"]
            for index, image_id in enumerate(image_ids):
                row = {
                    "image_id": str(image_id),
                    "threshold": self.threshold,
                }
                row.update(
                    {key: float(values[index]) for key, values in batch_metrics.items()}
                )
                per_image.append(row)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

        summary = summarize_metrics(per_image)
        return total_loss / max(total_samples, 1), summary, per_image

    def _save(
        self,
        path: Path,
        epoch: int,
    ) -> None:
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_score=self.best_score,
            threshold=self.threshold,
            config=self.config,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        start_epoch: int = 0,
    ) -> list[dict[str, Any]]:
        for epoch in range(start_epoch, self.epochs):
            started = time.time()
            train_loss = self._train_epoch(train_loader)
            val_loss, summary, _ = self._validate(val_loader)
            val_dice = summary["dice"]["mean"]

            lr = float(self.optimizer.param_groups[0]["lr"])
            if self.scheduler is not None:
                if self.scheduler_name == "plateau":
                    self.scheduler.step(val_dice)
                else:
                    self.scheduler.step()

            elapsed = time.time() - started
            improved = val_dice > (self.best_score + self.min_delta)
            if improved:
                self.best_score = val_dice
                self.epochs_without_improvement = 0
                self._save(self.best_checkpoint, epoch)
            else:
                self.epochs_without_improvement += 1

            row = {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_dice": val_dice,
                "val_iou": summary["iou"]["mean"],
                "val_precision": summary["precision"]["mean"],
                "val_recall": summary["recall"]["mean"],
                "epoch_seconds": elapsed,
                "best_dice": self.best_score,
            }
            self.history.append(row)
            self.logger.log(row)
            self._save(self.last_checkpoint, epoch)

            print(
                f"epoch {epoch + 1:03d}/{self.epochs} | "
                f"train {train_loss:.4f} | val {val_loss:.4f} | "
                f"dice {val_dice:.4f} | iou {summary['iou']['mean']:.4f} | "
                f"best {self.best_score:.4f} | {elapsed:.1f}s"
            )

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping pada epoch {epoch + 1}")
                break

        save_json(
            {
                "epochs": len(self.history),
                "best_dice": self.best_score,
                "history": self.history,
            },
            self.experiment_dir / "history.json",
        )
        return self.history
