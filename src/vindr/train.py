"""Training loop for VinDr-CXR multi-label classification."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from vindr.data import VinDrDataset
from vindr.labels import ALL_LABELS, load_train_csv, make_splits
from vindr.metrics import compute_metrics
from vindr.model import build_model, count_parameters_mb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vindr.train")


def build_transforms(image_size: int, train: bool):
    if train:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                A.RandomBrightnessContrast(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5),
                A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
                ToTensorV2(),
            ]
        )
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    acc_steps: int,
    log_every: int,
    epoch: int,
) -> float:
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                loss = criterion(model(x), y) / acc_steps
            scaler.scale(loss).backward()
            if (i + 1) % acc_steps == 0 or (i + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss = criterion(model(x), y) / acc_steps
            loss.backward()
            if (i + 1) % acc_steps == 0 or (i + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * acc_steps
        n_batches += 1
        if i % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            log.info(
                "epoch %d step %d/%d loss=%.4f lr=%.2e %.1f it/s",
                epoch, i, len(loader), total_loss / n_batches, lr,
                (i + 1) / (time.time() - t0),
            )
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, dict[str, float]]:
    model.eval()
    preds, targets = [], []
    total_loss, criterion = 0.0, nn.BCEWithLogitsLoss()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        preds.append(out.sigmoid().cpu().numpy())
        targets.append(y.cpu().numpy())
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(preds)
    m = compute_metrics(y_true, y_pred, ALL_LABELS)
    return total_loss / len(y_true), m["macro"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data/vindr")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acc-steps", type=int, default=2)
    ap.add_argument("--no-mixed", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--out-dir", default="./runs")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir) if args.images_dir else data_dir / "train"
    df = load_train_csv(data_dir / "train.csv")
    df = make_splits(df, val_fraction=args.val_fraction, seed=args.seed)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    train_ds = VinDrDataset(train_df, images_dir, ALL_LABELS, build_transforms(args.image_size, True))
    val_ds = VinDrDataset(val_df, images_dir, ALL_LABELS, build_transforms(args.image_size, False))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    log.info("train=%d val=%d", len(train_ds), len(val_ds))

    model = build_model(backbone=args.backbone, num_classes=len(ALL_LABELS))
    model = model.to(device)
    log.info("params: %.2fM", count_parameters_mb(model))

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = not args.no_mixed and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_dir = Path(args.out_dir) / (args.run_name or f"{args.backbone}_{args.image_size}")
    run_dir.mkdir(parents=True, exist_ok=True)

    best_auroc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            device, args.acc_steps, 50, epoch,
        )
        scheduler.step()
        if epoch % args.eval_every == 0:
            val_loss, macro = evaluate(model, val_loader, device)
            log.info(
                "=== epoch %d/%d train_loss=%.4f val_loss=%.4f AUROC=%.4f AP=%.4f (%.0fs) ===",
                epoch, args.epochs, loss, val_loss, macro["auroc"], macro["ap"],
                time.time() - t0,
            )
            new_best = False
            if macro["auroc"] > best_auroc:
                best_auroc = macro["auroc"]
                new_best = True
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "metrics": macro,
                        "labels": ALL_LABELS,
                        "backbone": args.backbone,
                    },
                    run_dir / "best.pt",
                )
                log.info("saved new best: AUROC=%.4f", best_auroc)
            if not new_best and best_auroc > 0:
                pass  # future: early stopping callback
    log.info("done. best AUROC=%.4f at %s", best_auroc, run_dir / "best.pt")


if __name__ == "__main__":
    main()