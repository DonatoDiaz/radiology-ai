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
import yaml
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from vindr.data import VinDrDataset
from vindr.labels import LUNG_LABELS, load_train_csv, make_splits, restrict_to_lung
from vindr.metrics import compute_metrics
from vindr.model import build_model, count_parameters_mb
from vindr.plots import plot_roc_curves, save_confusion_matrix

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
def evaluate(model, loader, device, labels) -> tuple[float, dict[str, float], np.ndarray, np.ndarray]:
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
    m = compute_metrics(y_true, y_pred, labels)
    return total_loss / len(y_true), m["macro"], y_true, y_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="optional YAML config (configs/train.yaml); CLI flags override it")
    ap.add_argument("--data-dir", default="./data/vindr")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--labels", choices=["lung", "all"], default="lung",
                    help="which label set to train on; default: lung-focused subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--acc-steps", type=int, default=2)
    ap.add_argument("--no-mixed", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--out-dir", default="./runs")
    ap.add_argument("--run-name", default=None)
    args, _ = ap.parse_known_args()

    cfg: dict = {}
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    d, m, t = cfg.get("data", {}), cfg.get("model", {}), cfg.get("train", {})
    if not args.images_dir:
        args.images_dir = d.get("images_dir")
    args.data_dir = d.get("data_dir", args.data_dir)
    args.backbone = m.get("backbone", args.backbone)
    args.image_size = d.get("image_size", args.image_size)
    args.epochs = t.get("epochs", args.epochs)
    args.batch_size = d.get("train_batch_size", args.batch_size)
    args.lr = t.get("lr", args.lr)
    args.weight_decay = t.get("weight_decay", args.weight_decay)
    args.val_fraction = d.get("val_fraction", args.val_fraction)
    args.seed = t.get("seed", args.seed)
    args.acc_steps = t.get("accumulation_steps", args.acc_steps)
    args.no_mixed = args.no_mixed or not t.get("mixed_precision", True)
    args.num_workers = d.get("num_workers", args.num_workers)
    args.out_dir = cfg.get("logging", {}).get("output_dir", args.out_dir)
    args.run_name = cfg.get("logging", {}).get("run_name", args.run_name)
    labels_cfg = m.get("labels", "lung")
    if labels_cfg in ("lung", "all"):
        args.labels = labels_cfg
    args.image_size = int(args.image_size)
    args.epochs = int(args.epochs)
    args.batch_size = int(args.batch_size)
    args.lr = float(args.lr)
    args.weight_decay = float(args.weight_decay)
    args.val_fraction = float(args.val_fraction)
    args.seed = int(args.seed)
    args.acc_steps = int(args.acc_steps)
    args.num_workers = int(args.num_workers)
    args.eval_every = int(args.eval_every)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("device: %s", device)

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir) if args.images_dir else data_dir / "train"
    df = load_train_csv(data_dir / "train.csv")
    # Use only labels actually present in the annotations, preferring the
    # lung-focused subset when those columns exist.
    available = [c for c in LUNG_LABELS if c in df.columns]
    if args.labels == "all" or not available:
        available = [c for c in df.columns if c != "image_id"]
    label_cols = available
    df = df[["image_id", *label_cols]].copy()
    df = df[df[label_cols].sum(axis=1) >= 1].reset_index(drop=True)
    df = make_splits(df, val_fraction=args.val_fraction, seed=args.seed)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    train_ds = VinDrDataset(train_df, images_dir, label_cols, build_transforms(args.image_size, True))
    val_ds = VinDrDataset(val_df, images_dir, label_cols, build_transforms(args.image_size, False))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    log.info("train=%d val=%d", len(train_ds), len(val_ds))

    model = build_model(backbone=args.backbone, num_classes=len(label_cols))
    model = model.to(device)
    log.info("params: %.2fM | labels: %s", count_parameters_mb(model), args.labels)

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
            val_loss, macro, y_true, y_pred = evaluate(model, val_loader, device, label_cols)
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
                        "labels": label_cols,
                        "backbone": args.backbone,
                    },
                    run_dir / "best.pt",
                )
                plot_roc_curves(y_true, y_pred, label_cols, run_dir / "roc_curves.png", top_k=0)
                save_confusion_matrix(y_true, y_pred, label_cols, run_dir / "auroc_top.png")
                log.info("saved new best: AUROC=%.4f (+ plots)", best_auroc)
            if not new_best and best_auroc > 0:
                pass  # future: early stopping callback
    log.info("done. best AUROC=%.4f at %s", best_auroc, run_dir / "best.pt")


if __name__ == "__main__":
    main()