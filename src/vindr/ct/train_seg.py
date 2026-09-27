"""Training loop for hemorrhage segmentation (Phase 3, localization).

Mirrors :mod:`vindr.ct.train` (YAML config, run_name subdir, best.pt by
validation Dice, mixed precision) but optimises the ROADMAP localization
metric — Dice >= 0.7 — instead of study-level AUROC.

Usage::

    uv run vindr-ct-seg-train --config configs/train_head_ct_seg.yaml
    uv run vindr-ct-seg-train --config configs/train_head_ct_seg.yaml --smoke
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from vindr.ct.dataset import HeadCTSegDataset, collate_seg, load_masked_studies
from vindr.ct.segmentation import build_seg_model, dice_score, iou_score, seg_loss

log = logging.getLogger("vindr.ct.train_seg")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def split_pairs(
    pairs: list, val_fraction: float, seed: int = 42
) -> tuple[list, list]:
    """Split studies (not slices) so a study never straddles train/val."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pairs))
    n_val = round(len(pairs) * val_fraction)
    n_val = min(max(n_val, 1 if pairs else 0), max(len(pairs) - 1, 0))
    val = [pairs[i] for i in order[:n_val]]
    train = [pairs[i] for i in order[n_val:]]
    return train, val


def evaluate(model, loader, device, threshold: float = 0.5) -> dict:
    """Mean Dice / IoU over validation slices, plus the positive-slice count."""
    model.eval()
    dices, ious, pos = [], [], 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            probs = model(x).sigmoid()
            dices.append(dice_score(probs, y, threshold))
            ious.append(iou_score(probs, y, threshold))
            pos += int((y.sum(dim=(1, 2, 3)) > 0).sum())
    return {
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
        "positive_slices": pos,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Head CT hemorrhage segmentation training (Phase 3)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None, help="override data.root (one subfolder per study)")
    ap.add_argument("--out-dir", default=None, help="override logging.output_dir")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a tiny subset, CPU")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    log_cfg = cfg.get("logging", {})

    root = Path(args.data_dir or data_cfg.get("root", "./data/head_ct_seg"))
    run_name = log_cfg.get("run_name", "head_ct_seg_v1")
    out_dir = Path(args.out_dir or log_cfg.get("output_dir", "./runs")) / run_name
    epochs = 2 if args.smoke else (args.epochs or int(train_cfg.get("epochs", 30)))

    if not root.exists():
        raise SystemExit(
            f"data root not found: {root}. Segmentation needs pixel masks; "
            "see ROADMAP §Phase 3 (the RSNA slice labels alone are not enough)."
        )

    pairs = load_masked_studies(root, data_cfg.get("rescale_csv"))
    if args.smoke:
        pairs = pairs[:4]
    if not pairs:
        raise SystemExit(f"no masked studies under {root}")
    tr_pairs, va_pairs = split_pairs(pairs, float(data_cfg.get("val_fraction", 0.15)))
    log.info("studies: %d train / %d val (loaded %d)", len(tr_pairs), len(va_pairs), len(pairs))

    shape = tuple(data_cfg.get("target_shape", (32, 224, 224)))
    context = int(data_cfg.get("context", 1))
    try:
        train_ds = HeadCTSegDataset(tr_pairs, context=context, target_shape=shape, train=True, augment=True)
        val_pairs_ = va_pairs or tr_pairs[:1]  # a 1-study dataset still needs a val step
        val_ds = HeadCTSegDataset(val_pairs_, context=context, target_shape=shape, train=False)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    log.info(
        "slices: %d train / %d val | %d mask class(es)",
        len(train_ds), len(val_ds), train_ds.n_classes,
    )

    bs = int(train_cfg.get("batch_size", 8))
    nw = int(data_cfg.get("num_workers", 2))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=nw,
        collate_fn=collate_seg, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_seg,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = train_ds.n_classes
    model = build_seg_model(
        in_channels=2 * context + 1,
        num_classes=num_classes,
        width=int(args.width or model_cfg.get("width", 32)),
        depth=int(args.depth or model_cfg.get("depth", 4)),
    ).to(device)
    log.info(
        "model U-Net %d classes on %s (%.1fM params)", num_classes, device.type,
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)),
                            weight_decay=float(train_cfg.get("weight_decay", 1e-5)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    dice_w = float(train_cfg.get("dice_weight", 1.0))
    bce_w = float(train_cfg.get("bce_weight", 1.0))
    thr = float(train_cfg.get("threshold", 0.5))

    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(x)
                loss = seg_loss(logits.float(), y, bce_weight=bce_w, dice_weight=dice_w)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss) * x.shape[0]
            n += x.shape[0]
        sched.step()
        metrics = evaluate(model, val_loader, device, threshold=thr)
        log.info(
            "epoch %d/%d loss=%.4f val_dice=%.4f val_iou=%.4f (%.0fs)",
            epoch, epochs, total / max(n, 1), metrics["dice"], metrics["iou"], time.time() - t0,
        )
        if metrics["dice"] > best:
            best = metrics["dice"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": "seg_unet",
                    "in_channels": 2 * context + 1,
                    "num_classes": num_classes,
                    "width": int(args.width or model_cfg.get("width", 32)),
                    "depth": int(args.depth or model_cfg.get("depth", 4)),
                    "context": context,
                    "target_shape": list(shape),
                    "threshold": thr,
                    "epoch": epoch,
                    "dice": best,
                },
                out_dir / "best.pt",
            )
    log.info("done. best val Dice=%.4f -> %s", best, out_dir / "best.pt")


if __name__ == "__main__":
    main()
