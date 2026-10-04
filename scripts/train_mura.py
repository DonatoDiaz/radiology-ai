#!/usr/bin/env python3
"""Train a study-level MURA fracture classifier.

The split must already be per patient — `scripts/prepare_mura.py` does that and
records the seed it used, because a per-study split would put the same arm in
train and val and the AUROC below would mean nothing.

    uv run python scripts/train_mura.py \
        --csv ./data/mura/studies.csv --root /data/MURA-v1.1 \
        --epochs 20 --image-size 224 --max-views 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from vindr.mura import (
    MuraStudyDataset,
    build_mura_model,
    collate_studies,
    patients_in_both_splits,
)
from vindr.train import build_transforms

log = logging.getLogger("train_mura")
STUDY_LABELS = ["abnormal"]
NOT_CLAIM_NOTE = "no MURA study has been trained; unvalidated, not a diagnosis"


def study_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Per-study probabilities and labels."""
    model.eval()
    probs, targets = [], []
    with torch.no_grad():
        for views, label, mask in loader:
            views, mask = views.to(device), mask.to(device)
            logit = model(views, mask)
            probs.append(logit.sigmoid().cpu().numpy())
            targets.append(label.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a study-level MURA classifier")
    ap.add_argument("--csv", type=Path, required=True, help="studies CSV from prepare_mura.py")
    ap.add_argument("--root", type=Path, required=True, help="MURA image root")
    ap.add_argument("--backbone", default="tf_efficientnet_b0")
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--max-views", type=int, default=4)
    ap.add_argument("--pooling", choices=["max", "mean"], default="max",
                    help="how views combine into a study score (default: max, per MURA)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("./runs"))
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(args.csv)
    leaked = patients_in_both_splits(df)
    if leaked:
        raise SystemExit(f"patient leakage in {args.csv}: {leaked[:5]}. Rebuild the split with prepare_mura.py.")
    log.info("%d studies, %d patients", len(df), df["patient_id"].nunique())

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise SystemExit("need both train and val studies; check --val-fraction in prepare_mura.py")

    size = args.image_size
    train_ds = MuraStudyDataset(train_df, args.root, build_transforms(size, True),
                                max_views=args.max_views)
    val_ds = MuraStudyDataset(val_df, args.root, build_transforms(size, False),
                              max_views=args.max_views)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=collate_studies, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_studies)

    model = build_mura_model(args.backbone, pretrained=not args.no_pretrained,
                             pooling=args.pooling).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_name = args.run_name or f"mura_{args.backbone}_{args.pooling}_seed{args.seed}"
    out_dir = args.out_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str))

    best = {"auroc": -1.0}
    for epoch in range(args.epochs):
        model.train()
        running, n = 0.0, 0
        for i, (views, label, mask) in enumerate(train_loader):
            views, label, mask = views.to(device), label.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(views, mask), label)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
            n += 1
            if i % 20 == 0:
                log.info("epoch %d step %d/%d loss=%.4f", epoch, i, len(train_loader), loss.item())
        scheduler.step()

        probs, targets = study_scores(model, val_loader, device)
        if len(np.unique(targets)) < 2:
            log.warning("val split has one class, AUROC is undefined")
            auroc = float("nan")
            ap = float("nan")
        else:
            auroc = float(roc_auc_score(targets, probs))
            ap = float(average_precision_score(targets, probs))
        log.info("epoch %d train_loss=%.4f val_auroc=%.4f val_ap=%.4f",
                 epoch, running / max(n, 1), auroc, ap)
        if not np.isnan(auroc) and auroc > best["auroc"]:
            best = {"auroc": auroc, "ap": ap, "epoch": epoch}
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "pooling": args.pooling, "auroc": auroc}, out_dir / "best.pt")

    summary = {"best_val_auroc": best["auroc"], "best_val_ap": best.get("ap"),
               "epoch": best.get("epoch"), "n_train": len(train_df), "n_val": len(val_df),
               "n_patients": int(df["patient_id"].nunique()), "pooling": args.pooling,
               "validated": False,
               "note": NOT_CLAIM_NOTE}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    log.info("best val AUROC %.4f -> %s", best["auroc"], out_dir)


if __name__ == "__main__":
    main()