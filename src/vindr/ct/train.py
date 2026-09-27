"""Training loop for head CT hemorrhage classification (Phase 3).

    uv run python -m vindr.ct.train --config configs/train_head_ct.yaml
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

from vindr.ct.dataset import (
    HeadCTSliceDataset,
    HeadCTStudyDataset,
    collate_slices,
    collate_studies,
    load_study_folders,
)
from vindr.ct.model import HEMORRHAGE_TYPES, NUM_OUTPUTS, any_label, build_head_ct_model, loss_fn
from vindr.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vindr.ct.train")

# metric/label names for AUROC reporting
METRIC_NAMES = list(HEMORRHAGE_TYPES) + ["any"]


def _load_labels(csv_path: str | Path) -> dict[str, list[float]]:
    """Read a per-study label CSV -> {study_id: [5 types..., any]}.

    Expected columns: ``study_id`` plus either the five type names or an
    ``any`` column. Missing studies are treated as all-negative.
    """
    import csv

    out: dict[str, list[float]] = {}
    if not Path(csv_path).exists():
        log.warning("label file %s not found — treating all studies as negative", csv_path)
        return out
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            keys = {k.lower(): v for k, v in row.items() if k}
            sid = str(keys.get("study_id") or keys.get("studyid") or keys.get("id") or "")
            if not sid:
                continue
            vec = []
            for name in HEMORRHAGE_TYPES:
                val = keys.get(name.replace(" ", "_")) or keys.get(name) or "0"
                vec.append(float(val))
            row_any = keys.get("any") or keys.get("any_hemorrhage")
            vec.append(float(row_any) if row_any is not None else float(sum(vec) > 0))
            out[sid] = vec
    return out


def _attach_labels(studies, labels: dict[str, list[float]]):
    """Match studies to label rows by study/series/patient id, else positional."""
    per_study: list[list[float]] = []
    for st in studies:
        key = st.study_id or st.series_id or st.patient_id
        vec = labels.get(key)
        if vec is None:
            for cand in (st.patient_id, st.series_id):
                if cand and cand in labels:
                    vec = labels[cand]
                    break
        per_study.append(vec if vec is not None else [0.0] * NUM_OUTPUTS)
    return per_study


def _split(studies, labels, val_fraction: float, seed: int = 42):
    idx = np.arange(len(studies))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * val_fraction)) if len(idx) > 1 else 0
    val_i, train_i = idx[:n_val], idx[n_val:]
    pick = lambda ii: ([studies[i] for i in ii], [labels[i] for i in ii])
    return pick(train_i), pick(val_i)


def evaluate(model, loader, device, study_level: bool) -> dict:
    """Validation metrics, always at *study* level.

    Slice-level datasets repeat the study label on every slice, so scoring raw
    slices would weight every study by its slice count. Slice probabilities are
    therefore max-pooled into their study before computing AUROC / AP.
    """
    model.eval()
    probs, targets, groups = [], [], []
    with torch.no_grad():
        for x, y, g in loader:
            x = x.to(device, non_blocking=True)
            # study loader yields (B, S, C, H, W); slice loader yields (B, C, H, W)
            logits = model(x)
            probs.append(logits.sigmoid().cpu().numpy())
            targets.append(y.numpy())
            groups.append(np.asarray(g))
    if not probs:
        return {}
    p = np.concatenate(probs)
    t = any_label(np.concatenate(targets))
    g = np.concatenate(groups)

    if not study_level:  # one row per study after pooling
        order: dict[int, int] = {}
        rows_p, rows_t = [], []
        for i, key in enumerate(g):
            slot = order.setdefault(int(key), len(rows_p))
            if slot == len(rows_p):
                rows_p.append(p[i])
                rows_t.append(t[i])
            else:
                rows_p[slot] = np.maximum(rows_p[slot], p[i])
        p = np.stack(rows_p) if rows_p else p
        t = np.stack(rows_t) if rows_t else t

    if t.shape[0] < 2 or len(np.unique(t[:, -1])) < 2:
        log.warning("val set has a single class (%d rows) — AUROC is undefined", t.shape[0])
    return compute_metrics(t, p, METRIC_NAMES)


def main() -> None:
    ap = argparse.ArgumentParser(description="Head CT hemorrhage training (Phase 3)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", default=None, help="override data.root (one subfolder per study)")
    ap.add_argument("--labels", default=None, help="override data.labels_csv")
    ap.add_argument("--out-dir", default=None, help="override logging.output_dir")
    ap.add_argument("--kind", choices=("study", "slice"), default=None,
                    help="override model.kind: study (attention-pooled) or slice")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a tiny subset, CPU")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    log_cfg = cfg.get("logging", {})

    root = Path(args.data_dir or data_cfg.get("root", "./data/head_ct"))
    labels_csv = Path(args.labels or data_cfg.get("labels_csv", root / "study_labels.csv"))
    run_name = log_cfg.get("run_name", "head_ct_v1")
    out_dir = Path(args.out_dir or log_cfg.get("output_dir", "./runs")) / run_name
    epochs = args.epochs or int(train_cfg.get("epochs", 12))
    if args.kind:
        model_cfg["kind"] = args.kind
    study_level = model_cfg.get("kind", "study") == "study"

    if not root.exists():
        raise SystemExit(f"data root not found: {root} (see ROADMAP §Phase 3 for the datasets)")

    studies = load_study_folders(root)
    if args.smoke:
        studies = studies[:4]
    if not studies:
        raise SystemExit(f"no studies under {root}")
    labels = _attach_labels(studies, _load_labels(labels_csv))
    (tr_s, tr_l), (va_s, va_l) = _split(studies, labels, float(data_cfg.get("val_fraction", 0.15)))
    log.info("studies: %d train / %d val (loaded %d)", len(tr_s), len(va_s), len(studies))

    shape = tuple(data_cfg.get("target_shape", (32, 224, 224)))
    slices_per = int(data_cfg.get("slices_per_study", 24))
    if study_level:
        mk = lambda st, lb, tr: HeadCTStudyDataset(st, lb, slices_per, target_shape=shape, train=tr)
        coll = collate_studies
    else:
        mk = lambda st, lb, tr: HeadCTSliceDataset(st, lb, target_shape=shape, train=tr, augment=tr)
        coll = collate_slices

    train_ds, val_ds = mk(tr_s, tr_l, True), mk(va_s, va_l, False)
    bs = int(train_cfg.get("batch_size", 2 if study_level else 8))
    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=True, num_workers=int(data_cfg.get("num_workers", 2)),
        collate_fn=coll, pin_memory=True, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, num_workers=int(data_cfg.get("num_workers", 2)), collate_fn=coll
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_head_ct_model(
        kind=model_cfg.get("kind", "study"),
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_outputs=NUM_OUTPUTS,
        width=int(model_cfg.get("width", 32)),
    ).to(device)
    log.info("model %s on %s (%.1fM params)", model_cfg.get("kind", "study"), device.type,
             sum(p.numel() for p in model.parameters()) / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 3e-4)),
                            weight_decay=float(train_cfg.get("weight_decay", 1e-5)))
    epochs = 2 if args.smoke else epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda")

    out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        total, n = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y = any_label(y)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss) * x.shape[0]
            n += x.shape[0]
        sched.step()
        metrics = evaluate(model, val_loader, device, study_level)
        auroc = float(metrics.get("macro", {}).get("auroc", 0.0))
        log.info("epoch %d/%d loss=%.4f val_auroc=%.4f (%.0fs)", epoch, epochs, total / max(n, 1), auroc, time.time() - t0)
        if auroc > best:
            best = auroc
            ckpt = {
                "model_state": model.state_dict(),
                "model": model_cfg.get("kind", "study"),
                "labels": list(HEMORRHAGE_TYPES) + ["any"],
                "in_channels": int(model_cfg.get("in_channels", 3)),
                "width": int(model_cfg.get("width", 32)),
                "target_shape": list(shape),
                "epoch": epoch,
                "auroc": auroc,
            }
            torch.save(ckpt, out_dir / "best.pt")
    log.info("done. best val AUROC=%.4f -> %s", best, out_dir / "best.pt")


if __name__ == "__main__":
    main()
