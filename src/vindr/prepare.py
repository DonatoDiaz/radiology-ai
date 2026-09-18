"""Data preparation for VinDr-CXR: load labels, make splits, smoke-test one DICOM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import albumentations as A
import numpy as np
import torch

from vindr.data import VinDrDataset, read_image
from vindr.labels import ALL_LABELS, load_meta_csv, load_train_csv, make_splits

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vindr.prepare")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare VinDr-CXR data")
    ap.add_argument("--data-dir", default="./data/vindr", help="dir with train/, train.csv")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="load one image to verify pipeline")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    df = load_train_csv(data_dir / "train.csv")
    meta = load_meta_csv(data_dir / "train_meta.csv")
    df = make_splits(df, val_fraction=args.val_fraction, seed=args.seed)

    n_train = int((df["split"] == "train").sum())
    n_val = int((df["split"] == "val").sum())
    log.info("train images: %d  val images: %d  classes: %d", n_train, n_val, len(ALL_LABELS))
    log.info("modality counts: %s", meta["modality"].value_counts().to_dict())

    # Class frequency (positive case counts) for sanity checks / class imbalance.
    freq = df[ALL_LABELS].sum(axis=0).sort_values()
    log.info("rarest classes:\n%s", freq.head(5).to_string())
    log.info("most common classes:\n%s", freq.tail(5).to_string())

    if args.smoke:
        transform = A.Compose([A.Resize(224, 224)])
        ds = VinDrDataset(df.head(8), data_dir / "train", ALL_LABELS, transforms=transform)
        arr, target = ds[0]
        log.info("smoke ok: image tensor %s, target %s", tuple(arr.shape), tuple(np.where(target.numpy())))

        raw = read_image(ds._image_path(df.iloc[0]["image_id"]))
        log.info("raw image dtype=%s shape=%s min=%d max=%d", raw.dtype, raw.shape, raw.min(), raw.max())


if __name__ == "__main__":
    main()