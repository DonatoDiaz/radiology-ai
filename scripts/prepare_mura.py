#!/usr/bin/env python3
"""Prepare a MURA release for study-level fracture screening.

MURA's manifests carry the patient, series and view but *not* whether a study
is abnormal — that is in the folder name. This script joins the two, collapses
images into studies, and writes the split **per patient**, because a patient
contributes several studies and a per-study split would put the same arm in
train and val.

    uv run python scripts/prepare_mura.py \\
        --csv /data/MURA-v1.1/train_image_paths.csv \\
              /data/MURA-v1.1/valid_image_paths.csv \\
        --root /data/MURA-v1.1 --out ./data/mura --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from vindr.mura import (
    collect_studies,
    make_patient_splits,
    parse_mura_csv,
    patients_in_both_splits,
    split_summary,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare MURA for study-level fracture screening")
    ap.add_argument("--csv", type=Path, nargs="+", required=True,
                    help="MURA manifest CSVs, e.g. train_image_paths.csv valid_image_paths.csv")
    ap.add_argument("--root", type=Path, required=True, help="dir the manifest paths are relative to")
    ap.add_argument("--out", type=Path, default=None, help="write studies CSV here (default: stdout)")
    ap.add_argument("--val-fraction", type=float, default=0.15,
                    help="share of *patients* held out, not studies (default: 0.15)")
    ap.add_argument("--seed", type=int, default=42, help="split seed (default: 42)")
    ap.add_argument("--body-parts", default=None,
                    help="comma-separated XR_* prefixes to keep (default: all)")
    ap.add_argument("--max-studies", type=int, default=0, help="limit for a dry run (0 = all)")
    ap.add_argument("--no-count-images", action="store_true",
                    help="trust the manifest counts instead of walking the image tree")
    ap.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = ap.parse_args()

    frames = [parse_mura_csv(p) for p in args.csv]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="mura_id")

    if args.body_parts:
        keep = {s.strip() for s in args.body_parts.split(",") if s.strip()}
        df = df[df["body_part"].isin(keep)]

    studies = collect_studies(df, args.root, require_images=not args.no_count_images)
    if (studies["n_images"] == 0).any():
        missing = studies.loc[studies["n_images"] == 0, "study_folder"].head(3).tolist()
        print(f"[warn] {int((studies['n_images'] == 0).sum())} study folder(s) have no images "
              f"under {args.root}, e.g. {missing}", file=sys.stderr)
    if args.max_studies:
        studies = studies.sort_values("study_folder").head(args.max_studies)

    studies = make_patient_splits(studies, val_fraction=args.val_fraction, seed=args.seed)
    leaked = patients_in_both_splits(studies)
    if leaked:
        raise SystemExit(f"patient leakage across the split: {leaked[:5]}")

    summary = split_summary(studies)
    print(f"studies: {summary['n_studies']} from {summary['n_patients']} patients "
          f"(positive rate {summary['positive_rate']})")
    for split in ("train", "val"):
        print(f"  {split}: {summary[f'{split}_studies']} studies / "
              f"{summary[f'{split}_patients']} patients / {summary[f'{split}_abnormal']} abnormal")
    print("\nper body part (train/val studies):")
    table = studies.pivot_table(index="body_part", columns="split", values="study_folder",
                                aggfunc="count", fill_value=0)
    print(table.to_string())
    views = studies.groupby("split")["n_images"].agg(["mean", "min", "max"]).round(2)
    print("\nprojections per study:")
    print(views.to_string())
    print(f"\npatients in both splits: {len(leaked)}")

    if args.dry_run:
        print("\ndry run, nothing written")
        return
    if args.out is None:
        print("\nno --out given, nothing written")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["study_folder", "patient_id", "body_part", "laterality", "abnormal",
            "n_views", "n_images", "split"]
    studies[cols].to_csv(args.out, index=False)
    print(f"\nwrote {len(studies)} studies -> {args.out}")


if __name__ == "__main__":
    main()