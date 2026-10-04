#!/usr/bin/env python3
"""Convert a GRAZPEDWRI-style release (Pascal VOC XML) into YOLO labels.

    uv run python scripts/prepare_grazpedwri.py \
        --images /data/GRAZPEDWRI-DX/images \
        --out ./data/grazpedwri

If the release ships a study/patient CSV, pass it so the split is per patient —
a per-image split puts the same child's other radiograph in both sides and
inflates mAP50. Without it the script says so rather than pretending.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from vindr.grazpedwri import convert_release


def load_groups(csv_path: Path | None, key: str) -> dict[str, str] | None:
    """image filename -> patient/study id, from the release's own CSV."""
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    image_col = next((cols[c] for c in ("image", "filename", "image_name", "file") if c in cols), None)
    group_col = next((cols[c] for c in ("patient", "patient_id", "study", "study_id") if c in cols), None)
    if image_col is None or group_col is None:
        raise SystemExit(f"{csv_path}: need an image column and a patient/study column, got {list(df.columns)}")
    return dict(zip(df[image_col].astype(str), df[group_col].astype(str), strict=True))


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert GRAZPEDWRI Pascal VOC XML to YOLO labels")
    ap.add_argument("--images", type=Path, required=True, help="dir with radiographs")
    ap.add_argument("--out", type=Path, required=True, help="output dir for labels/ and data.yaml")
    ap.add_argument("--groups-csv", type=Path, default=None,
                    help="release CSV with image + patient/study columns, for a per-patient split")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep-non-fractures", action="store_true",
                    help="also train on 'text'/'normal' objects instead of dropping them")
    ap.add_argument("--dry-run", action="store_true", help="report and stop before writing")
    args = ap.parse_args()

    groups = load_groups(args.groups_csv, "patient") if args.groups_csv else None
    if groups is None:
        print("[warn] no --groups-csv: splitting per image. A child contributes several radiographs, "
              "so mAP50 will be optimistic. Pass the release CSV for a per-patient split.")

    stats = convert_release(
        args.images, args.out, groups=groups, val_fraction=args.val_fraction, seed=args.seed,
        fractures_only=not args.keep_non_fractures,
    )
    print(f"images: {stats['n_images']}  boxes: {stats['n_boxes']}  "
          f"negative: {stats['n_negative_images']}  classes: {stats['n_classes']}")
    print(f"train/val: {stats['train_images']}/{stats['val_images']} (split by {stats['split_basis']})")
    print("\nclasses:")
    for cid, name in stats["classes"].items():
        print(f"  {cid}: {name}")
    leaked = stats["groups_in_both_splits"]
    if leaked:
        print(f"[error] {len(leaked)} group(s) in both splits, e.g. {leaked[:5]}")
    if stats["empty_splits"]:
        print(f"[warn] empty split(s): {', '.join(stats['empty_splits'])}. Lower --val-fraction or add "
              "images; no metric can be computed from an empty split.")
    if args.dry_run:
        print("\ndry run, nothing written")
        return
    print(f"\nwrote {args.out}/labels and {args.out}/data.yaml")
    (args.out / "convert_stats.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()