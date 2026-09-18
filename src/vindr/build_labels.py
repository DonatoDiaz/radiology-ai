"""Convert VinDr detection annotations (bounding boxes) to classification labels.

The Kaggle `vindr-cxr-csv` file has one row per bounding box:
    image_id,class_name,class_id,rad_id,x_min,y_min,x_max,y_max

This script pivots it into a per-image multi-label matrix and writes
`train.csv` in the format our pipeline expects (28-character hex ids,
one binary column per class, plus `No finding`).

Usage:
    python -m vindr.build_labels --annotations train.csv --out data/vindr/train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_classification_labels(annotations: str | Path) -> pd.DataFrame:
    det = pd.read_csv(annotations)
    piv = det.pivot_table(
        index="image_id", columns="class_name", values="class_id",
        aggfunc="count", fill_value=0,
    ).reset_index()
    # binary: any box for a class -> 1
    label_cols = [c for c in piv.columns if c != "image_id"]
    piv[label_cols] = piv[label_cols].gt(0).astype(int)
    # 'No finding' normalized to keep parity with the pipeline.
    if "No finding" in piv.columns:
        piv["No finding"] = piv["No finding"].clip(upper=1)
    return piv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = build_classification_labels(args.annotations)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}  images={len(df)}  classes={len(df.columns) - 1}")

    langs_avail = [c for c in df.columns if c in ["Atelectasis", "Consolidation", "Infiltration",
        "Lung Opacity", "Nodule/Mass", "Pleural effusion", "Pleural thickening", "Pneumothorax",
        "Pulmonary fibrosis", "Other lesion", "No finding"]]
    print(f"lung-relevant classes present: {len(langs_avail)}")
    freq = df[langs_avail].sum(axis=0).sort_values(ascending=False)
    print(freq.to_string())


if __name__ == "__main__":
    main()