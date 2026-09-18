"""EDA report: class frequencies, image statistics, saved as figures + markdown summary."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vindr.labels import ALL_LABELS, LUNG_LABELS, load_meta_csv, load_train_csv

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vindr.eda")


def class_frequency_chart(df: pd.DataFrame, labels: list[str], out: Path) -> None:
    freq = df[labels].sum(axis=0).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(freq.index, freq.values, color="#4C72B0")
    ax.set_title("Positive case count per class")
    ax.set_xlabel("count")
    for i, v in enumerate(freq.values):
        ax.text(v, i, str(int(v)), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def top_combination_chart(df: pd.DataFrame, labels: list[str], n: int = 12, out: Path = None):
    """Top label combinations over the fraction of the sample."""
    combo = (
        df[labels]
        .apply(lambda r: "+".join(c for c in labels if r[c]), axis=1)
        .value_counts()
        .head(n)
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(combo.index[::-1], combo.values[::-1], color="#55A868")
    ax.set_title(f"Top {n} label combinations")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def image_statistics(images_dir: Path, image_ids: list[str], sample: int = 200):
    """Sample dimensions & value stats for a subset of images."""
    from PIL import Image

    rng = np.random.default_rng(0)
    ids = rng.choice(image_ids, size=min(sample, len(image_ids)), replace=False)
    rows = []
    for image_id in ids:
        for ext in (".dicom", ".dcm", ".png", ".jpg"):
            p = images_dir / f"{image_id}{ext}"
            if not p.exists():
                continue
            try:
                if p.suffix.lower() in (".dicom", ".dcm"):
                    from vindr.data import read_dicom

                    arr = read_dicom(p)
                else:
                    arr = np.asarray(Image.open(p))
                rows.append(
                    {
                        "w": arr.shape[1],
                        "h": arr.shape[0],
                        "flat": arr.ndim == 2,
                        "dtype": str(arr.dtype),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("skip %s: %s", image_id, exc)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="EDA report for VinDr-CXR dataset")
    ap.add_argument("--data-dir", default="./data/vindr")
    ap.add_argument("--images-dir", default=None)
    ap.add_argument("--out", default="./reports", help="output directory")
    ap.add_argument("--labels", default="all", choices=["all", "lung"])
    ap.add_argument("--sample-images", type=int, default=200)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    images_dir = Path(args.images_dir) if args.images_dir else data_dir / "train"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_train_csv(data_dir / "train.csv")
    meta = load_meta_csv(data_dir / "train_meta.csv")
    labels = LUNG_LABELS if args.labels == "lung" else ALL_LABELS
    if args.labels == "lung":
        from vindr.labels import restrict_to_lung

        df = restrict_to_lung(df)

    log.info("n_images=%d rows=%d classes=%d", len(df), len(df), len(labels))
    class_frequency_chart(df, labels, out / "class_frequency.png")
    top_combination_chart(df, labels, out=out / "top_combinations.png")

    stats = image_statistics(images_dir, list(df["image_id"]), sample=args.sample_images)
    summary_md = [
        "# VinDr-CXR EDA",
        "",
        f"- images: **{len(df)}**",
        f"- label columns: **{len(labels)}**",
        f"- modalities: {meta['modality'].value_counts().to_dict()}",
        f"- rows with a finding: **{(df[labels].sum(axis=1) >= 1).sum()}**",
        "",
        "## Image statistics (sampled)",
        "",
    ]
    if len(stats):
        desc = stats.describe().to_markdown()
        summary_md.append(f"dimensions: w={stats['w'].median():.0f} h={stats['h'].median():.0f}")
        summary_md.append("")
        summary_md.append("```")
        summary_md.append(desc)
        summary_md.append("```")
        summary_md.append("")
        log.info("median image size: %dx%d", stats["w"].median(), stats["h"].median())
    else:
        summary_md.append("_no images found in images_dir_")

    (out / "summary.md").write_text("\n".join(summary_md), encoding="utf-8")
    log.info("report written to %s", out / "summary.md")


if __name__ == "__main__":
    main()