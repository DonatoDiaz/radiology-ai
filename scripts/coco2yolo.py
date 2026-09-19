#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert VinDr-CXR COCO annotations to YOLO label files")
    ap.add_argument("--coco-dir", type=Path, required=True,
                    help="dir with annotations/instances_*.json and images/{train,val}")
    ap.add_argument("--splits", default="train,val",
                    help="comma-separated splits to convert (default: train,val)")
    ap.add_argument("--labels-out", type=Path, default=None,
                    help="overwrite labels dir (default: <coco-dir>/labels)")
    args = ap.parse_args()

    coco_dir = args.coco_dir
    ann_dir = coco_dir / "annotations"

    for split in args.splits.split(","):
        ann_path = ann_dir / f"instances_{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"missing {ann_path}")
        data = json.loads(ann_path.read_text())

        cat2cls = {c["id"]: i for i, c in enumerate(data["categories"])}
        img2file = {im["id"]: im for im in data["images"]}

        anns_by_img: dict[int, list] = {}
        for a in data["annotations"]:
            anns_by_img.setdefault(a["image_id"], []).append(a)

        labels_out = args.labels_out or (coco_dir / "labels")
        out_dir = labels_out / split
        out_dir.mkdir(parents=True, exist_ok=True)

        n_written = 0
        n_bad = 0
        for im in data["images"]:
            w, h = im["width"], im["height"]
            lines = []
            for a in anns_by_img.get(im["id"], []):
                cls = cat2cls[a["category_id"]]
                x, y, bw, bh = a["bbox"]
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                bw_n, bh_n = bw / w, bh / h
                if cx < 0 or cy < 0 or bw_n <= 0 or bh_n <= 0:
                    n_bad += 1
                    continue
                if cx > 1 or cy > 1 or cx + bw_n / 2 > 1 or cy + bh_n / 2 > 1:
                    bw_n = min(2 * min(cx, 1 - cx), bw_n)
                    bh_n = min(2 * min(cy, 1 - cy), bh_n)
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
            txt = Path(im["file_name"]).with_suffix(".txt")
            (out_dir / txt).write_text("\n".join(lines) + ("\n" if lines else ""))
            n_written += 1

        n_cls = len(data["categories"])
        n_anns = len(data["annotations"])
        print(f"[{split}] wrote {n_written} labels -> {out_dir} "
              f"(classes={n_cls}, anns={n_anns}, skipped_bad={n_bad})")


if __name__ == "__main__":
    main()