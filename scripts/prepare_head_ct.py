#!/usr/bin/env python3
"""Prepare a head CT study tree from the RSNA ICH 16-bit PNG dataset.

Source layout (``ianpan/rsna-intracranial-hemorrhage-16bit-png``, MIT):

    <root>/
      slice_labels.csv     ID,hemorrhage_type
      rescale_values.csv   ID,rescale_slope,rescale_intercept
      images/              <study_id>_<slice>.png   (16-bit, stored=hu*(-slope)+intercept)

This script turns it into the layout the trainer expects: one subfolder per
study plus a single ``study_labels.csv`` used by ``vindr-ct-train``.

    python scripts/prepare_head_ct.py \\
        --png-dir  /data/rsna_ich/png \\
        --labels   /data/rsna_ich/slice_labels.csv \\
        --rescale  /data/rsna_ich/rescale_values.csv \\
        --out      /data/head_ct

Slice IDs look like ``id_61e1e0fd3d4d544ee8b3ce1a2a6a9a53_0001`` — the study
key is everything before the final ``_<index>`` group.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HEMORRHAGE_TYPES = (
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
)
SPLIT_NAMES = ("train", "val")


def split_id(slice_id: str) -> str:
    """``<study>_<nnn>`` -> ``<study>``; tolerates ids without an index."""
    head, sep, tail = slice_id.rpartition("_")
    if sep and tail.isdigit() and head:
        return head
    return slice_id


def read_slice_labels(path: Path) -> dict[str, set[str]]:
    """Read slice_labels.csv -> {study_id: {hemorrhage_type, ...}}."""
    out: dict[str, set[str]] = defaultdict(set)
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("ID") or row.get("id") or "").strip()
            if not sid:
                continue
            labels = (row.get("hemorrhage_type") or "").strip()
            if not labels:
                continue
            for name in labels.split(","):
                name = name.strip().lower().replace(" ", "_")
                if name in HEMORRHAGE_TYPES:
                    out[split_id(sid)].add(name)
    return out


def read_rescale(path: Path | None) -> dict[str, tuple[float, float]]:
    """Read rescale_values.csv -> {study_id: (slope, intercept)} (median across slices)."""
    acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if path is None or not path.exists():
        return {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("ID") or row.get("id") or "").strip()
            if not sid:
                continue
            try:
                slope = float(row.get("rescale_slope", 1.0) or 1.0)
                inter = float(row.get("rescale_intercept", 0.0) or 0.0)
            except ValueError:
                continue
            acc[split_id(sid)].append((slope, inter))
    out: dict[str, tuple[float, float]] = {}
    for study, pairs in acc.items():
        slope = sorted(p[0] for p in pairs)[len(pairs) // 2]
        inter = sorted(p[1] for p in pairs)[len(pairs) // 2]
        out[study] = (slope, inter)
    return out


def index_pngs(png_dir: Path) -> dict[str, list[Path]]:
    """Group ``<study>_<nnn>.png`` files per study, sorted by slice index."""
    studies: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in sorted(png_dir.rglob("*.png")):
        sid = p.stem
        study = split_id(sid)
        tail = sid.rpartition("_")[2]
        studies[study].append((int(tail) if tail.isdigit() else 0, p))
    return {k: [p for _, p in sorted(v, key=lambda t: t[0])] for k, v in studies.items()}


def link_or_copy(files: list[Path], dest: Path, copy: bool) -> int:
    """Materialise the slices of one study into ``dest``; returns slice count."""
    import shutil

    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in files:
        target = dest / f.name
        if target.exists():
            n += 1
            continue
        if copy:
            shutil.copy2(f, target)
        else:
            target.symlink_to(f.resolve())
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--png-dir", type=Path, required=True, help="dir with the 16-bit PNG slices")
    ap.add_argument("--labels", type=Path, required=True, help="slice_labels.csv")
    ap.add_argument("--rescale", type=Path, default=None, help="rescale_values.csv (optional)")
    ap.add_argument("--out", type=Path, required=True, help="output study tree")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--copy", action="store_true", help="copy slices instead of symlinking")
    ap.add_argument("--max-studies", type=int, default=0, help="limit for a dry run (0 = all)")
    ap.add_argument("--negative-only", action="store_true", help="keep only studies with no hemorrhage")
    ap.add_argument("--positive-only", action="store_true", help="keep only studies with hemorrhage")
    ap.add_argument("--dry-run", action="store_true", help="report counts without writing")
    args = ap.parse_args()

    if not args.png_dir.exists():
        raise SystemExit(f"png dir not found: {args.png_dir}")
    if not args.labels.exists():
        raise SystemExit(f"labels not found: {args.labels}")
    if args.negative_only and args.positive_only:
        raise SystemExit("--negative-only and --positive-only are mutually exclusive")

    slice_labels = read_slice_labels(args.labels)
    rescale = read_rescale(args.rescale)
    by_study = index_pngs(args.png_dir)
    if not by_study:
        raise SystemExit(f"no PNG slices found under {args.png_dir}")

    kept: list[str] = []
    for study in sorted(by_study):
        pos = bool(slice_labels.get(study))
        if args.negative_only and pos:
            continue
        if args.positive_only and not pos:
            continue
        kept.append(study)
        if args.max_studies and len(kept) >= args.max_studies:
            break

    n_pos = sum(1 for s in kept if slice_labels.get(s))
    n_slices = sum(len(by_study[s]) for s in kept)
    print(f"studies: {len(kept)} ({n_pos} positive / {len(kept) - n_pos} negative), slices: {n_slices}")
    print(f"rescale table: {'loaded' if rescale else 'absent — assuming HU in pixel values'}")
    if not slice_labels:
        print("WARNING: no hemorrhage labels parsed; all studies will be negative", file=sys.stderr)

    if args.dry_run:
        print("dry run — nothing written")
        return

    # deterministic split: sort by study id so runs are reproducible
    import random

    order = list(kept)
    random.Random(42).shuffle(order)
    n_val = max(1, int(len(order) * args.val_fraction))
    val = set(order[:n_val])

    out_labels = args.out / "study_labels.csv"
    args.out.mkdir(parents=True, exist_ok=True)
    rows = ["study_id," + ",".join(HEMORRHAGE_TYPES) + ",any"]
    n_written = 0
    for study in kept:
        split = "val" if study in val else "train"
        dest = args.out / "studies" / study
        if dest.exists() and any(dest.iterdir()):
            n_slices_ok = len(list(dest.glob("*.png")))
        else:
            n_slices_ok = link_or_copy(by_study[study], dest, args.copy)
        if n_slices_ok != len(by_study[study]):
            print(
                f"  warning: {study} has {n_slices_ok}/{len(by_study[study])} slices on disk",
                file=sys.stderr,
            )
        types = sorted(slice_labels.get(study, ()))
        vec = ["1" if t in types else "0" for t in HEMORRHAGE_TYPES]
        rows.append(f"{study}," + ",".join(vec) + ("," + ("1" if types else "0")))
        n_written += 1
        if n_written % 200 == 0:
            print(f"  linked {n_written}/{len(kept)} studies -> {split}")

    (args.out / "splits.txt").write_text(
        "\n".join(f"{s} {'val' if s in val else 'train'}" for s in kept) + "\n"
    )
    slope_note = ""
    if rescale:
        sample = next(iter(rescale.items()))
        slope_note = (
            f"\nrescale: {sample[0]} -> slope={sample[1][0]}, intercept={sample[1][1]}"
            f" (written to <out>/rescale_values.csv)"
        )
        with open(args.out / "rescale_values.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["study_id", "rescale_slope", "rescale_intercept"])
            for study, (sl, ic) in rescale.items():
                if study in set(kept):
                    w.writerow([study, sl, ic])

    out_labels.write_text("\n".join(rows) + "\n")
    print(f"wrote {out_labels} ({n_written} studies){slope_note}")
    print(f"wrote {args.out / 'splits.txt'} (val={len(val)}, train={len(kept) - len(val)})")
    print("next: uv run vindr-ct-train --config configs/train_head_ct.yaml "
          f"--data-dir {args.out / 'studies'} --labels {out_labels}")


if __name__ == "__main__":
    main()
