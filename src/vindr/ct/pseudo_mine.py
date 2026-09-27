"""Turn a trained head CT classifier into a pseudo-masked dataset (Phase 3).

RSNA labels whole slices, so there is nothing to segment on. This script runs
the weakly-supervised path in :mod:`vindr.ct.pseudo` — CAM seeds, GrabCut, a
density band — and writes ``mask.npy`` next to each image, which is exactly the
layout :func:`vindr.ct.dataset.load_masked_studies` reads. The result trains
the U-Net in :mod:`vindr.ct.segmentation`; it is a bootstrap, not a ground truth.

Run::

    uv run python -m vindr.ct.pseudo_mine --ckpt runs/head_ct_v1_2p5d/best.pt \\
        --data-dir ./data/head_ct --out-dir ./data/head_ct_pseudo
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from vindr.ct.dataset import load_study_folders
from vindr.ct.predict import load_model
from vindr.ct.pseudo import (
    DEFAULT_BLOOD_HU,
    DEFAULT_CORE_FRACTION,
    DEFAULT_MIN_SHARPNESS,
    DEFAULT_SEED_QUANTILE,
    pseudo_mask_study,
    write_pseudo_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vindr.ct.pseudo_mine")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", required=True, help="classifier checkpoint from vindr-ct-train")
    ap.add_argument("--data-dir", required=True, help="study root (one subfolder per study)")
    ap.add_argument("--out-dir", required=True, help="where to write <study_id>/mask.npy")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N studies")
    ap.add_argument("--prob-threshold", type=float, default=0.5, help="min per-slice confidence")
    ap.add_argument("--min-sharpness", type=float, default=DEFAULT_MIN_SHARPNESS,
                    help="min CAM concentration (top 2%% over the mean)")
    ap.add_argument("--seed-quantile", type=float, default=DEFAULT_SEED_QUANTILE)
    ap.add_argument("--core-fraction", type=float, default=DEFAULT_CORE_FRACTION)
    ap.add_argument("--blood-hu", type=float, nargs=2, default=list(DEFAULT_BLOOD_HU),
                    metavar=("LOW", "HIGH"), help="acute-blood density band in HU")
    ap.add_argument("--no-grabcut", action="store_true", help="skip GrabCut, keep the density band only")
    ap.add_argument("--class-index", type=int, default=-1,
                    help="classifier column to explain (-1 = 'any'; 0..4 = subtype index)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, blob = load_model(args.ckpt, device)
    shape = tuple(blob.get("target_shape", (32, 224, 224)))
    context = int(blob.get("in_channels", 3) - 1) // 2
    log.info("checkpoint %s (%s) on %s", args.ckpt, blob.get("model", "study"), device.type)

    studies = load_study_folders(args.data_dir)
    if args.limit:
        studies = studies[: args.limit]
    if not studies:
        raise SystemExit(f"no studies under {args.data_dir}")

    reports, skipped = [], 0
    for i, st in enumerate(studies, 1):
        rep = pseudo_mask_study(
            model, st,
            context=context, target_shape=shape, class_index=args.class_index,
            prob_threshold=args.prob_threshold, seed_quantile=args.seed_quantile,
            min_sharpness=args.min_sharpness, core_fraction=args.core_fraction,
            blood_hu=tuple(args.blood_hu),
            use_grabcut=not args.no_grabcut,
        )
        if rep is None:
            skipped += 1
            continue
        reports.append(rep)
        log.info(
            "[%d/%d] %s: %d slices in %s, max_prob=%.3f",
            i, len(studies), rep["study_id"], rep["n_masked_slices"], rep["z_range"], rep["max_prob"],
        )

    if not reports:
        raise SystemExit(
            "no study produced a mask — lower --prob-threshold or --min-sharpness, or check that "
            "the checkpoint matches the data window and that the labels describe these studies"
        )
    written = write_pseudo_dataset(reports, args.out_dir, image_root=args.data_dir)
    summary = {
        "studies_total": len(studies),
        "studies_masked": len(written),
        "studies_skipped": skipped,
        "ckpt": str(args.ckpt),
        "prob_threshold": args.prob_threshold,
        "min_sharpness": args.min_sharpness,
        "blood_hu": list(args.blood_hu),
        "class_index": args.class_index,
        "use_grabcut": not args.no_grabcut,
        "target_shape": list(shape),
    }
    (Path(args.out_dir) / "pseudo_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    log.info("masked %d/%d studies (%d skipped) -> %s", len(written), len(studies), skipped, args.out_dir)
    log.info("next: uv run vindr-ct-seg-train --config configs/train_head_ct_seg.yaml --data-dir %s", args.out_dir)


if __name__ == "__main__":
    main()
