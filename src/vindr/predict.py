"""Inference: predict labels for images and rank findings."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from vindr.data import read_image
from vindr.detect import detect, load_default_detector, render_overlay
from vindr.i18n import label_name, LANGUAGES
from vindr.model import build_model
from vindr.report import render_protocol

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vindr.predict")


@torch.no_grad()
def predict_image(model, image_path: str | Path, image_size: int = 512, device=None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arr = read_image(Path(image_path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    x = transform(image=arr)["image"].unsqueeze(0).to(device)
    model.to(device).eval()
    return model(x).sigmoid().cpu().numpy()[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="best.pt checkpoint")
    ap.add_argument("--image", required=True, help="path to a .dcm or .png image")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--lang", choices=LANGUAGES, default="en", help="output language (en/ru/zh)")
    ap.add_argument("--detect", action="store_true",
                    help="also run YOLOv8 detector and save bbox overlay (Phase 2)")
    ap.add_argument("--det-ckpt", default=None, help="path to detector weights (default: latest runs_det best.pt)")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    labels = ckpt.get("labels", [])
    model = build_model(backbone=ckpt.get("backbone", "tf_efficientnet_b0"), num_classes=len(labels))
    model.load_state_dict(ckpt["model_state"])

    probs = predict_image(model, args.image)
    order = np.argsort(probs)[::-1][: args.top_k]
    log.info("top-%-d findings for %s (%s):", args.top_k, args.image, args.lang)
    for i in order:
        if probs[i] > 0.1:
            log.info("  %-24s %.3f", label_name(labels[i], args.lang), probs[i])
    log.info("")
    log.info(
        "%s",
        render_protocol(
            probs.tolist(),
            labels,
            args.lang,
            image_path=str(args.image),
            top_k=args.top_k,
        ),
    )

    if args.detect:
        det = load_default_detector(args.det_ckpt)
        finds = detect(det, args.image)
        out = Path(args.image).with_name(Path(args.image).stem + "_det.jpg")
        render_overlay(args.image, finds, out_path=out)
        log.info("")
        log.info("detector: %d found (saved %s):", len(finds), out)
        for f in finds:
            log.info("  %-24s conf=%.3f  bbox=%s", label_name(f["name"], args.lang), f["conf"], f["bbox"])


if __name__ == "__main__":
    main()