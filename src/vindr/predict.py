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
from vindr.model import build_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vindr.predict")


@torch.no_grad()
def predict_image(model, image_path: str | Path, image_size: int = 512, device=None) -> np.ndarray:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    arr = read_image(Path(image_path))
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
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    labels = ckpt.get("labels", [])
    model = build_model(backbone=ckpt.get("backbone", "tf_efficientnet_b0"), num_classes=len(labels))
    model.load_state_dict(ckpt["model_state"])

    probs = predict_image(model, args.image)
    order = np.argsort(probs)[::-1][: args.top_k]
    log.info("top-%-d findings for %s:", args.top_k, args.image)
    for i in order:
        if probs[i] > 0.1:
            log.info("  %-24s %.3f", labels[i], probs[i])


if __name__ == "__main__":
    main()