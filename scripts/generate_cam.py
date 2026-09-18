"""Generate Grad-CAM overlays for an image given a checkpoint.

    uv run python scripts/generate_cam.py --ckpt runs/best.pt --image case_001.dcm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

from vindr.data import read_image
from vindr.gradcam import GradCAM
from vindr.model import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Grad-CAM overlay generation")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--class-index", type=int, default=None, help="target class; default = argmax")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="cam_overlay.png")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    labels = ckpt.get("labels", [])
    model = build_model(backbone=ckpt.get("backbone", "tf_efficientnet_b0"), num_classes=len(labels))
    model.load_state_dict(ckpt["model_state"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    arr = read_image(Path(args.image))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    tr = A.Compose(
        [
            A.Resize(args.size, args.size),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    x = tr(image=arr)["image"].unsqueeze(0).to(device)

    cam = GradCAM(model)
    overlay = cam.overlay(x, target_class=args.class_index)
    Image.fromarray(overlay).save(args.out)
    idx = args.class_index if args.class_index is not None else int(model(x).argmax(1))
    print(f"saved {args.out}  target_class={idx} ({labels[idx] if idx < len(labels) else '?'})")


if __name__ == "__main__":
    main()