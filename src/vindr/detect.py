"""Detection: run YOLOv8 detector on an image and render bounding-box overlay."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

DET_COLORS = {
    "Aortic enlargement": "#e6194b",
    "Pleural thickening": "#3cb44b",
    "Pleural effusion": "#ffe119",
    "Cardiomegaly": "#4363d8",
    "Lung Opacity": "#f58231",
    "Nodule/Mass": "#911eb4",
    "Consolidation": "#42d4f4",
    "Pulmonary fibrosis": "#f032e6",
    "Infiltration": "#bfef45",
    "Atelectasis": "#fabed4",
    "Other lesion": "#469990",
    "ILD": "#dcbeff",
    "Pneumothorax": "#9a6324",
    "Calcification": "#800000",
}


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def detect(model, image_path: str | Path, conf: float = 0.25, iou: float = 0.45):
    """Run detector; return list of dicts {bbox, class_id, name, conf}."""
    model = model or _detector()
    res = model.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]
    names = res.names
    out = []
    for box in res.boxes:
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
        cid = int(box.cls[0])
        out.append(
            {
                "bbox": [x1, y1, x2, y2],
                "class_id": cid,
                "name": names[cid],
                "conf": float(box.conf[0]),
            }
        )
    return out


def render_overlay(image_path: str | Path, detections: list[dict], out_path: str | Path | None = None) -> np.ndarray:
    """Draw bounding boxes + labels on the image; save if out_path given, return RGB array."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    font = _load_font(max(14, img.width // 80))
    for d in detections:
        color = DET_COLORS.get(d["name"], "#000000")
        x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=max(2, img.width // 400))
        label = f"{d['name']} {d['conf']:.2f}"
        draw.rectangle([x1, y1, x1 + len(label) * 9, y1 + 18], fill=color)
        draw.text((x1 + 2, y1 + 1), label, fill="black", font=font)
    arr = np.asarray(img)
    if out_path:
        Image.fromarray(arr).save(out_path)
    return arr


class DetectionModel:
    """Lazy-loaded ultralytics YOLO wrapper."""

    def __init__(self, weights: str):
        self.weights = str(weights)
        self._model = None

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, source, conf: float = 0.25, iou: float = 0.45, verbose: bool = False):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self.weights)
        return self._model.predict(source=source, conf=conf, iou=iou, verbose=verbose)


_default_weights = None


def load_default_detector(ckpt: str | Path | None = None):
    """Create a DetectionModel from runs_det or explicit weights path."""
    global _default_weights
    if ckpt is None:
        if _default_weights is None:
            cands = sorted(
                Path("/run/media/donatodiaz/w_/Github/radiology/runs_det").glob("*/weights/best.pt"),
                key=lambda p: p.stat().st_mtime,
            )
            if not cands:
                raise RuntimeError("no detection checkpoint found; run Phase 2 training first")
            _default_weights = str(cands[-1])
        ckpt = _default_weights
    return DetectionModel(ckpt)