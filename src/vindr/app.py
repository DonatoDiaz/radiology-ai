"""Minimal web demo: upload an X-ray image, see predicted findings.

    uv run uvicorn vindr.app:app --reload --port 8000
"""

from __future__ import annotations

import io
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from PIL import Image

from vindr.data import read_image
from vindr.gradcam import GradCAM
from vindr.i18n import label_name, LANGUAGES
from vindr.labels import LUNG_LABELS
from vindr.model import build_model

app = FastAPI(title="Lung Radiology AI Demo", docs_url="/docs")

CKPT_PATH = Path(__file__).resolve().parent.parent.parent / "runs" / "best.pt"
_image_size = 512
_model = None
_cam = None
_model_labels: list[str] = []


def _load_model():
    global _model, _cam, _model_labels
    if _model is not None:
        return
    if not CKPT_PATH.exists():
        raise RuntimeError(
            f"No checkpoint found at {CKPT_PATH}. Run training first:  python -m vindr.train"
        )
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    _model_labels = ckpt.get("labels", LUNG_LABELS)
    _model = build_model(
        backbone=ckpt.get("backbone", "tf_efficientnet_b0"),
        num_classes=len(_model_labels),
    )
    _model.load_state_dict(ckpt["model_state"])
    _cam = GradCAM(_model)


def _preprocess(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    tr = A.Compose(
        [
            A.Resize(_image_size, _image_size),
            A.Normalize(mean=0.5, std=0.5, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return tr(image=arr)["image"].unsqueeze(0)


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Lung Radiology AI</title></head>
<body>
  <h2>Upload Chest X-ray</h2>
  <form method=post enctype=multipart/form-data action=/predict>
    <input name=file type=file accept="image/*,application/dicom,.dcm">
    <select name=lang>
      <option value="en">English</option>
      <option value="ru" selected>Русский</option>
      <option value="zh">中文</option>
    </select>
    <button type=submit>Predict</button>
  </form>
</body>
</html>
"""

@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile, lang: str = Form("ru")):
    _load_model()
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("L")
    arr = np.asarray(image)
    x = _preprocess(arr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device).eval()
    with torch.no_grad():
        logits = _model(x.to(device))
    probs = logits.sigmoid().cpu().numpy()[0]
    rows = "\n".join(
        f"<tr><td>{label_name(l, lang)}</td><td>{probs[i]:.3f}</td></tr>"
        for i, l in enumerate(_model_labels)
        if not l.startswith("No finding") and probs[i] > 0.05
    ) or f"<tr><td colspan=2>{label_name('No finding', lang)}</td></tr>"
    return f"""
<table border=1>
  <tr><th>Finding / Находка / 发现</th><th>Confidence</th></tr>
  {rows}
</table>
"""

@app.post("/predict/cam")
async def predict_cam(file: UploadFile):
    """Return a GradCAM overlay JPEG for the top-1 finding."""
    _load_model()
    raw = await file.read()
    image = Image.open(io.BytesIO(raw)).convert("L")
    arr = np.asarray(image)
    x = _preprocess(arr)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model.to(device).eval()
    logits = _model(x.to(device))
    top_class = int(logits.argmax(dim=1))
    overlay = _cam.overlay(x.to(device), target_class=top_class)
    pil_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")