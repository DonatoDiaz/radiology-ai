# Lung Radiology AI

AI classifier for **chest X-rays** to triage lung pathologies. Portfolio project: Medicine × AI.

> **Read this in:** [English](README.en.md) | [Русский](README.ru.md) | [中文](README.zh.md)

## Scope

- **Dataset:** VinDr-CXR (PhysioNet / Kaggle) — 15,000 chest X-rays.
- **Task:** multi-label classification → finding scoring.
- **Lung focus:** `LUNG_LABELS` — 18 classes (17 lung parenchyma/pleura/airway pathologies + "No finding"): atelectasis, consolidation, emphysema, infiltration, pleural effusion, pneumothorax, fibrosis, nodule/mass, etc.
- **Backbone:** `tf_efficientnet_b0` (timm), 512×512, AMP, AdamW + cosine.
- **Metrics:** mean AUROC / mean Average Precision + per-class ROC curves.

## Installation

```bash
uv sync
# data: download VinDr-CXR from Kaggle/PhysioNet, put into data/vindr/
uv run vindr-prepare --data-dir data/vindr --smoke
```

## Training

```bash
uv run vindr-train \
  --data-dir data/vindr \
  --backbone tf_efficientnet_b0 \
  --image-size 512 --epochs 25 --batch-size 16 \
  --out-dir runs
```

Best checkpoint: `runs/<backbone>_<size>/best.pt`.

## EDA report

```bash
uv run vindr-eda --data-dir data/vindr --labels lung --out reports
# plots: class_frequency.png, top_combinations.png, summary.md
```

## Inference and Grad-CAM

```bash
uv run vindr-predict --ckpt runs/.../best.pt --image case_001.dcm --top-k 5 --lang en
uv run python scripts/generate_cam.py --ckpt runs/.../best.pt --image case_001.dcm \
  --class-index 10 --out cam_overlay.png
```

`--lang en|ru|zh` selects the output language for findings.

### How to interpret the result

`predict` outputs independent probabilities (0–1) per finding (multi-label, so the sum ≠ 1):

| Score | Interpretation |
|-------|----------------|
| > 0.9   | confident finding |
| 0.5–0.9 | probable finding (worth looking at the image) |
| 0.2–0.5 | uncertain / early sign |
| < 0.2   | essentially "none" |

Sanity checks:
- a **normal** image should give `No finding ≈ 1.0`, others ≈ 0;
- ROC curves in `reports/demo/roc_curves.png` — the closer a curve is to the top-left corner, the better the class separates (model mean AUROC 0.949);
- rare classes (e.g. pneumothorax, 96 train images) are detected weaker — expected given the class imbalance.

### Findings glossary (radiological semiotics)

Short radiological descriptions of the model labels (from textbook radiology/CT), printed in the "DESCRIPTION" block of the report:

| Label | Radiological semiotics |
|-------|------------------------|
| **Consolidation** | increased density with complete obliteration of alveolar air spaces, vessels no longer visible; typical triangular shadow with its base toward the pleura (lobar pneumonia) |
| **Lung Opacity** | ground-glass: increased density while vessels and lobules stay visible (alveolar filling, interalveolar septal thickening) |
| **Infiltration** | ill-defined areas of density with perifocal inflammation; differential: pneumonia / tuberculosis |
| **Atelectasis** | collapsed lung tissue; air bronchograms → compressive, their absence → obstructive |
| **Nodule/Mass** | focus up to 1 cm, nodule/mass larger; lobulated/irregular borders and growth → CT, verification when > ~1 cm |
| **Pleural effusion** | fluid in the pleural space (blunted sinus); empyema when secondarily infected |
| **Pneumothorax** | air in the pleural space with lung collapse, no lung markings |
| **Pleural thickening** | parietal pleura thickening, plaques, subpleural linear densities |
| **Pulmonary fibrosis** | intralobular septal thickening, reticular densities, traction bronchiectasis; honeycombing in the end stage |

### Protocol-style report

Besides the top-k list, `predict` renders a protocol-style text (DESCRIPTION → CONCLUSION → RECOMMENDATIONS, including differential-diagnosis hints) that helps the practitioner interpret the study at the level of a specialist radiologist:

```bash
uv run vindr-predict --ckpt runs/.../best.pt --image case_001.dcm --lang ru
```

## Web demo

```bash
uv run uvicorn vindr.app:app --port 8000
# http://localhost:8000  — upload image → predictions (language select: en/ru/zh)
# http://localhost:8000/predict/cam — Grad-CAM overlay (POST)
```

## Structure

```
configs/train.yaml      # training config
src/vindr/
  labels.py             # 28 classes + LUNG_LABELS subset, splits
  data.py               # Dataset (DICOM/PNG + augmentations)
  model.py              # timm backbone + multi-label head
  metrics.py            # per-class AUROC / AP + macro
  i18n.py               # label translations (en/ru/zh)
  prepare.py            # data prep, smoke test
  eda.py                # EDA report (class frequency, image stats)
  gradcam.py            # visualize "where the model looks"
  plots.py              # per-class ROC curves
  train.py              # training loop (AMP, logging, best.pt)
  predict.py            # single-image inference + protocol report
  report.py             # medical knowledge base: glossary, diff rules, protocol
  app.py                # FastAPI web demo
scripts/generate_cam.py # CLI: Grad-CAM overlay from a checkpoint
```

## Roadmap

- [x] Scaffold: train / predict / prepare
- [x] Lung focus (`LUNG_LABELS`)
- [x] Grad-CAM visualizations
- [x] EDA report
- [x] Per-class ROC curves
- [x] Web demo (FastAPI) with multilingual output (en/ru/zh)
- [x] Download & prepare VinDr-CXR (Kaggle, 15,000 images), train first model: **mean AUROC 0.949** (EffNet-B0, 512px, 5 epochs; weights `runs/lung_v1/best.pt`)
- [ ] Head CT (RSNA ICH), brain MRI (BraTS) — "multi-modality triage" package
- [ ] API service in Docker

## License

Code — **GPL-3.0-or-later** (copyleft: any versions and improvements must stay open). VinDr-CXR data — under VinGroup/PhysioNet license (registration required).