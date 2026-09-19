# Radiology AI

AI system for interpreting imaging studies — chest X-rays first (CXR), then CT and MRI. Specialist-level diagnostics for every healthcare worker. Portfolio project: Medicine × AI.

> **Read this in:** [English](README.en.md) | [Русский](README.ru.md) | [中文](README.zh.md)

## Scope

- **Dataset:** VinDr-CXR (PhysioNet / Kaggle) — 15,000 chest X-rays.
- **Task:** multi-label classification → finding scoring.
- **Model — 15 classes (Phase 1 complete):** 11 lung (atelectasis, consolidation, infiltration, ground-glass opacity, pleural effusion/thickening, pneumothorax, fibrosis, nodule/mass, etc.) + cardiovascular/other: Cardiomegaly, Aortic enlargement, Calcification, ILD + "No finding".
- **Backbone:** `tf_efficientnet_b0` (timm), 512×512, AMP, AdamW + cosine.
- **Metrics:** macro AUROC / macro Average Precision + per-class ROC curves.

## Installation

```bash
uv sync
# data: download VinDr-CXR from Kaggle/PhysioNet, put into data/vindr/
uv run vindr-prepare --data-dir data/vindr --smoke
```

## Training

```bash
uv run vindr-train --config configs/train_full.yaml   # Phase 1: 15 classes
# or manually:
uv run vindr-train \
  --data-dir data/vindr --images-dir data/vindr/images \
  --labels all --backbone tf_efficientnet_b0 \
  --image-size 512 --epochs 12 --batch-size 4 \
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
- ROC curves in `reports/demo/roc_curves.png` — the closer a curve is to the top-left corner, the better the class separates (Phase 1 model macro-AUROC 0.950);
- rare classes (e.g. pneumothorax, 96 train images) are detected weaker — expected given the class imbalance.

### Metrics (Phase 1, `runs/full_v1_b0_512/best.pt`)

macro-AUROC **0.950** on val (n=1507), all 15 classes ≥ 0.85:

| Class | AUROC | Class | AUROC |
|-------|-------|-------|-------|
| No finding | 0.991 | Infiltration | 0.955 |
| Pleural effusion | 0.986 | Pulmonary fibrosis | 0.956 |
| Cardiomegaly | 0.985 | Lung Opacity | 0.952 |
| Aortic enlargement | 0.984 | Pleural thickening | 0.950 |
| ILD | 0.971 | Nodule/Mass | 0.941 |
| Consolidation | 0.964 | Calcification | 0.927 |
| Atelectasis | 0.917 | Other lesion | 0.913 |
| | | Pneumothorax | 0.850 |

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
| **Cardiomegaly** | enlarged cardiac silhouette: cardiothoracic ratio > 0.5 |
| **Aortic enlargement** | widening of the aortic arch/ascending aorta; if marked — Echo/CT angiography |
| **Calcification** | calcification: aortic arches, vessel walls, pleural plaques |
| **ILD** | interstitial lung involvement: thickened interlobular septa, peribronchovascular lines, reticular pattern |

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
configs/train_full.yaml    # Phase 1 config (15 classes)
src/vindr/
  labels.py             # 28 classes (full VinDr) + available subset, splits
  data.py               # Dataset (DICOM/PNG + augmentations)
  model.py              # timm backbone + multi-label head
  metrics.py            # per-class AUROC / AP + macro
  i18n.py               # label translations (en/ru/zh)
  prepare.py            # data prep, smoke test
  eda.py                # EDA report (class frequency, image stats)
  gradcam.py            # visualize "where the model looks"
  plots.py              # per-class ROC curves
  train.py              # training loop (--config, AMP, logging, best.pt)
  predict.py            # single-image inference + protocol report
  report.py             # medical knowledge base: glossary, diff rules, protocol
  app.py                # FastAPI web demo
scripts/generate_cam.py # CLI: Grad-CAM overlay from a checkpoint
```

## Roadmap

- [x] Scaffold: train / predict / prepare
- [x] Lung labels (`LUNG_LABELS`, mean AUROC 0.949)
- [x] Grad-CAM visualizations
- [x] EDA report
- [x] Per-class ROC curves
- [x] Web demo (FastAPI) with multilingual output (en/ru/zh)
- [x] **Phase 1 complete:** all 15 available VinDr-CXR classes, macro-AUROC **0.950** (`runs/full_v1_b0_512/best.pt`)
- [ ] Phase 2: finding detector (YOLOv8s/RT-DETR) on `vindr-cxr-coco`
- [ ] Head CT (RSNA ICH), brain MRI (BraTS) — "multi-modality triage" package
- [ ] API service in Docker
  Full map — [ROADMAP.en.md](ROADMAP.en.md)

## License

Code — **GPL-3.0-or-later** (copyleft: any versions and improvements must stay open). VinDr-CXR data — under VinGroup/PhysioNet license (registration required).