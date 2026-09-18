# Lung Radiology AI

AI-классификатор **рентгенограмм грудной клетки** для триажной диагностики лёгочных патологий. Портфолио-проект: Медицина × ИИ.

## Scope

- **Датасет:** VinDr-CXR (PhysioNet / Kaggle) — 15 000 рентгенограмм.
- **Задача:** мульти-лейбл классификация → скоринг находок.
- **Фокус — лёгкие:** `LUNG_LABELS` — 18 классов (17 патологий паренхимы/плевры/дыхательных путей + «No finding»): ателектаз, консолидация, эмфизема, инфильтрация, плевральный выпот, пневмоторакс, фиброз, узлы/массы и др.
- **Backbone:** `tf_efficientnet_b0` (timm), 512×512, AMP, AdamW + cosine.
- **Метрики:** mean AUROC / mean Average Precision + ROC-кривые по классам.

## Установка

```bash
uv sync
# данные: скачать VinDr-CXR с PhysioNet (нужна регистрация), положить в data/vindr/
uv run vindr-prepare --data-dir data/vindr --smoke
```

## Обучение

```bash
uv run vindr-train \
  --data-dir data/vindr \
  --backbone tf_efficientnet_b0 \
  --image-size 512 --epochs 25 --batch-size 16 \
  --out-dir runs
```

Лучший чекпоинт: `runs/<backbone>_<size>/best.pt`.

## EDA-отчёт

```bash
uv run vindr-eda --data-dir data/vindr --labels lung --out reports
# графики: class_frequency.png, top_combinations.png, summary.md
```

## Инференс и Grad-CAM

```bash
uv run vindr-predict --ckpt runs/.../best.pt --image case_001.dcm --top-k 5
uv run python scripts/generate_cam.py --ckpt runs/.../best.pt --image case_001.dcm \
  --class-index 10 --out cam_overlay.png
```

## Веб-демо

```bash
uv run uvicorn vindr.app:app --port 8000
# http://localhost:8000  — загрузка снимка → предсказания
# http://localhost:8000/predict/cam — Grad-CAM оверлей (POST)
```

## Структура

```
configs/train.yaml      # конфиг обучения
src/vindr/
  labels.py             # 28 классов + подмножество LUNG_LABELS, сплиты
  data.py               # Dataset (DICOM/PNG + аугментации)
  model.py              # timm backbone + мульти-лейбл голова
  metrics.py            # AUROC / AP по классам и macro
  prepare.py            # подготовка данных, smoke-тест
  eda.py                # EDA-отчёт (частоты классов, статистика снимков)
  gradcam.py            # визуализация «куда смотрит модель»
  plots.py              # ROC-кривые по классам
  train.py              # цикл обучения (AMP, логгинг, best.pt)
  predict.py            # инференс по одному снимку
  app.py                # FastAPI веб-демо
scripts/generate_cam.py # CLI: Grad-CAM оверлей по чекпоинту
```

## Roadmap

- [x] Каркас: train / predict / prepare
- [x] Фокус на лёгкие (`LUNG_LABELS`)
- [x] Grad-CAM визуализации
- [x] EDA-отчёт
- [x] ROC-кривые по классам
- [x] Веб-демо (FastAPI)
- [x] Скачать и подготовить VinDr-CXR (Kaggle, 15 000 снимков), обучить первую модель: **mean AUROC 0.949** (Эф-нет-В0, 512px, 5 эпох; веса `runs/lung_v1/best.pt`)
- [ ] КТ-голова (RSNA ICH), МРТ-мозг (BraTS) — пакет «триаж по модальностям»
- [ ] API-сервис в Docker

## Лицензия

Код — **GPL-3.0-or-later** (copyleft: любые версии и улучшения обязаны оставаться открытыми). Данные VinDr-CXR — по лицензии VinGroup/PhysioNet (требуется регистрация).