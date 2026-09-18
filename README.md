# VinDr-CXR Multi-Label Classifier

Мульти-лейбл классификатор рентгенограмм грудной клетки для лучевой триажной диагностики (портфолио-проект).

## Scope

- Датасет: **VinDr-CXR** (PhysioNet / Kaggle) — 15 000 рентгенограмм, **28 классов** (22 патологии + 6 находок).
- Тип задачи: multi-label классификация → скоринг рентгенограмм.
- Backbone: `tf_efficientnet_b0` (timm), 512×512, AMP, AdamW + cosine.
- Метрики: **mean AUROC** и mean Average Precision по 28 классам.

## Установка

```bash
uv sync            # создаст venv и поставит зависимости
# данные: скачать VinDr-CXR с Kaggle/PhysioNet
python -m vindr.prepare --data-dir data/vindr --smoke
```

## Обучение

```bash
python -m vindr.train \
  --data-dir data/vindr \
  --backbone tf_efficientnet_b0 \
  --image-size 512 --epochs 25 --batch-size 16 \
  --out-dir runs
```

Лучший чекпоинт: `runs/<backbone>_<size>/best.pt`.

## Инференс

```bash
python -m vindr.predict --ckpt runs/tf_efficientnet_b0_512/best.pt --image case_001.png --top-k 5
```

## Структура

```
configs/train.yaml   # конфиг обучения
src/vindr/
  labels.py          # 28 классов, загрузка train.csv, сплиты
  data.py            # Dataset (DICOM/PNG + аугментации)
  model.py           # timm backbone + мульти-лейбл голова
  metrics.py         # AUROC / AP по классам и macro
  prepare.py         # подготовка данных, smoke-тест
  train.py           # цикл обучения (AMP, логгинг, best.pt)
  predict.py         # инференс по одному снимку
```

## Планы (roadmap)

- [x] Каркас + тренинг + инференс
- [ ] Загрузка данных VinDr-CXR
- [ ] Град-CAM-визуализации
- [ ] Пакет моделей: КТ-голова (RSNA ICH), МРТ-мозг (BraTS)
- [ ] EDA-отчёт, ROC-кривые, API-демо

## Лицензия

Код — **GPL-3.0-or-later** (copyleft: любые версии и улучшения обязаны оставаться открытыми). Данные VinDr-CXR — по лицензии VinGroup/PhysioNet (требуется регистрация).