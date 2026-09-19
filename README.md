# Lung Radiology AI

AI-классификатор рентгенограмм грудной клетки для триажной диагностики лёгочных патологий. Портфолио-проект: Медицина × ИИ.

**AI classifier for chest X-rays** to triage lung pathologies. Portfolio project: Medicine × AI.

**用于胸放射线肺部病变分诊诊断的AI分类器。** 作品集项目：医学 × AI。

---

## Read this in / Читать на / 阅读语言

- [**English**](README.en.md)
- [**Русский**](README.ru.md)
- [**中文**](README.zh.md)

---

**Quick start:**

```bash
uv sync
uv run vindr-predict --ckpt runs/.../best.pt --image case_001.dcm --top-k 5 --lang en
uv run uvicorn vindr.app:app --port 8000   # web demo, language select: en/ru/zh
```

Trained model — **mean AUROC 0.949** (`runs/lung_v1/best.pt`). Code license — **GPL-3.0-or-later**.