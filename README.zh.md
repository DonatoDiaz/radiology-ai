# Lung Radiology AI

用于胸放射线（胸部X光）肺部病变分诊诊断的 **AI 分类器**。作品集项目：医学 × AI。

> **阅读语言：** [English](README.en.md) | [Русский](README.ru.md) | [中文](README.zh.md)

## 项目范围

- **数据集：** VinDr-CXR（PhysioNet / Kaggle）— 15,000 张胸部X光片。
- **任务：** 多标签分类 → 病变评分。
- **模型 — 15 类（第1阶段完成）：** 11 个肺部类（肺不张、实变、浸润、磨玻璃影、胸腔积液/增厚、气胸、纤维化、结节/肿块等）+ 心血管/其他：心脏肥大、主动脉增宽、钙化、ILD + “No finding”。
- **骨干网络：** `tf_efficientnet_b0`（timm），512×512，AMP，AdamW + cosine LR。
- **指标：** macro AUROC / macro Average Precision + 各类别 ROC 曲线。

## 安装

```bash
uv sync
# 数据：从 Kaggle/PhysioNet 下载 VinDr-CXR，放入 data/vindr/
uv run vindr-prepare --data-dir data/vindr --smoke
```

## 训练

```bash
uv run vindr-train --config configs/train_full.yaml   # 第1阶段：15类
# 或手动：
uv run vindr-train \
  --data-dir data/vindr --images-dir data/vindr/images \
  --labels all --backbone tf_efficientnet_b0 \
  --image-size 512 --epochs 12 --batch-size 4 \
  --out-dir runs
```

最佳权重：`runs/<backbone>_<size>/best.pt`。

## EDA 报告

```bash
uv run vindr-eda --data-dir data/vindr --labels lung --out reports
# 图表：class_frequency.png、top_combinations.png、summary.md
```

## 推理与 Grad-CAM

```bash
uv run vindr-predict --ckpt runs/.../best.pt --image case_001.dcm --top-k 5 --lang zh
uv run python scripts/generate_cam.py --ckpt runs/.../best.pt --image case_001.dcm \
  --class-index 10 --out cam_overlay.png
```

`--lang en|ru|zh` 选择输出的病变名称语言。

### 如何解读结果

`predict` 输出每个病变的独立概率（0–1）（多标签，因此总和 ≠ 1）：

| 分数 | 解读 |
|------|------|
| > 0.9   | 确定的病变 |
| 0.5–0.9 | 可能的病变（建议查看原片） |
| 0.2–0.5 | 不确定 / 早期征象 |
| < 0.2   | 基本“无” |

合理性检查：
- **正常**片应得到 `No finding ≈ 1.0`，其余病变 ≈ 0；
- `reports/demo/roc_curves.png` 中的 ROC 曲线 — 曲线越靠近左上角，类别区分越强（第1阶段模型 macro-AUROC 0.950）；
- 稀有类别（如气胸，训练集仅 96 张）检测较弱 — 在类别不平衡下属预期现象。

### 指标（第1阶段，`runs/full_v1_b0_512/best.pt`）

验证集（n=1507）macro-AUROC **0.950**，全部15类 ≥ 0.85：

| 类别 | AUROC | 类别 | AUROC |
|------|-------|------|-------|
| No finding | 0.991 | Infiltration | 0.955 |
| Pleural effusion | 0.986 | Pulmonary fibrosis | 0.956 |
| Cardiomegaly | 0.985 | Lung Opacity | 0.952 |
| Aortic enlargement | 0.984 | Pleural thickening | 0.950 |
| ILD | 0.971 | Nodule/Mass | 0.941 |
| Consolidation | 0.964 | Calcification | 0.927 |
| Atelectasis | 0.917 | Other lesion | 0.913 |
| | | Pneumothorax | 0.850 |

### 病变术语表（放射学征象）

报告中“描述”部分会给出各标签对应的放射学征象描述：

| 标签 | 放射学征象 |
|------|-----------|
| **实变** | 密度增高并肺泡内含气腔隙完全闭塞、血管影消失；基底朝向胸膜的三角形致密影（大叶性肺炎） |
| **肺野致密影** | 磨玻璃影：密度增高但仍可见血管及小叶结构（肺泡内渗出、小叶间隔增厚） |
| **浸润** | 边界不清的致密影伴周围炎性反应；鉴别：肺炎 / 肺结核 |
| **肺不张** | 肺组织萎陷；可见空气支气管征→压迫性，未见→阻塞性 |
| **结节/肿块** | 病灶≤1cm，结节/肿块更大；分叶、边界不规则或生长→行CT，>~1cm时活检 |
| **胸腔积液** | 胸膜腔内液体（肋膈角变钝）；继发感染为脓胸 |
| **气胸** | 胸膜腔积气伴肺萎陷、无肺纹理透亮区 |
| **胸膜增厚** | 壁层胸膜增厚、斑片、与胸壁平行的线样致密影 |
| **肺纤维化** | 小叶间隔增厚、网状致密影、牵拉性支气管扩张；终末期蜂窝肺 |
| **心脏肥大** | 心影增大：心胸比 > 0.5 |
| **主动脉增宽** | 主动脉弓/升主动脉影增宽；明显增宽时行超声心动/CTA |
| **钙化** | 钙化影：主动脉弓、血管壁、胸膜斑 |
| **ILD** | 肺间质受累：小叶间隔增厚、支气管血管束旁线、网状改变 |

### 报告式输出

除 top-k 列表外，`predict` 会生成报告式文本（描述→结论→建议，含鉴别诊断提示），辅助专业人员以专科影像医师的水平解读影像：

```bash
uv run vindr-predict --ckpt runs/.../best.pt --image case_001.dcm --lang ru
```

## 网页演示

```bash
uv run uvicorn vindr.app:app --port 8000
# http://localhost:8000  — 上传图像 → 预测（可选语言：en/ru/zh）
# http://localhost:8000/predict/cam — Grad-CAM 叠加图（POST）
```

## 项目结构

```
configs/train_full.yaml    # 第1阶段配置（15类）
src/vindr/
  labels.py             # 28 类（完整 VinDr）+ 可用子集，数据集划分
  data.py               # Dataset（DICOM/PNG + 数据增强）
  model.py              # timm 骨干 + 多标签头
  metrics.py            # 各类别 AUROC / AP + macro
  i18n.py               # 标签翻译（en/ru/zh）
  prepare.py            # 数据准备、冒烟测试
  eda.py                # EDA 报告（类别频率、图像统计）
  gradcam.py            # 可视化“模型关注哪里”
  plots.py              # 各类别 ROC 曲线
  train.py              # 训练循环（--config、AMP、日志、best.pt）
  predict.py            # 单图推理 + 报告式输出
  report.py             # 医学知识库：术语表、鉴别规则、报告生成
  app.py                # FastAPI 网页演示
scripts/generate_cam.py # CLI：从权重生成 Grad-CAM 叠加图
```

## 路线图

- [x] 基础结构：train / predict / prepare
- [x] 肺部标签（`LUNG_LABELS`，mean AUROC 0.949）
- [x] Grad-CAM 可视化
- [x] EDA 报告
- [x] 各类别 ROC 曲线
- [x] 网页演示（FastAPI），支持三语输出（en/ru/zh）
- [x] **第1阶段完成：** VinDr-CXR 全部 15 个可用类别，macro-AUROC **0.950**（`runs/full_v1_b0_512/best.pt`）
- [ ] 第2阶段：病变检测器（YOLOv8s/RT-DETR）基于 `vindr-cxr-coco`
- [ ] 头部 CT（RSNA ICH）、脑 MRI（BraTS）—“多模态分诊”套餐
- [ ] Docker 化的 API 服务
  完整路线图 — [ROADMAP.zh.md](ROADMAP.zh.md)

## 许可证

代码 — **GPL-3.0-or-later**（copyleft：任何版本与改进必须保持开源）。VinDr-CXR 数据遵循 VinGroup/PhysioNet 许可（需要注册）。