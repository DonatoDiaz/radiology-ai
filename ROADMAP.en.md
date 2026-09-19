# Roadmap: Full coverage of imaging diagnostics

**Mission:** open software that lets a mid-level healthcare worker **interpret any imaging study at specialist level without escalation** — cheaper than a salaried radiologist and equally reliable.

Model training is **crowdsourced across the community**: every contributor trains tasks on their own hardware (home GPU, Kaggle/Colab, rented instances). The roadmap only defines *what* and *how*; it does not depend on the power of any single machine.

> [English](ROADMAP.en.md) · [Русский](ROADMAP.ru.md) · [中文](ROADMAP.zh.md)

---

## 1. Vision and economics

- **Why cheaper:** a one-time software purchase and deployment vs. permanent staff costs; 24/7 operation; a uniform quality standard anywhere (FAP/primary care, field).
- **Replacement model:** "mid-level practitioner + AI = result of a specialist radiologist." AI does not escalate "upward" — it gives the practitioner knowledge and a verifiable interpretation on site.
- **What we ship:** an auto-protocol for every study (DESCRIPTION → CONCLUSION → RECOMMENDATIONS), an economics calculator "cost per study", a Docker service.
- **Openness:** code under GPL-3.0-or-later; every training run and every set of weights stays public so anyone can reproduce and improve.

## 2. Method: end-to-end pipeline per task

Each modality goes through the full cycle:

```
classification → localization (bbox/seg) → quantitative measurements → auto-protocol → evaluation
    (AUROC/AP)        (IoU/dice)             (R²/δ)                       (UAT)      AUROC/AP/dice/R²
```

Principles:
- **DICOM-first**, with a bridge to PNG/JPEG (respecting Windowing/ROI).
- **Task registry** — every modality is described by one uniform structure (see §4).
- **Evaluation on fixed val splits** with a reproducible seed — no "fitting to the leaderboard".
- **Threshold calibration** via Youden's J, multilingual output (en/ru/zh).

## 3. Community contribution (how it works)

Training is volunteer-driven. Every registry entry provides:

- **pickup package** — dataset links, config, run script, expected GPU budget (e.g. "4h on a T4");
- **target tiers** — "baseline / good / best" with expected metrics;
- **result handover** — weights on HF Hub or Google Drive, a report checklist (metrics, reproduction notes, screenshots);
- **review** — the task owner verifies the report and merges it into the registry.

This way a newcomer can start with a small task on a single GTX (resolution 256–512, batch 4–8), while an experienced contributor can take a CT task with 3D→2.5D distillation on 2×GPU.

## 4. Task registry (uniform structure)

```yaml
task:
  name: cxr_full_28              # unique name
  modality: cxr                  # cxr | ct_brain | ct_abdo | ct_sinus | msk | gi
  organ: chest
  task_type: clf                 # clf | det | seg | quant | seq
  labels: [ ... ]                # labels from labels.py
  dataset:
    source: kaggle|physionet|hf
    name: vinbigdata-cxr
    expected_size_gb: 3.9
    license: vingroup
  backbone: tf_efficientnet_b0
  image_size: 512
  train_budget: "T4: 6h -> baseline"
  status: done | in_progress | open
  owner: <github-user>
```

## 5. Phases

> Each phase = one modality/package. "done" = metrics reached on val and weights reviewed. Budget hints assume a single T4 (Kaggle/Colab).

### Phase 0 — Foundation ✅ (done)
- CXR classifier: 11 lung labels, EfficientNet-B0, 512px.
- mean AUROC **0.949** (weights `runs/lung_v1/best.pt`).
- Multilingual protocol (DESCRIPTION→CONCLUSION→RECOMMENDATIONS), Grad-CAM, web demo (FastAPI), README en/ru/zh.

### Phase 1 — Full CXR (28 labels) ⬜ open
- **Task:** train the non-lung classes: Cardiomegaly, Aortic enlargement, Mediastinal/tracheal shift, Rib/Clavicle/Other fracture, Calcification.
- **Data:** VinDr-CXR-full (same `train.csv`, all 28 columns).
- **Method:** same `train.py` with `labels=ALL_LABELS`; class weights for rare classes.
- **done:** AUROC ≥ 0.85 for all 28 classes; macro-average ≥ 0.90.
- **Budget:** ~8–12h on a T4.

### Phase 2 — CXR detection and measurements ⬜ open
- **Data already downloaded:** `vindr-cxr-coco` (VinDr-Ad, bbox) — ready anchors for detection.
- **Task A:** finding detector (YOLOv8s / RT-DETR) — box-mAP50 ≥ 0.5; support sub-seg labels.
- **Task B:** quantitative signs from the teaching notes:
  - cardiothoracic ratio (heart vs. chest width);
  - fluid level in sinus/pleura;
  - "triangular shadow" shape of consolidation (apex toward the hilum, base toward the pleura).
- **done:** mAP50 ≥ 0.5; R² ≥ 0.9 for measurements on an annotated subset.

### Phase 3 — Head CT ("Brain" section) ⬜ open
- **Tasks:** stroke (ischemic/hemorrhagic), hematomas (epidural/subdural/SAH), tumors (meningioma, glioma, metastases), trauma (skull vault fractures), shift signs.
- **Data:** RSNA ICH, CQ500, Head-CT datasets (HF), BraTS (MRI, for tumors).
- **Method:** 2.5D (≥3 slices, semi-3D) already feasible on a GTX 1650; 3D→2.5D distillation for stronger nets.
- **Notes-semiotics:** differentiate hematomas by location (biconvex, confined by sutures = epidural; unconfined = subdural; along sulci = SAH) — encode into the report semiotics.
- **done:** binary AUROC ≥ 0.90 (hemorrhage), ≥ 0.85 per subtype; localization dice ≥ 0.7.

### Phase 4 — Kidney/adrenal CT + mediastinum ⬜ open
- **Task A (from notes):** adrenal adenoma — attenuation < 10 HU (threshold); washout ≥ 50–60% by 10 min; cyst-like (no contrast uptake). This is "measurement by HU".
- **Task B:** chronic pyelonephritis (parenchymal thinning, decreased enhancement), nephrosclerosis ("shrunken kidney"), distorted calyces.
- **Task C:** mediastinum — thymoma (invasion stages), teratoma (inhomogeneous), intrathoracic goiter, cysts (thin-walled round), lymphoma (nodes < 1 cm = normal).
- **Data:** check KiTS / Kidney-Tumor-Seg (kidneys), find/collect mediastinal sets.
- **done:** adenoma AUC-per-lesion ≥ 0.85; kidney segmentation dice ≥ 0.85.

### Phase 5 — Paranasal sinus CT ⬜ open
- **Tasks:** sinusitis (mucosal thickening, fluid level), polypoid sinusitis, mucocele (dilated sinus with bone remodeling), fungal ball/mycetoma (hyperostosis/destruction), malignancy (ill-defined borders).
- **Data:** collect open "sinus CT" datasets or a de-identified extension.
- **done:** binary sinusitis AUROC ≥ 0.90; mucocele/topology segmentation dice ≥ 0.7.

### Phase 6 — Skeletal X-ray / fractures ⬜ open
- **Tasks:** clavicle, rib, long-bone/wrist fractures; box detection.
- **Data:** MURA (9,912, shoulder/elbow/wrist), GRAZPEDWRI (pediatric, 10.2k).
- **Method:** same detection pipeline as Phase 2; bone contrast is favorable.
- **done:** MURA AUROC ≥ 0.88; box-mAP50 ≥ 0.5.

### Phase 7 — GI (barium / irrigoscopy) 🔬 most ambitious
- **Tasks:** esophageal achalasia (grades 1–3 by barium retention), esophageal varices (contour deformation), diverticula (Zenker, bifurcation, epiphrenic), Kloiber's cups (acute obstruction), mucosal relief syndromes.
- **Method:** temporal series → video/VLM approach (fluoroscopy frame sequences). Heaviest in data/compute.
- **done:** binary achalasia AUROC ≥ 0.85; obstruction — det of fluid-level nodes.

### Phase 8 — Integration and economics ⬜ open
- Auto-protocols for all modalities (unified XML/JSON report format → print).
- Docker service: DICOM ingestion, batch inference, web UI.
- **Substitution calculator:** cost per study (electricity, amortization, licenses) vs. radiologist labor cost; break-even point by study volume.
- **"Triage across modalities" package:** CXR + head CT + sinuses in one deployment.

## 6. Success metrics (global)

| Stratum | Metric | Target |
|---|---|---|
| Classification | macro-AUROC | ≥ 0.90 (per phase) |
| Classification | mAP / partial-AUROC for rare classes | "best effort" + calibration |
| Detection | mAP50 | ≥ 0.5 |
| Segmentation | dice | ≥ 0.7 |
| Measurements (R²/abs error) | correlation with ground truth | R² ≥ 0.9 |
| Protocol | UAT by physician/feldsher (expert consensus) | ≥ 95% correctly structured reports |

## 7. Risks and mitigation

| Risk | Mitigation |
|---|---|
| ~4.8GB HTTP cap — some datasets cannot be downloaded in one shot | Per-file HF mirrors, range-based chunked download, split archives |
| 4GB VRAM on baseline machines | 2.5D / distillation; volunteers with T4/A100 cover heavy phases |
| Dataset licensing locks (PhysioNet) | Registration, license pages in the registry; prioritize open HF mirrors |
| Shortage of manually labeled data (sinuses, GI) | Start public; volunteer crowdsourced annotation |
| Modality drift across vendors | Normalization (windowing), domain adaptation, continue-finetune |

## 8. Getting started (volunteer quick start)

1. Clone the repo, `uv sync` (environment is packaged).
2. Pick an `open` task from the registry (§4) — e.g. `cxr_full_28` or `cxr_det_vindr`.
3. `uv run vindr-train --config configs/train.yaml` (+ `--out-dir runs/<task>`).
4. Run `vindr-eda` and `vindr-predict`, hand in weights + a report (checklist along the way).

Each phase is closed by a PR to this file (done checkbox ticked, weights in Releases).