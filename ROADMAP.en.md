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

### Phase 1 — Full CXR (15 available labels) ✅ done
- **Task:** add the non-lung classes present in the data: Cardiomegaly, Aortic enlargement, Calcification, ILD (on top of the 11 lung labels).
- **Data:** Kaggle mirror of VinDr-CXR (`train.csv`, 15,000 images) — **15 label columns** available. The full 28-class set (rib/clavicle fractures, emphysema, mediastinal shift, etc.) requires the original VinDr-CXR from PhysioNet (registration/license) and will be added later via the same `train.py`.
- **Method:** `uv run vindr-train --config configs/train_full.yaml` (labels=all, 15 classes); class weights for rare classes.
- **Result:** `runs/full_v1_b0_512/best.pt` (epoch 7). macro-AUROC **0.950**, all 15 classes ≥ 0.85; weakest (rare): Pneumothorax 0.850 (96 samples), Other lesion 0.913.
- **done:** AUROC ≥ 0.85 across 15 classes; macro-average ≥ 0.90. — ✅ met
- **Budget:** ~4–6h on a GTX 1650 / ~2h on a T4 — confirmed (~3.5h, 12 epochs).

### Phase 2 — CXR detection and measurements 🔧 in progress (code ready)
- **Data already downloaded:** `vindr-cxr-coco` (VinDr-Ad, bbox) — ready anchors for detection.
- **Task A:** finding detector (YOLOv8s) — box-mAP50 ≥ 0.5; imgsz=1024 (objects are small: median ~0.2% of frame).
- **Code ready:** COCO→YOLO converter (`scripts/coco2yolo.py`), detection in CLI (`predict --detect`) and web (`/predict/det`), Colab notebook for cloud training.
- **Task B:** quantitative signs (cardiothoracic ratio, fluid level, "triangular shadow").
- **Training and validation happen at the very end** (see §6): all code first, then one final training + validation run.
- **done:** mAP50 ≥ 0.5; R² ≥ 0.9 for measurements on an annotated subset.

### Phase 3 — Head CT ("Brain" section) 🟡 code ready
- **Tasks:** stroke (ischemic/hemorrhagic), hematomas (epidural/subdural/SAH), tumors (meningioma, glioma, metastases), trauma (skull vault fractures), shift signs.
- **Data:** RSNA ICH, CQ500, Head-CT datasets (HF), BraTS (MRI, for tumors).
- **Method:** 2.5D (≥3 slices, semi-3D) already feasible on a GTX 1650; 3D→2.5D distillation for stronger nets.
- **Notes-semiotics:** differentiate hematomas by location (biconvex, confined by sutures = epidural; unconfined = subdural; along sulci = SAH) — encode into the report semiotics.
- **Code ready:** DICOM / 16-bit PNG / NIfTI loading to HU (`src/vindr/ct/volume.py`), RSNA preparation script (`scripts/prepare_head_ct.py`), slice + attention-pooled study models for the 5 RSNA subtypes plus `any` (`src/vindr/ct/model.py`), 2.5D datasets (`dataset.py`), training loop (`train.py`, `vindr-ct-train`), inference with JSON report (`predict.py`, `vindr-ct-predict`), HU measurements — lesion density/volume, midline shift, Evans index (`measure.py`).
- **Weakly-supervised labels (RSNA has slice labels, not pixel masks):** `src/vindr/ct/pseudo.py` mines pseudo-masks from a trained slice classifier — per-slice CAM → hi-res feature map → GrabCut seeded from the CAM core → acute-blood HU band (50–110, configurable; chronic collections near 25 HU need a different band) → per-study 3D mask, with a `pseudo_labels.json` log carrying the evidence (max/mean probability, z-range, masked slices) so training can filter by quality. Three guards keep the labels honest: the CAM must be *concentrated* (`cam_sharpness`, since a normalized CAM's maximum is 1.0 by construction and says nothing), the GrabCut seed is a fixed pixel budget rather than a value threshold (a threshold degenerates to "the whole slice" on a flat CAM), and the HU band is read on **native** voxels — resampling averages a 70 HU bleed against 30 HU brain and air, so a band computed on the network grid deletes the very hemorrhage it should confirm. Every native slice is examined via a native→grid z map, so a coarser grid cannot skip a thin bleed. `python -m vindr.ct.pseudo_mine --ckpt … --data-dir … --out-dir …` symlinks the images next to each `mask.npy`, producing a dataset `vindr-ct-seg-train` reads directly. These are pseudo-labels, not ground truth: the HU band and the lesion semiology are unvalidated heuristics, and ROADMAP Dice ≥ 0.7 is unproven without a real pixel-level dataset.
- **Localization code ready:** 2.5D U-Net for the hemorrhage mask (`src/vindr/ct/segmentation.py`, BCE + soft Dice, `dice_score`/`iou_score`), mask loader accepting `mask.nii.gz` / `mask.npy` / `masks/*.png` / `*_mask.png` next to the scan (`dataset.py`, `HeadCTSegDataset`), segmentation training loop (`train_seg.py`, `vindr-ct-seg-train`, `configs/train_head_ct_seg.yaml`), and 3D lesion extraction with the notes' semiology (`src/vindr/ct/lesions.py`): components merged across consecutive slices, biconvex-at-the-bone → epidural, crescent crossing the midline → subdural, serpiginous → subarachnoid, rounded inside the parenchyma → intraparenchymal, plus volumes and bounding boxes. Wired into inference via `predict --seg-ckpt` (optional `--save-mask`); `vindr-ct-predict --ckpt … --seg-ckpt …` reports subtypes and localization together. 94 tests cover the pipeline, including a synthetic classifier → pseudo-mine → `load_masked_studies` → `train_seg` end-to-end run.
- **Fractures (heuristic, `fracture.py`):** per-slice calvarium ring analysis finds a lucent
  suture-like gap (0.6-3.5 mm, midline guarded), a displaced step (each direction vs its mirror
  across the midline, so the defect no longer cancels itself), and free bone debris; findings are
  grouped across consecutive slices and matched to hematoma location. Reported as
  `measurements.skull_fractures`; a full bbox detector is Phase 6.
- **Not yet coded:** tumor models, a trained fracture detector.
- **Open data question:** localization needs *pixel* masks. The RSNA 16-bit PNG release carries slice-level labels only, so a mask dataset (e.g. a PhysioNet hemorrhage-segmentation set) must be chosen before the final run. The semiotics above are heuristic morphology descriptors, not a diagnosis.
- **Training and validation happen at the very end** (see §6): all code first, then one final training + validation run.
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

## 6. Model training and validation — at the very end 🔚 one single final run

**Principle:** write **all the code for every phase** first (all organ systems — from CXR to GI), and only when the whole program is ready, train and validate **all weights in one final pass**. This lets configs, architecture and pipelines change before the expensive training, so models are trained on the final version of the code.

**Final run order (once all code is done):**
1. **CXR classifier (Phase 1) — already trained** ✅ macro-AUROC **0.950** (`runs/full_v1_b0_512/best.pt`).
2. **CXR detector (Phase 2)** — cloud T4 (Colab), `notebooks/phase2_det_colab.ipynb`, imgsz=1024, batch 16, up to 40 epochs, patience 15. Target mAP50 ≥ 0.5.
3. **All CT/MRI models (Phases 3–8)** — trained as one package once their code is ready: head CT, kidney/mediastinum, sinuses, skeleton, GI. Heavy ones (2.5D/3D) on volunteers' T4/A100.
4. **Validation of all models** — run the metrics from §7 for every model; verify measurements (CTR R²) on an annotated subset.
5. **Protocol UAT** — clinician review (≥ 95% structurally correct reports).

Weights are not committed to the repository (`.gitignore`); they are stored separately and tied to a commit via release/README.

## 7. Success metrics (global)

| Stratum | Metric | Target |
|---|---|---|
| Classification | macro-AUROC | ≥ 0.90 (per phase) |
| Classification | mAP / partial-AUROC for rare classes | "best effort" + calibration |
| Detection | mAP50 | ≥ 0.5 |
| Segmentation | dice | ≥ 0.7 |
| Measurements (R²/abs error) | correlation with ground truth | R² ≥ 0.9 |
| Protocol | UAT by physician/feldsher (expert consensus) | ≥ 95% correctly structured reports |

## 8. Risks and mitigation

| Risk | Mitigation |
|---|---|
| ~4.8GB HTTP cap — some datasets cannot be downloaded in one shot | Per-file HF mirrors, range-based chunked download, split archives |
| 4GB VRAM on baseline machines | 2.5D / distillation; volunteers with T4/A100 cover heavy phases |
| Dataset licensing locks (PhysioNet) | Registration, license pages in the registry; prioritize open HF mirrors |
| Shortage of manually labeled data (sinuses, GI) | Start public; volunteer crowdsourced annotation |
| Modality drift across vendors | Normalization (windowing), domain adaptation, continue-finetune |

## 9. Getting started (volunteer quick start)

1. Clone the repo, `uv sync` (environment is packaged).
2. Pick an `open` task from the registry (§4) — e.g. `cxr_full_28` or `cxr_det_vindr`.
3. Write the task code (pipeline, config, metrics) — **do not start training**, see §6.
4. Run `vindr-eda` and `vindr-predict` on random samples, hand in code + a report.

Each phase is closed by a PR to this file (done checkbox ticked, weights in Releases).