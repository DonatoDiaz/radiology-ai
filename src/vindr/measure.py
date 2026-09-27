"""Quantitative measurements from detections (Phase 2, Task B).

Cardiothoracic ratio (CTR) — the classic PA-view screening measure:
    CTR = max transverse cardiac diameter / max internal thoracic diameter.
CTR > 0.50 on an upright PA radiograph is the standard radiographic
threshold for cardiomegaly (approximate; the formal threshold is defined
on a posteroanterior upright film at full inspiration).

The cardiac width is taken from the detector's `Cardiomegaly` box. The
internal thoracic diameter is estimated from the union of the lung-field
boxes (`Lung Opacity` / `Consolidation` / `Infiltration` / `Atelectasis`),
i.e. the widest inner rib-cage span those findings span. If no lung boxes
are present, the full image width is used as a conservative fallback.
"""

from __future__ import annotations

CARDIAC_CLASSES = ("Cardiomegaly",)
LUNG_CLASSES = (
    "Lung Opacity",
    "Consolidation",
    "Infiltration",
    "Atelectasis",
    "Pleural effusion",
    "Pulmonary fibrosis",
)
CTR_THRESHOLD = 0.50


def _union_width(boxes: list[list[float]]) -> float:
    if not boxes:
        return 0.0
    return max(b[2] for b in boxes) - min(b[0] for b in boxes)


def cardiothoracic_ratio(
    detections: list[dict],
    image_width: int,
    ctr_threshold: float = CTR_THRESHOLD,
) -> dict:
    """Estimate CTR from detector findings. Returns a dict with value + context."""
    cardiac = [d["bbox"] for d in detections if d["name"] in CARDIAC_CLASSES]
    lung = [d["bbox"] for d in detections if d["name"] in LUNG_CLASSES]

    if not cardiac:
        return {
            "ctr": None,
            "interpretation": "no cardiac box — measure manually",
            "threshold": ctr_threshold,
            "cardiomegaly_by_ctr": None,
        }

    heart_w = _union_width(cardiac)
    thorax_w = _union_width(lung) or float(image_width)
    if thorax_w <= 0:
        return {
            "ctr": None,
            "interpretation": "thoracic width unavailable",
            "threshold": ctr_threshold,
            "cardiomegaly_by_ctr": None,
        }

    ctr = heart_w / thorax_w
    enlarged = ctr > ctr_threshold
    return {
        "ctr": round(ctr, 3),
        "heart_width_px": round(heart_w, 1),
        "thorax_width_px": round(thorax_w, 1),
        "thorax_source": "lung boxes" if lung else "image width (fallback)",
        "threshold": ctr_threshold,
        "cardiomegaly_by_ctr": enlarged,
        "interpretation": (
            f"CTR {ctr:.2f} > {ctr_threshold:.2f} — meets radiographic criteria for cardiomegaly"
            if enlarged
            else f"CTR {ctr:.2f} <= {ctr_threshold:.2f} — below cardiomegaly threshold"
        ),
    }


def fluid_level_hint(detections: list[dict]) -> dict:
    """Qualitative flag for pleural effusion (meniscus / fluid level sign).

    A true fluid-level (hydropneumothorax) needs a horizontal air–fluid
    interface which requires lateral decubitus/ erect views; on a single
    PA/AP film we can only flag the presence of a pleural effusion box as a
    prompt to look for a meniscus.
    """
    eff = [d for d in detections if d["name"] == "Pleural effusion"]
    ptx = [d for d in detections if d["name"] == "Pneumothorax"]
    if eff and ptx:
        return {
            "flag": "effusion + pneumothorax",
            "hint": "consider hydropneumothorax; verify air–fluid level on erect/decubitus view",
        }
    if eff:
        return {
            "flag": "pleural effusion",
            "hint": "look for meniscus sign and check for an air–fluid level",
        }
    return {"flag": None, "hint": "no effusion-related box"}
