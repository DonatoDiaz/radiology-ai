"""Quantitative CT measurements for head CT (Phase 3, Task B).

Everything here is computed on the HU volume — the same measurements a
radiologist makes with the ruler and the Hounsfield cursor:

* lesion density (HU) → blood vs. edema vs. infarct vs. tumor discrimination;
* hemorrhage volume in mL (from a mask or bbox);
* midline shift in mm (early sign of mass effect / herniation);
* Evans index → hydrocephalus;
* skull-fracture suspicion from the bone window.
"""

from __future__ import annotations

import numpy as np

from vindr.ct.volume import HU_ACUTE_BLOOD, HU_BLOOD, HU_EDEMA, HU_HEMORRHAGE_LATER, CTSeries

# Density bands used to name what the model is looking at
DENSITY_BANDS: tuple[tuple[str, float, float, str], ...] = (
    ("acute_hemorrhage", HU_ACUTE_BLOOD[0], HU_ACUTE_BLOOD[1], "острая кровь"),
    ("hemorrhage_later", HU_HEMORRHAGE_LATER[0], HU_HEMORRHAGE_LATER[1], "кровь поздней стадии"),
    ("blood_mixed", HU_BLOOD[0], HU_BLOOD[1], "кровь / смешанная плотность"),
    ("edema", HU_EDEMA[0], HU_EDEMA[1], "отёк"),
    ("fat", -150.0, -30.0, "жир"),
    ("csf", -20.0, 15.0, "ликвор"),
    ("brain_parenchyma", 20.0, 45.0, "мозговое вещество"),
    ("calcification", 90.0, 400.0, "кальцинат"),
    ("bone", 400.0, 3000.0, "кость"),
)


def classify_density(hu: float) -> dict:
    """Name a measured HU value using the standard density bands."""
    for name, lo, hi, ru in DENSITY_BANDS:
        if lo <= hu <= hi:
            return {"band": name, "label_ru": ru, "hu": round(float(hu), 1)}
    if hu < -150.0:
        return {"band": "air", "label_ru": "воздух", "hu": round(float(hu), 1)}
    return {"band": "unknown", "label_ru": "не определено", "hu": round(float(hu), 1)}


def lesion_density(volume: np.ndarray, mask: np.ndarray) -> dict:
    """Mean/median HU inside a lesion mask + its density band."""
    vox = np.asarray(volume)[np.asarray(mask).astype(bool)]
    if vox.size == 0:
        return {"mean_hu": None, "median_hu": None, "n_voxels": 0, "band": "empty", "label_ru": "нет данных"}
    mean, med = float(vox.mean()), float(np.median(vox))
    band = classify_density(med)
    return {
        "mean_hu": round(mean, 1),
        "median_hu": round(med, 1),
        "std_hu": round(float(vox.std()), 1),
        "n_voxels": int(vox.size),
        "band": band["band"],
        "label_ru": band["label_ru"],
        "is_blood_density": bool(HU_BLOOD[0] <= med <= HU_BLOOD[1]),
    }


def lesion_volume_ml(mask: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    """Volume of a mask in millilitres from (z, y, x) spacing in mm."""
    vox_ml = float(spacing[0]) * float(spacing[1]) * float(spacing[2]) / 1000.0
    n = int(np.asarray(mask).astype(bool).sum())
    ml = n * vox_ml
    return {
        "n_voxels": n,
        "volume_ml": round(ml, 2),
        "voxel_ml": round(vox_ml, 5),
        "grade": _hemorrhage_grade(ml),
    }


def _hemorrhage_grade(ml: float) -> str:
    """Common clinical size grading of intracranial bleeding."""
    if ml < 0.1:
        return "trace (punctate)"
    if ml < 10:
        return "small"
    if ml < 30:
        return "moderate"
    return "large / possible surgical consideration"


def bbox_mask(shape: tuple[int, int, int], bbox_zyx: tuple[float, float, float, float]) -> np.ndarray:
    """Boolean mask from a (z1, y1, z2, y2) box in slice coordinates."""
    z1, y1, z2, y2 = (round(float(v)) for v in bbox_zyx)
    mask = np.zeros(shape, dtype=bool)
    z1, z2 = sorted((max(0, min(shape[0] - 1, z1)), max(0, min(shape[0] - 1, z2))))
    y1, y2 = sorted((max(0, min(shape[1] - 1, y1)), max(0, min(shape[1] - 1, y2))))
    mask[z1 : z2 + 1, y1 : y2 + 1, :] = True
    return mask


def midline_shift_mm(series: CTSeries, threshold_hu: float = 20.0) -> dict:
    """Estimate midline shift by comparing left/right brain-tissue centroids.

    Shift > 5 mm is clinically significant (subfalcine herniation risk).
    """
    vol = series.volume
    _, _h, w = vol.shape
    cx = w / 2.0
    mid = int(cx)
    per_slice: list[float] = []
    for z in range(vol.shape[0]):
        sl = vol[z]
        tissue = sl > threshold_hu
        left = tissue[:, :mid]
        right = tissue[:, mid:]
        if left.sum() < 20 or right.sum() < 20:
            continue
        per_slice.append(float((_centroid_x(right) + cx) - _centroid_x(left)))
    if not per_slice:
        return {"max_shift_mm": None, "mean_shift_mm": None, "significant": None}
    arr = np.abs(np.asarray(per_slice))
    px_to_mm = float(series.spacing[2]) or 1.0
    return {
        "max_shift_mm": round(float(arr.max()) * px_to_mm, 1),
        "mean_shift_mm": round(float(arr.mean()) * px_to_mm, 1),
        "n_slices": len(per_slice),
        "significant": bool(float(arr.max()) * px_to_mm > 5.0),
    }


def _centroid_x(mask: np.ndarray) -> float:
    cols = np.nonzero(mask.any(axis=0))[0]
    if cols.size == 0:
        return 0.0
    weights = mask.sum(axis=0).astype(np.float64)
    return float((cols * weights[cols]).sum() / max(weights[cols].sum(), 1e-6))


def evans_index(volume: np.ndarray) -> dict:
    """Evans index for hydrocephalus: frontal horn width / inner skull width.

    > 0.30 is the classic adult threshold.
    """
    vol = np.asarray(volume)
    sl = vol[vol.shape[0] // 2] if vol.ndim == 3 else vol
    brain = sl > 0.0
    cols = np.nonzero(brain.any(axis=0))[0]
    if cols.size < 10:
        return {"evans_index": None, "hydrocephalus": None}
    inner_width_px = float(cols[-1] - cols[0] + 1)
    csf = (sl > -20.0) & (sl < 20.0)
    upper = csf[: max(1, csf.shape[0] // 2), :]
    horn_cols = np.nonzero(upper.any(axis=0))[0]
    if horn_cols.size < 3 or inner_width_px <= 0:
        return {"evans_index": None, "hydrocephalus": None}
    horns_px = float(horn_cols[-1] - horn_cols[0] + 1)
    evans = horns_px / inner_width_px
    return {
        "evans_index": round(evans, 3),
        "inner_width_px": round(inner_width_px, 1),
        "horns_width_px": round(horns_px, 1),
        "hydrocephalus": bool(evans > 0.30),
    }


def fracture_suspicion(series: CTSeries, min_slices: int = 2) -> dict:
    """Skull-vault fracture suspicion.

    The whole calvarium is above 700 HU, so counting dense voxels flagged every
    normal study; the real detector looks for a discontinuity in the vault ring
    and lives in :mod:`vindr.ct.fracture`. Kept here as the measure-module
    entry point.
    """
    from vindr.ct.fracture import skull_fractures

    return skull_fractures(series, min_slices=min_slices)


def summarize_study(series: CTSeries, lesion_mask: np.ndarray | None = None) -> dict:
    """Full quantitative summary for one study (what the report will print)."""
    out: dict = {
        "n_slices": series.n_slices,
        "shape": series.shape,
        "spacing_mm": tuple(round(float(v), 2) for v in series.spacing),
        "midline_shift": midline_shift_mm(series),
        "hydrocephalus": evans_index(series.volume),
        "fracture_suspicion": fracture_suspicion(series),
    }
    if lesion_mask is not None and np.asarray(lesion_mask).any():
        out["lesion_density"] = lesion_density(series.volume, lesion_mask)
        out["lesion_volume"] = lesion_volume_ml(lesion_mask, series.spacing)
    return out
