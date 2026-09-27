"""Phase 4 — abdominal and thoracic organ measurements.

Every criterion here is one a radiologist applies by eye, written down so it
can be tested and reproduced. The functions split into two groups:

* **anchors** (:func:`body_mask`, :func:`lung_mask`, :func:`spine_mask`,
  :func:`aorta_mask`) find a structure from HU alone. These are only as good
  as their assumptions, so each returns what it assumed and refuses to guess
  when the study does not fit.
* **features** (:func:`aorta_findings`, :func:`adrenal_findings`,
  :func:`kidney_findings`, :func:`lymph_node_findings`,
  :func:`mediastinal_mass_findings`) measure a region that was already found,
  either by a trained model (Phase 4 data: KiTS, mediastinal sets) or by an
  anchor. They work on a mask and never guess where the organ is.

Nothing in this module is validated. Thresholds are the commonly published
cut-offs; a positive flag is a reason to look, not a diagnosis, and the
`note` fields say so in the report.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from vindr.ct.shapes import (
    equivalent_diameter_mm,
    extent_mm,
    feret_diameter_mm,
    hydraulic_thickness_mm,
    perimeter_mm,
    short_axis_mm,
    volume_ml,
)

CONN3 = np.ones((3, 3, 3), bool)  # 26-connectivity: a lumen may drift diagonally

# published cut-offs, gathered here so they can be changed in one place
ANEURYSM_ABDOMEN_MM = 30.0  # aorta at or above this needs a surgeon
ANEURYSM_URGENT_MM = 40.0
ADENOMA_LIPID_HU = 10.0  # unenhanced mean below this is adenoma until proven otherwise
WASHOUT_ABSOLUTE_PCT = 60.0  # (plain - portal) / plain
WASHOUT_RELATIVE_PCT = 40.0  # (portal - delayed) / portal
ADENOMA_BENIGN_SIZE_MM = 40.0  # larger than this and malignancy dominates
NODE_NORMAL_MM = 10.0  # short axis; size alone never decides
NODE_BORDERLINE_MM = 15.0
KIDNEY_SHRUNK_AXIS_MM = 80.0
KIDNEY_CORTEX_THIN_MM = 7.0
KIDNEY_ENHANCEMENT_ASYMMETRY_HU = 15.0
CYST_WALL_MAX_MM = 2.0
FAT_HU = -30.0  # macroscopic fat
CALC_HU = 200.0
# simple fluid sits in a band; a one-sided "< 20 HU" test also catches fat,
# which is how a fat mass gets mislabelled as a cyst
FLUID_HU_RANGE = (-20.0, 20.0)

NOT_CLAIM = "не диагноз: критерий для проверки врачом"


def _largest(mask3d: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected component."""
    lab, n = ndi.label(mask3d, structure=CONN3)
    if n <= 1:
        return mask3d.astype(bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == int(sizes.argmax())


def _keep_major_components(mask3d: np.ndarray, min_frac: float = 0.25) -> tuple[np.ndarray, int]:
    """Keep every component that is a real part of the structure.

    An intimal flap is hypodense and falls out of an enhancement window, which
    splits a vessel in two. Keeping only the biggest piece would silently throw
    away half an aorta and still report a plausible caliber, so any component
    within `min_frac` of the largest is kept.
    """
    lab, n = ndi.label(mask3d, structure=CONN3)
    if n <= 1:
        return mask3d.astype(bool), n
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    keep = np.flatnonzero(sizes >= min_frac * sizes.max())
    return np.isin(lab, keep), int(keep.size)


_bbox_extent_mm = extent_mm


def _longest_axis_mm(mask3d: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Largest single dimension of a mask, in mm."""
    return float(max(_bbox_extent_mm(mask3d, spacing)))


# --------------------------------------------------------------------------- #
# anchors: find a structure from HU alone
# --------------------------------------------------------------------------- #


_perimeter_mm = perimeter_mm


def body_mask(volume: np.ndarray, hu_min: float = -300.0) -> np.ndarray:
    """Body outline, with the lungs filled in as interior holes."""
    soft = volume > hu_min
    filled = np.stack([ndi.binary_fill_holes(soft[z]) for z in range(soft.shape[0])])
    return filled


def lung_mask(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    hu_max: float = -400.0,
    min_ml: float = 50.0,
) -> np.ndarray:
    """Air inside the body, i.e. the lungs.

    The outside air is excluded by intersecting with the filled body mask,
    which is what separates lungs from the air around the patient.
    """
    air = volume < hu_max
    inside = air & body_mask(volume)
    lab, n = ndi.label(inside, structure=CONN3)
    if n == 0:
        return inside
    sizes = np.bincount(lab.ravel(), weights=np.repeat(np.prod([float(s) for s in spacing]) / 1000.0, lab.size))
    sizes[0] = 0.0
    keep = {int(i) for i in np.flatnonzero(sizes >= min_ml)}
    return np.isin(lab, list(keep)) if keep else np.zeros_like(inside)


def spine_mask(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    hu_min: float = 250.0,
    half_width_mm: float = 45.0,
) -> np.ndarray:
    """Vertebral column: dense, and within `half_width_mm` of the midline.

    Both conditions are needed. A HU cut-off alone also catches the ribs, the
    skull base and the calcified aorta; the width cut-off alone catches
    everything soft in the middle of the body.
    """
    dense = volume > hu_min
    _, _, x = np.indices(volume.shape)
    width = int(max(1, round(half_width_mm / float(spacing[2]))))
    midline = (volume.shape[2] - 1) / 2.0
    return dense & (np.abs(x - midline) <= width)


def aorta_mask(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    hu_range: tuple[float, float] = (100.0, 600.0),
    search_mm: float = 40.0,
    max_diameter_mm: float = 60.0,
) -> tuple[np.ndarray, dict]:
    """Contrast-filled aorta: the HU window, next to the spine.

    Needs a contrast-enhanced study. Without one the aorta sits at
    30-50 HU, outside the window, and this returns an empty mask and says so
    rather than grabbing a vein.

    Known limitation: in the portal or arterial phase a kidney and the
    enhancing bowel also fall inside the window, and a kidney can sit close
    enough to the spine to pass. The width guard catches the gross cases -- a
    40 mm kidney is wider than any aorta -- but a small one can still be
    mistaken. This locator is for the caliber and the flap on a study where the
    aorta is obvious; anything finer needs a Phase 4 model.
    """
    lo, hi = hu_range
    sp = [float(s) for s in spacing]
    _, y, x = np.indices(volume.shape)
    spine = spine_mask(volume, spacing)
    if not spine.any():
        return np.zeros(volume.shape, bool), {"ok": False, "reason": "нет костного позвоночника"}
    _, yc, _ = [float(v) for v in ndi.center_of_mass(spine)]
    dist = np.sqrt(((y - yc) * sp[1]) ** 2 + ((x - volume.shape[2] / 2.0) * sp[2]) ** 2)
    near_spine = dist <= search_mm
    pool = (volume >= lo) & (volume <= hi) & near_spine
    # a hypodense intimal flap splits the contrast column; close the 1-voxel
    # gaps it makes so the vessel stays one object
    pool = np.stack([ndi.binary_closing(pool[z], np.ones((3, 3), bool)) for z in range(pool.shape[0])])
    pool, n_parts = _keep_major_components(pool)
    if not pool.any():
        return pool, {
            "ok": False,
            "reason": f"нет мягкой ткани в окне {lo:.0f}..{hi:.0f} HU рядом с позвоночником — вероятно, исследование без контраста",
        }
    widest = max(
        feret_diameter_mm(pool[z], (sp[1], sp[2])) for z in range(pool.shape[0]) if pool[z].any()
    )
    if widest > max_diameter_mm:
        # the HU window alone also fits the liver, the portal vein and muscle;
        # the aorta is not 60 mm across unless it is an aneurysm
        return (
            np.zeros(volume.shape, bool),
            {"ok": False, "reason": f"область в окне {lo:.0f}..{hi:.0f} HU шириной {widest:.0f} мм — это не аорта"},
        )
    hu = float(np.median(volume[pool]))
    return pool, {
        "ok": True,
        "assumed": "контрастированная аорта",
        "median_hu": round(hu, 1),
        "n_parts_kept": n_parts,
        "note": "просвет с закрытыми мелкими разрывами; крупный тромб не закрывается, "
        "тогда диаметр просвета занижен относительно наружного",
    }


def contrast_hu_guess(volume: np.ndarray, mask: np.ndarray) -> float | None:
    """Median HU inside a located vessel, to confirm the study is enhanced."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    return float(np.median(np.asarray(volume)[m]))


# --------------------------------------------------------------------------- #
# aorta
# --------------------------------------------------------------------------- #


def aorta_findings(
    mask3d: np.ndarray,
    spacing: tuple[float, float, float],
    hu_volume: np.ndarray | None = None,
) -> dict:
    """Aortic caliber and a possible intimal flap.

    A flap splits the enhanced lumen into two, so the rule is geometric: take
    the lumen, cut it at the midpoint of its own HU range, and see whether a
    large part of the cross-section disappears. That needs `hu_volume`; without
    it only the caliber is reported.

    The mask is the contrast-filled lumen, so the caliber is a lower bound on
    the outer diameter wherever there is mural thrombus.
    """
    mask3d = np.asarray(mask3d).astype(bool)
    out: dict = {
        "volume_ml": round(volume_ml(mask3d, spacing), 2),
        "max_diameter_mm": None,
        "diameter_by_slice_mm": [],
        "flags": [],
    }
    if not mask3d.any():
        out["flags"] = ["аорта не найдена"]
        return out
    sp_xy = (float(spacing[1]), float(spacing[2]))
    diam_by_z: list[dict] = []
    for z in range(mask3d.shape[0]):
        sl = mask3d[z]
        if sl.sum() < 6:
            continue
        area = float(sl.sum()) * sp_xy[0] * sp_xy[1]
        feret = feret_diameter_mm(sl, sp_xy)
        diam_by_z.append(
            {
                "slice": z,
                "feret_mm": round(feret, 1),
                "equivalent_mm": round(equivalent_diameter_mm(area), 1),
            }
        )
    if not diam_by_z:
        out["flags"] = ["аорта слишком тонкая для измерения"]
        return out
    out["diameter_by_slice_mm"] = diam_by_z
    worst = max(diam_by_z, key=lambda d: d["feret_mm"])
    out["max_diameter_mm"] = worst["feret_mm"]
    out["worst_slice"] = worst["slice"]
    if worst["feret_mm"] >= ANEURYSM_URGENT_MM:
        out["flags"].append(f"диаметр {worst['feret_mm']} мм — крупная аневризма, показана консультация")
    elif worst["feret_mm"] >= ANEURYSM_ABDOMEN_MM:
        out["flags"].append(f"диаметр {worst['feret_mm']} мм — аневризма брюшной аорты")
    if hu_volume is not None:
        out["flap"] = _intimal_flap(mask3d, np.asarray(hu_volume), spacing)
        if out["flap"]["suspected"]:
            out["flags"].append("возможна интимальная заплатка (диссекция) по двум просветам")
    out["note"] = "просвет в маске: при тромбе истинный внешний диаметр больше. " + NOT_CLAIM
    return out


def _intimal_flap(
    mask3d: np.ndarray,
    hu_volume: np.ndarray,
    spacing: tuple[float, float, float],
    min_area_mm2: float = 30.0,
) -> dict:
    """Look for a septum that splits the lumen in two, slice by slice.

    The HU range comes from the 1st and 99th percentiles of the lumen, not the
    5th and 95th: a real intimal flap is a thin strip and can be well under
    1% of the cross-section, which a robust range reads as flat and misses
    entirely. To pay for the sensitivity, the split has to appear on two
    consecutive slices and both halves have to be at least `min_area_mm2`, so
    a noisy voxel in one slice cannot call a dissection.
    """
    px_area = float(spacing[1]) * float(spacing[2])
    min_px = max(1, round(min_area_mm2 / px_area))
    per_slice: list[dict] = []
    for z in range(mask3d.shape[0]):
        lumen = mask3d[z]
        if lumen.sum() < 20:
            continue
        vals = hu_volume[z][lumen]
        lo, hi = float(np.percentile(vals, 1)), float(np.percentile(vals, 99))
        if hi - lo < 40.0:  # flat enhancement: nothing to split
            continue
        lab, n = ndi.label(lumen & (hu_volume[z] >= (lo + hi) / 2.0), structure=np.ones((3, 3), bool))
        if n < 2:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        if int((sizes >= min_px).sum()) >= 2:
            per_slice.append({"slice": z, "n_lumens": int(n)})
    # a flap is a structure, not a coincidence: require consecutive slices
    runs = 0
    best = 0
    prev = None
    for item in per_slice:
        runs = runs + 1 if prev is not None and item["slice"] == prev + 1 else 1
        best = max(best, runs)
        prev = item["slice"]
    return {
        "suspected": best >= 2,
        "max_consecutive_slices": int(best),
        "slices": per_slice[:10],
        "note": "две крупные области в просвете после порога по HU на двух и более соседних срезах; "
        "нужен контраст и опытный просмотр. " + NOT_CLAIM,
    }


# --------------------------------------------------------------------------- #
# adrenal
# --------------------------------------------------------------------------- #


def washout(
    hu: dict[str, float | None],
    kind: str = "auto",
) -> dict | None:
    """Contrast washout, absolute or relative.

    Absolute: (plain - portal) / plain, the unenhanced study against the
    portal phase, published cut-off 60%.
    Relative: (portal - delayed) / portal, portal against a 10-minute film,
    cut-off 40%. Both are used in practice, and both are meaningless when the
    denominator is at or below the noise floor of a soft tissue region, which
    is why a non-positive result is returned as None rather than a number.
    """
    plain, portal, delayed = (hu.get(k) for k in ("plain", "portal", "delayed"))
    if plain is not None and portal is not None and plain > 20.0:
        value = (plain - portal) / plain * 100.0
        if kind in ("auto", "absolute"):
            return {
                "kind": "absolute",
                "pct": round(value, 1),
                "passes_adenoma_cutoff": value >= WASHOUT_ABSOLUTE_PCT,
                "note": "(plain - portal) / plain; валидно только при plain > 20 HU",
            }
    if portal is not None and delayed is not None and portal > 20.0:
        value = (portal - delayed) / portal * 100.0
        if kind in ("auto", "relative"):
            return {
                "kind": "relative",
                "pct": round(value, 1),
                "passes_adenoma_cutoff": value >= WASHOUT_RELATIVE_PCT,
                "note": "(portal - delayed) / portal; валидно только при portal > 20 HU",
            }
    return None


def adrenal_findings(
    mask3d: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    hu_by_phase: dict[str, float] | None = None,
) -> dict:
    """Adrenal lesion: attenuation, size, and washout if phases are given.

    `hu_by_phase` is the mean HU of the same lesion on other series, keyed
    `plain` / `portal` / `delayed`. Under 10 HU unenhanced is an adenoma
    until proven otherwise, which is the single most useful rule here; it does
    not apply to a lesion that is entirely cystic or that is large, where
    malignancy moves to the front regardless of the HU.
    """
    mask3d = np.asarray(mask3d).astype(bool)
    if not mask3d.any():
        return {"ok": False, "reason": "пустая маска"}
    vals = np.asarray(volume)[mask3d].astype(float)
    long_axis = _longest_axis_mm(mask3d, spacing)
    median_hu = float(np.median(vals))
    std_hu = float(vals.std())
    out = {
        "ok": True,
        "volume_ml": round(volume_ml(mask3d, spacing), 2),
        "long_axis_mm": round(long_axis, 1),
        "short_axis_mm": round(short_axis_mm(mask3d, spacing), 1),
        "mean_hu": round(float(vals.mean()), 1),
        "median_hu": round(median_hu, 1),
        "hu_std": round(std_hu, 1),
        "flags": [],
    }
    lipid_rich = median_hu < ADENOMA_LIPID_HU
    out["lipid_rich"] = lipid_rich
    if hu_by_phase:
        w = washout(hu_by_phase)
        if w is not None:
            out["washout"] = w
        else:
            out["washout"] = None
            out["flags"].append("washout не рассчитан: нет фаз или фон HU слишком низкий")
    if lipid_rich:
        out["flags"].append(f"неконтрастное HU {median_hu:.0f} < 10 — типично для аденомы")
    if hu_by_phase and out.get("washout") and out["washout"]["passes_adenoma_cutoff"]:
        out["flags"].append("washout в диапазоне аденомы")
    if long_axis > ADENOMA_BENIGN_SIZE_MM:
        out["flags"].append(f"размер {long_axis:.0f} мм > 40 — доброкачественная аденома менее вероятна")
    if std_hu > 25.0:
        out["flags"].append("HU неоднородны — простая аденома менее вероятна")
    out["note"] = "аттенюация и washout — основа диагностики аденомы; размер и неоднородность против неё. " + NOT_CLAIM
    return out


# --------------------------------------------------------------------------- #
# kidney
# --------------------------------------------------------------------------- #


def _cortex_thickness_mm(mask2d: np.ndarray, spacing_yx: tuple[float, float]) -> float:
    """Cortex thickness, the 2A/P estimate; see :func:`shapes.hydraulic_thickness_mm`."""
    return hydraulic_thickness_mm(mask2d, spacing_yx)


def kidney_findings(
    mask3d: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    other_mask3d: np.ndarray | None = None,
    portal_hu: float | None = None,
    other_portal_hu: float | None = None,
) -> dict:
    """Kidney size, cortex thickness, and the chronic-disease signs.

    The two findings that need no enhancement data are measured from the mask:
    a long axis under 80 mm with the other kidney normal is a shrunken
    kidney, and cortex thinner than 7 mm — or thinner than half the other
    side — suggests chronic pyelonephritis. A one-sided fall in enhancement
    is the third sign, and it does need portal-phase HU.
    """
    mask3d = np.asarray(mask3d).astype(bool)
    if not mask3d.any():
        return {"ok": False, "reason": "пустая маска"}
    sp_xy = (float(spacing[1]), float(spacing[2]))
    axis = _longest_axis_mm(mask3d, spacing)
    thick = [round(_cortex_thickness_mm(mask3d[z], sp_xy), 1) for z in range(mask3d.shape[0]) if mask3d[z].sum() >= 8]
    med_thick = float(np.median(thick)) if thick else 0.0
    out = {
        "ok": True,
        "volume_ml": round(volume_ml(mask3d, spacing), 2),
        "long_axis_mm": round(axis, 1),
        "cortex_thickness_mm": round(med_thick, 1),
        "flags": [],
    }
    if other_mask3d is not None and np.asarray(other_mask3d).any():
        other = np.asarray(other_mask3d).astype(bool)
        other_axis = _longest_axis_mm(other, spacing)
        other_thick_list = [
            round(_cortex_thickness_mm(other[z], sp_xy), 1) for z in range(other.shape[0]) if other[z].sum() >= 8
        ]
        other_thick = float(np.median(other_thick_list)) if other_thick_list else 0.0
        other_vol = volume_ml(other, spacing)
        out["contralateral"] = {
            "long_axis_mm": round(other_axis, 1),
            "cortex_thickness_mm": round(other_thick, 1),
            "volume_ml": round(other_vol, 2),
        }
        if axis < 0.75 * other_axis or (other_vol > 0 and volume_ml(mask3d, spacing) < 0.5 * other_vol):
            out["flags"].append(
                f"длина {axis:.0f} мм против {other_axis:.0f} мм на другой стороне — подозрение на нефросклероз"
            )
        if other_thick > 0 and med_thick < 0.5 * other_thick:
            out["flags"].append(f"кора {med_thick:.1f} мм против {other_thick:.1f} мм — истончение, хронический пиелонефрит")
    if med_thick < KIDNEY_CORTEX_THIN_MM:
        out["flags"].append(f"кора {med_thick:.1f} мм < {KIDNEY_CORTEX_THIN_MM} мм — истончение")
    if axis < KIDNEY_SHRUNK_AXIS_MM:
        out["flags"].append(f"длина {axis:.0f} мм < {KIDNEY_SHRUNK_AXIS_MM} мм — уменьшенная почка")
    if portal_hu is not None and other_portal_hu is not None:
        delta = abs(float(portal_hu) - float(other_portal_hu))
        out["enhancement_delta_hu"] = round(delta, 1)
        if delta > KIDNEY_ENHANCEMENT_ASYMMETRY_HU:
            out["flags"].append(f"разница усиления {delta:.0f} HU между сторонами — сниженное усиление")
    hyd = _hydronephrosis_proxy(mask3d, np.asarray(volume))
    out["collecting_system_fluid_fraction"] = round(hyd, 3)
    if hyd > 0.15:
        out["flags"].append("жидкость в центре почечной тени — возможна гидронефроз")
    out["note"] = "толщина коры оценена как 2A/P по заливке почки; это оценка, а не измерение. " + NOT_CLAIM
    return out


def _hydronephrosis_proxy(
    mask3d: np.ndarray,
    volume: np.ndarray,
    fluid_range: tuple[float, float] = FLUID_HU_RANGE,
) -> float:
    """Fraction of the central half of the kidney that sits at fluid density."""
    mask3d = mask3d.astype(bool)
    total, fluid = 0, 0
    for z in range(mask3d.shape[0]):
        sl = mask3d[z]
        if sl.sum() < 8:
            continue
        yy = np.nonzero(sl.any(1))[0]
        xx = np.nonzero(sl.any(0))[0]
        cy = round((yy.min() + yy.max()) / 2.0)
        cx = round((xx.min() + xx.max()) / 2.0)
        r = max(1, int(0.25 * min(yy.max() - yy.min(), xx.max() - xx.min())))
        core = np.zeros_like(sl)
        core[max(0, cy - r) : cy + r + 1, max(0, cx - r) : cx + r + 1] = True
        sel = sl & core
        n = int(sel.sum())
        if n == 0:
            continue
        total += n
        fluid += int(((volume[z][sel] >= fluid_range[0]) & (volume[z][sel] <= fluid_range[1])).sum())
    return float(fluid / total) if total else 0.0


# --------------------------------------------------------------------------- #
# mediastinum
# --------------------------------------------------------------------------- #


def lymph_node_findings(
    mask3d: np.ndarray,
    spacing: tuple[float, float, float],
    min_voxels: int = 8,
) -> dict:
    """Per-node short axis against the 10/15 mm rules.

    Size is a screening rule, not a diagnosis: an 11 mm node can be reactive
    and a 9 mm one can be malignant, so PET and follow-up decide. The rule
    that survives practice is the short axis in the axial plane, which is why
    the two in-plane extents are used rather than the 3D bounding box.
    """
    mask3d = np.asarray(mask3d).astype(bool)
    lab, n = ndi.label(mask3d, structure=CONN3)
    nodes: list[dict] = []
    for i in range(1, n + 1):
        comp = lab == i
        if int(comp.sum()) < min_voxels:
            continue
        dz, dy, dx = _bbox_extent_mm(comp, spacing)
        short = min(dy, dx)
        grade = "normal" if short < NODE_NORMAL_MM else "borderline" if short < NODE_BORDERLINE_MM else "pathologic"
        nodes.append(
            {
                "id": i,
                "short_axis_mm": round(short, 1),
                "long_axis_mm": round(max(dy, dx), 1),
                "craniocaudal_mm": round(dz, 1),
                "volume_ml": round(volume_ml(comp, spacing), 2),
                "size_grade": grade,
            }
        )
    nodes.sort(key=lambda d: -d["short_axis_mm"])
    return {
        "n_nodes": len(nodes),
        "n_pathologic": sum(1 for d in nodes if d["size_grade"] == "pathologic"),
        "nodes": nodes[:20],
        "note": f"<{NODE_NORMAL_MM:.0f} мм норма, {NODE_NORMAL_MM:.0f}-{NODE_BORDERLINE_MM:.0f} мм неопределённо, "
        f">{NODE_BORDERLINE_MM:.0f} мм подозрительно; размер не решает. " + NOT_CLAIM,
    }


def mediastinal_mass_findings(
    mask3d: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    cyst_cutoff_hu: float = FLUID_HU_RANGE[1],
) -> dict:
    """Describe a mediastinal mass by its contents.

    Three combinations are recognisable from a mask plus the HU inside it:
    a thin-walled fluid mass is a cyst, macroscopic fat plus a calcific focus
    is a teratoma, and marked HU heterogeneity over a large soft-tissue mass
    is what a thymoma looks like. Vessel encasement is reported when an
    aorta mask is supplied.
    """
    mask3d = np.asarray(mask3d).astype(bool)
    if not mask3d.any():
        return {"ok": False, "reason": "пустая маска"}
    vals = np.asarray(volume)[mask3d].astype(float)
    dz, dy, dx = _bbox_extent_mm(mask3d, spacing)
    fat = float((vals < FAT_HU).mean())
    calc = _calcification(mask3d, np.asarray(volume))
    fluid = float(((vals >= FLUID_HU_RANGE[0]) & (vals <= cyst_cutoff_hu)).mean())
    std_hu = float(vals.std())
    out = {
        "ok": True,
        "volume_ml": round(volume_ml(mask3d, spacing), 2),
        "extent_mm": [round(v, 1) for v in (dz, dy, dx)],
        "mean_hu": round(float(vals.mean()), 1),
        "hu_std": round(std_hu, 1),
        "fat_fraction": round(fat, 3),
        "fluid_fraction": round(fluid, 3),
        "calcification_voxels": int(calc),
        "suggests": None,
        "flags": [],
    }
    # "suggests" is the machine-readable half: the flags below are prose for
    # the report, and prose is a bad thing to parse
    if fat > 0.2 and calc > 0:
        out["suggests"] = "teratoma"
        out["flags"].append("жировая ткань + обызвествление — характерно для тератомы")
    elif fluid > 0.9 and fat < 0.05 and std_hu < 15.0:
        out["suggests"] = "cyst"
        out["flags"].append("почти вся масса имеет плотность простой жидкости — кистозная структура")
    elif fat > 0.2:
        out["suggests"] = "fat_mass"
        out["flags"].append("жир без обызвествления — типичнее для липомы, чем для тератомы")
    elif std_hu > 35.0:
        out["suggests"] = "thymoma"
        out["flags"].append("выраженная неоднородность HU — стоит думать о тимоме")
    else:
        out["suggests"] = None
    if fat > 0.2:
        out["flags"].append(f"{fat * 100:.0f}% массы ниже {FAT_HU:.0f} HU — макроскопический жир")
    out["note"] = "классификация по содержимому; для лимфомы и thymoma нужна оценка инкапсуляции сосудов. " + NOT_CLAIM
    return out


def _calcification(mask3d: np.ndarray, volume: np.ndarray, min_voxels: int = 2) -> int:
    """Voxels in calcific-density clusters, a teratoma's other ingredient."""
    calc = mask3d & (volume >= CALC_HU)
    lab, n = ndi.label(calc, structure=CONN3)
    if n == 0:
        return 0
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return int(sizes[sizes >= min_voxels].sum())


def vessel_encasement(mass_mask: np.ndarray, vessel_mask: np.ndarray) -> dict:
    """Does the mass wrap a vessel? A lymphoma sign, hard to fake."""
    m, v = np.asarray(mass_mask).astype(bool), np.asarray(vessel_mask).astype(bool)
    if not m.any() or not v.any():
        return {"ok": False, "reason": "пустая маска или сосуд"}
    ring = ndi.binary_dilation(m, np.ones((3, 3, 3), bool), iterations=2) & ~m
    inside = int((ring & v).sum())
    total = int(v.sum())
    return {
        "ok": True,
        "vessel_voxels_surrounded": inside,
        "vessel_fraction_surrounded": round(inside / total, 3) if total else 0.0,
        "note": "инкапсуляция сосуда массой; при >0.5 доли — вероятна лимфома. " + NOT_CLAIM,
    }


# --------------------------------------------------------------------------- #
# report entry point
# --------------------------------------------------------------------------- #


def organ_report(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    kidney: np.ndarray | None = None,
    kidney_other: np.ndarray | None = None,
    adrenal: np.ndarray | None = None,
    nodes: np.ndarray | None = None,
    mass: np.ndarray | None = None,
    hu_by_phase: dict[str, float] | None = None,
    portal_hu: float | None = None,
    other_portal_hu: float | None = None,
    locate_aorta: bool = True,
) -> dict:
    """Phase 4 block for the report.

    The aorta is located from HU, because the contrast pool next to the spine
    is findable without a model. The kidneys, adrenals, nodes and masses are
    measured only if a mask is handed in -- from a Phase 4 model once KiTS and
    the mediastinal sets are available -- and each is reported as skipped
    rather than guessed, so a missing mask is visible in the output instead of
    silently producing a normal-looking result.
    """
    volume = np.asarray(volume)
    out: dict = {"phase": 4, "note": NOT_CLAIM}
    skipped: list[str] = []

    if locate_aorta:
        mask, info = aorta_mask(volume, spacing)
        if not info["ok"]:
            out["aorta"] = {"ok": False, "reason": info["reason"]}
        else:
            out["aorta"] = aorta_findings(mask, spacing, hu_volume=volume)
            out["aorta"]["located_by"] = info
    else:
        skipped.append("aorta")

    if kidney is not None:
        out["kidney"] = kidney_findings(
            kidney,
            volume,
            spacing,
            other_mask3d=kidney_other,
            portal_hu=portal_hu,
            other_portal_hu=other_portal_hu,
        )
    else:
        skipped.append("kidney")

    if adrenal is not None:
        out["adrenal"] = adrenal_findings(adrenal, volume, spacing, hu_by_phase=hu_by_phase)
    else:
        skipped.append("adrenal")

    if nodes is not None:
        out["lymph_nodes"] = lymph_node_findings(nodes, spacing)
    else:
        skipped.append("lymph_nodes")

    if mass is not None:
        out["mediastinal_mass"] = mediastinal_mass_findings(mass, volume, spacing)
    else:
        skipped.append("mediastinal_mass")

    if skipped:
        out["skipped"] = skipped
        out["skip_reason"] = "нужна маска: Phase 4 ещё без обученной модели сегментации (см. ROADMAP)"
    return out
