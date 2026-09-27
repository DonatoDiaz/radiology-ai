"""Phase 5 — paranasal sinus CT.

The loader keeps no orientation, so nothing here decides "front" from "back".
Every criterion is built to work without it:

* a sinus is a cavity **enclosed by bone**, and a cavity is a region that does
  not reach the edge of the image, so the nasal airway and the air around the
  patient drop out while the sinuses and the mastoid air cells stay;
* **mucosal thickening** lines the wall, so its thickness is twice the distance
  transform of the soft tissue inside the cavity;
* a **fluid level** is a *flat interface cutting across* the cavity, which is
  what separates it from a thickened lining and from a polyp: the interface
  has to be a straight chord spanning the cavity, not a curve following it;
* a **mucocele** is a dilated cavity with a thinned, expanded wall;
* a **fungal ball** is an opacified cavity with dense bone beside it and
  hyperdense material inside;
* **malignancy** is suggested by a broken wall with soft tissue outside it.

Cavities are located from HU where that is reliable, and the rest is
measured on a cavity mask, so a Phase 5 model can be dropped in later. The
locator cannot tell a sinus from a mastoid air cell, and the orbits can pass
the wall test, so every cavity is reported with the fraction of its wall that
is bone and the caller decides what to make of it. Nothing here is validated.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from vindr.ct.shapes import extent_mm, hydraulic_thickness_mm, volume_ml

CONN3 = np.ones((3, 3, 3), bool)

# published-ish cut-offs, in one place
MUCOSA_THICK_MM = 4.0  # normal mucosa is under 2 mm; 4 mm is the usual line
MUCOSA_FRACTION = 1.0 / 3.0  # or a third of the cavity, whichever comes first
BONE_HU = 200.0
AIR_HU = -400.0
FLUID_HU = (-10.0, 30.0)  # simple fluid, as in a level or a mucocele
SOFT_HU = (-100.0, 200.0)  # mucosa, polyp, tumour
HYPEROSTOSIS_RATIO = 1.35  # wall bone this much denser than the rest of the skull
MUCOCELE_WALL_MM = 1.5  # a wall this thin is one voxel; two voxels is normal
LEVEL_STRAIGHTNESS_MM = 2.0  # a level is flat; a lining is not
LEVEL_SPAN_RATIO = 0.6  # ...and it crosses the cavity rather than following it
CLEARANCE_MM = 6.0  # what counts as a cavity at all
CLEARANCE_REL = 0.35  # ...or this share of the head's own thickness
MIN_CAVITY_ML = 0.2
MAX_CAVITY_ML = 40.0  # the brain is a bone-enclosed soft-tissue region too

NOT_CLAIM = "не диагноз: критерий для проверки врачом"


def _head_extent_mm(volume: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Head thickness in mm: the 95-99 percentile spread of soft tissue."""
    soft = volume > SOFT_HU[0]
    if not soft.any():
        return float(volume.shape[2]) * float(spacing[2])
    proj = soft.sum(axis=(0, 1))
    idx = np.flatnonzero(proj)
    return float(idx.max() - idx.min() + 1) * float(spacing[2])


def _touches_edge(mask2d: np.ndarray, margin: int = 1) -> bool:
    if mask2d.shape[0] <= 2 * margin or mask2d.shape[1] <= 2 * margin:
        return True
    return bool(
        mask2d[:margin].any()
        or mask2d[-margin:].any()
        or mask2d[:, :margin].any()
        or mask2d[:, -margin:].any()
    )


def find_cavities(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    clearance_mm: float | None = None,
    min_ml: float = MIN_CAVITY_ML,
    max_ml: float = MAX_CAVITY_ML,
    wall_bone_min: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """Locate bone-enclosed cavities: air-filled ones and opacified ones.

    Two passes, because a fully opacified sinus contains no air at all and an
    air-only anchor would miss exactly the cases worth reporting. The air pass
    finds aerated sinuses; the soft-tissue pass finds opacified ones, with a
    volume cap that keeps the brain out. Both are filtered by how much of the
    region's shell is bone, and that fraction is returned per cavity so the
    caller can see how confident the locator is.

    The mastoid air cells and the orbits can pass these tests. That is a
    limitation, not a bug, and the report says so.
    """
    volume = np.asarray(volume)
    sp = [float(s) for s in spacing]
    px_ml = sp[0] * sp[1] * sp[2] / 1000.0
    head = _head_extent_mm(volume, spacing)
    if clearance_mm is None:
        clearance_mm = max(2.0, CLEARANCE_REL * head)
    n_px = round(clearance_mm / max(sp[2], 1e-6))
    n_px = max(1, min(n_px, min(volume.shape[1], volume.shape[2]) // 3))
    found: list[dict] = []
    labels = np.zeros(volume.shape, np.int32)

    def _candidates(mask: np.ndarray, kind: str) -> None:
        lab, n = ndi.label(mask, structure=CONN3)
        for i in range(1, n + 1):
            comp = lab == i
            ml = float(comp.sum()) * px_ml
            if ml < min_ml or ml > max_ml:
                continue
            # a cavity does not reach the edge of the head
            reaches = any(
                _touches_edge(comp[z], n_px) for z in range(comp.shape[0]) if comp[z].any()
            )
            if reaches:
                continue
            shell = ndi.binary_dilation(comp, CONN3) & ~comp
            wall = _wall_assessment(comp, shell, volume, spacing)
            if wall["bone_fraction"] < wall_bone_min:
                continue
            labels[comp] = len(found) + 1
            found.append(
                {
                    "id": len(found),
                    "kind": kind,
                    "volume_ml": round(ml, 2),
                    "centroid_zyx": [round(float(v), 1) for v in ndi.center_of_mass(comp)],
                    **wall,
                }
            )

    _candidates(volume < AIR_HU, "aerated")
    _candidates((volume >= SOFT_HU[0]) & (volume <= SOFT_HU[1]), "opacified")

    info = {
        "n_cavities": len(found),
        "clearance_mm": round(clearance_mm, 1),
        "wall_bone_min": wall_bone_min,
        "cavities": found,
        "note": "локатор не различает пазухи и ячейки сосцевидного отростка; "
        "орбиты тоже могут пройти порог стенки. Смотрите bone_fraction. " + NOT_CLAIM,
    }
    return labels, info


def _wall_assessment(
    cavity: np.ndarray,
    shell: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    reference_bone_hu: float | None = None,
    bone_hu: float = BONE_HU,
) -> dict:
    """How much of the cavity's shell is bone, how dense, and where it breaks.

    `hyperostosis_ratio` compares this wall against the *rest* of the skull, so
    it is only reported when a reference is handed in. Deriving the reference
    from this same volume would compare the wall with itself and return 1.0 on
    every scan, which looks like a measurement and is not one.
    """
    if not shell.any():
        return {
            "bone_fraction": 0.0,
            "wall_bone_hu": None,
            "wall_gaps": 0,
            "wall_thickness_mm": 0.0,
            "reference_bone_hu": None,
            "hyperostosis_ratio": None,
        }
    shell_bone = shell & (volume > bone_hu)
    fraction = float(shell_bone.sum()) / float(shell.sum())
    wall_hu = float(volume[shell_bone].mean()) if shell_bone.any() else None
    return {
        "bone_fraction": round(fraction, 3),
        "wall_bone_hu": round(wall_hu, 1) if wall_hu is not None else None,
        "wall_gaps": _wall_gaps(cavity, volume, spacing, bone_hu),
        "wall_thickness_mm": round(_wall_thickness_mm(cavity, volume, spacing, bone_hu), 2),
        "reference_bone_hu": round(float(reference_bone_hu), 1) if reference_bone_hu else None,
        "hyperostosis_ratio": (
            round(wall_hu / float(reference_bone_hu), 2)
            if wall_hu and reference_bone_hu
            else None
        ),
    }


def _reference_bone_hu(
    volume: np.ndarray,
    bone_hu: float = BONE_HU,
    exclude: np.ndarray | None = None,
) -> float | None:
    """Mean density of the skull's cortical bone, for a relative hyperostosis.

    `exclude` drops the bone belonging to the cavity being measured, so the
    comparison is against the rest of the skull and not against itself.
    """
    bone = volume > bone_hu
    if exclude is not None:
        bone = bone & ~np.asarray(exclude).astype(bool)
    if not bone.any():
        return None
    return float(volume[bone].mean())


def _wall_gaps(
    cavity: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    bone_hu: float = BONE_HU,
) -> int:
    """Count stretches of the cavity wall where bone is missing.

    A normal sinus is circled by bone on all sides. A defect shows up as a
    patch of the shell that is not bone, so the shell is labelled and the
    patches above a size floor are counted — a one-voxel nick along a staircase
    edge is noise, a patch covering a few percent of the circumference is not.
    The count is taken as the worst single slice, since a defect shows on the
    slices it crosses.
    """
    worst = 0
    px_area = float(spacing[1]) * float(spacing[2])
    for z in range(cavity.shape[0]):
        sl = cavity[z]
        if sl.sum() < 20:
            continue
        shell = ndi.binary_dilation(sl, np.ones((3, 3), bool)) & ~sl
        if not shell.any():
            continue
        not_bone = shell & ~(volume[z] > bone_hu)
        lab, n = ndi.label(not_bone, structure=np.ones((3, 3), bool))
        if n == 0:
            continue
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        # a patch has to be real, not a one-voxel nick on a staircase edge
        min_patch_mm2 = max(3.0, 0.01 * float(shell.sum()) * px_area)
        worst = max(worst, int((sizes[1:] * px_area >= min_patch_mm2).sum()))
    return worst


def _wall_thickness_mm(
    cavity: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    bone_hu: float = BONE_HU,
) -> float:
    """Thinnest bony wall around the cavity, as 2A/P on the bone ring.

    The same hydraulic argument as the kidney cortex: a wall of thickness t
    has area about P*t/2, so 2A/P lands on t. It is only meaningful where the
    bone is an actual *ring* around the cavity. At the top and bottom of a
    cavity the ring closes in and the bone mask becomes a filled disc, and 2A/P
    on a disc of radius r returns 4r/pi — several millimetres of "wall" where
    there is none. Those slices are therefore dropped, and the thinnest of what
    remains is reported, because a mucocele thins its wall where it expands.

    Returns 0.0 when no slice has a real ring, which means "not measured"
    rather than "no wall".
    """
    best: float | None = None
    for z in range(cavity.shape[0]):
        sl = cavity[z]
        if sl.sum() < 20:
            continue
        bone = volume[z] > bone_hu
        near = ndi.binary_dilation(sl, np.ones((3, 3), bool), iterations=2) & bone
        if not near.any():
            continue
        ring = near & ~ndi.binary_erosion(near, np.ones((3, 3), bool))
        if not ring.any():
            continue
        # a real ring does not cover the cavity's own centre
        cy, cx = ndi.center_of_mass(sl)
        yi, xi = round(cy), round(cx)
        if not (0 <= yi < ring.shape[0] and 0 <= xi < ring.shape[1]) or ring[yi, xi]:
            continue
        t = hydraulic_thickness_mm(ring, (spacing[1], spacing[2]))
        if t > 0.0:
            best = t if best is None else min(best, t)
    return round(best, 2) if best is not None else 0.0


def mucosal_thickness(
    cavity: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    soft: tuple[float, float] = SOFT_HU,
) -> dict:
    """Thickness of the soft-tissue lining of a cavity.

    The distance transform of the soft tissue inside the cavity peaks halfway
    across the lining, so the thickness is twice that peak. The 95th
    percentile is reported as well: a polyp or a tumour is thick in the middle
    and thin at the edges, and only the typical value separates it from a
    uniform lining.
    """
    cavity = np.asarray(cavity).astype(bool)
    soft_mask = cavity & (volume >= soft[0]) & (volume <= soft[1])
    if not soft_mask.any():
        return {"max_mm": 0.0, "typical_mm": 0.0, "soft_fraction": 0.0}
    # distance to the nearest non-soft voxel, i.e. to the wall or to the air
    dt = ndi.distance_transform_edt(soft_mask, sampling=(spacing[0], spacing[1], spacing[2]))
    # the percentile is over the lining itself: most of the volume is air, and
    # including it would report a 0 mm lining on every scan
    lined = dt[soft_mask]
    return {
        "max_mm": round(2.0 * float(lined.max()), 2),
        "typical_mm": round(2.0 * float(np.percentile(lined, 95)), 2),
        "soft_fraction": round(float(soft_mask.sum()) / float(cavity.sum()), 3),
    }


def fluid_level(
    cavity: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    straightness_mm: float = LEVEL_STRAIGHTNESS_MM,
    span_ratio: float = LEVEL_SPAN_RATIO,
) -> dict:
    """A level is a flat interface cutting across the cavity, not a lining.

    On each axial slice both air and simple fluid must be present in useful
    amounts, and the interface between them has to be a straight chord: the
    residual of a line fit through the interface has to stay under
    `straightness_mm`, and the chord has to span `span_ratio` of the cavity.
    A thickened lining follows the wall, so its interface is curved and short,
    and neither test passes. Two or more consecutive slices are required, so a
    single noisy slice cannot report a level.
    """
    cavity = np.asarray(cavity).astype(bool)
    sp = (float(spacing[1]), float(spacing[2]))
    per_slice: list[dict] = []
    for z in range(cavity.shape[0]):
        sl = cavity[z]
        area = float(sl.sum())
        if area < 40:
            continue
        air = sl & (volume[z] < AIR_HU)
        fluid = sl & (volume[z] >= FLUID_HU[0]) & (volume[z] <= FLUID_HU[1])
        if air.sum() < 0.15 * area or fluid.sum() < 0.15 * area:
            continue
        iface = (
            ndi.binary_dilation(air, np.ones((3, 3), bool))
            & ndi.binary_dilation(fluid, np.ones((3, 3), bool))
        ) & sl
        if iface.sum() < 8:
            continue
        yy, xx = np.nonzero(iface)
        pts = np.stack([yy * sp[0], xx * sp[1]], axis=1)
        pts = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        direction = vt[0]
        # how flat the interface is: the spread of its distance to the fitted
        # line, i.e. the projection onto that line's normal. RMS rather than
        # max so one stray voxel at the edge of the band cannot decide it.
        normal = vt[-1]
        dist = np.abs(pts @ normal)
        residual = float(np.sqrt(np.mean(dist**2)))
        angle = float(np.degrees(np.arctan2(direction[1], direction[0])))
        proj = pts @ direction
        chord = float(proj.max() - proj.min())
        # the cavity's own extent along the same direction
        cyy, cxx = np.nonzero(sl)
        cpts = np.stack([cyy * sp[0], cxx * sp[1]], axis=1) - np.array(
            [yy.mean() * sp[0], xx.mean() * sp[1]]
        )
        cproj = cpts @ direction
        cav_extent = float(cproj.max() - cproj.min())
        ratio = chord / cav_extent if cav_extent > 0 else 0.0
        if residual <= straightness_mm and ratio >= span_ratio:
            per_slice.append(
                {
                    "slice": z,
                    "straightness_mm": round(residual, 2),
                    "tilt_deg": round(angle % 90.0, 1),
                    "span_ratio": round(ratio, 2),
                    "air_area_frac": round(float(air.sum()) / area, 3),
                    "fluid_area_frac": round(float(fluid.sum()) / area, 3),
                }
            )
    run = best = 0
    prev = None
    for item in per_slice:
        run = run + 1 if prev is not None and item["slice"] == prev + 1 else 1
        best = max(best, run)
        prev = item["slice"]
    return {
        "present": best >= 2,
        "max_consecutive_slices": int(best),
        "slices": per_slice[:10],
        "note": "уровень — плоская граница, пересекающая полость; "
        f"допуск {straightness_mm} мм и {span_ratio:.0%} ширины полости. "
        "какая сторона границы воздушная, по этому не определяется: ориентация "
        "в загрузчике не хранится. tilt_deg — наклон границы в своей плоскости. " + NOT_CLAIM,
    }


def cavity_measurements(
    cavity: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    reference_volume_ml: float | None = None,
) -> dict:
    """Plain contents of a cavity: how much air, fluid, soft tissue, bone.

    Simple fluid sits inside the soft-tissue band, because HU alone cannot tell
    a level of fluid from a polyp — only the shape of the interface does that.
    So `fluid_fraction` is part of `soft_fraction` and the two do not sum to
    1; `soft_excluding_fluid_fraction` is given for anyone who needs the
    lining on its own.
    """
    cavity = np.asarray(cavity).astype(bool)
    n = float(cavity.sum())
    vals = volume[cavity]
    is_fluid = (vals >= FLUID_HU[0]) & (vals <= FLUID_HU[1])
    is_soft = (vals >= SOFT_HU[0]) & (vals <= SOFT_HU[1])
    air = float((vals < AIR_HU).mean())
    fluid = float(is_fluid.mean())
    soft = float(is_soft.mean())
    bone = float((vals > BONE_HU).mean())
    dense = float((vals > 150.0).mean())
    vol_ml = volume_ml(cavity, spacing)
    out = {
        "volume_ml": round(vol_ml, 2),
        "air_fraction": round(air, 3),
        "fluid_fraction": round(fluid, 3),
        "soft_fraction": round(soft, 3),
        "soft_excluding_fluid_fraction": round(float((is_soft & ~is_fluid).mean()) if n else 0.0, 3),
        "bone_fraction": round(bone, 3),
        "dense_fraction": round(dense, 3),
        "mean_hu": round(float(vals.mean()), 1) if n else None,
        "extent_mm": [round(v, 1) for v in extent_mm(cavity, spacing)],
        "note": "простая жидкость входит в soft_fraction: доли перекрываются",
    }
    if reference_volume_ml:
        out["volume_ratio_to_reference"] = round(vol_ml / reference_volume_ml, 2)
    return out


def sinus_findings(
    cavity: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    reference_bone_hu: float | None = None,
    reference_volume_ml: float | None = None,
) -> dict:
    """Everything the notes ask for, on one cavity.

    Sinusitis is mucosal thickening, judged either by an absolute 4 mm or by
    a third of the cavity, whichever is reached first. A mucocele is an
    expanded cavity with a thin wall. A fungal ball is an opacified cavity
    whose wall bone is denser than the rest of the skull and whose contents
    are not simple fluid. A broken wall with soft tissue outside it is what
    would raise malignancy.
    """
    cavity = np.asarray(cavity).astype(bool)
    if not cavity.any():
        return {"ok": False, "reason": "пустая маска"}
    contents = cavity_measurements(cavity, volume, spacing, reference_volume_ml)
    mucosa = mucosal_thickness(cavity, volume, spacing)
    level = fluid_level(cavity, volume, spacing)
    shell = ndi.binary_dilation(cavity, CONN3) & ~cavity
    if reference_bone_hu is None:
        # the rest of the skull, so the wall is not compared with itself
        reference_bone_hu = _reference_bone_hu(volume, exclude=shell)
    wall = _wall_assessment(cavity, shell, volume, spacing, reference_bone_hu=reference_bone_hu)
    smallest = min(contents["extent_mm"][1], contents["extent_mm"][2])
    out: dict = {
        "ok": True,
        "contents": contents,
        "mucosa": mucosa,
        "fluid_level": level,
        "wall": wall,
        "suggests": None,
        "flags": [],
    }
    if contents["air_fraction"] > 0.5:
        out["suggests"] = "normal_aerated"
    if mucosa["typical_mm"] >= MUCOSA_THICK_MM or mucosa["max_mm"] >= MUCOSA_FRACTION * smallest:
        out["flags"].append(
            f"слизистая {mucosa['typical_mm']} мм (типично) / {mucosa['max_mm']} мм (макс) — утолщение"
        )
        out["suggests"] = "sinusitis"
    if level["present"]:
        out["flags"].append("горизонтальный уровень жидкости в полости")
        out["suggests"] = "fluid_level"
    if contents["air_fraction"] < 0.1 and contents["soft_fraction"] > 0.5:
        out["flags"].append("полость полностью закрыта мягкотканным содержимым")
        if out["suggests"] is None or out["suggests"] == "normal_aerated":
            out["suggests"] = "opacified"
    if out["suggests"] == "opacified" and contents["dense_fraction"] > 0.05:
        out["flags"].append(
            f"плотные включения {contents['dense_fraction']:.0%} содержимого — грибковый шар, а не простая жидкость"
        )
        out["suggests"] = "fungal_ball"
    ratio = wall.get("hyperostosis_ratio")
    if ratio and ratio >= HYPEROSTOSIS_RATIO and out["suggests"] in ("opacified", "fungal_ball"):
        out["flags"].append(f"кора полости в {ratio} раза плотнее остального черепа — гиперостеоз")
    if wall["wall_gaps"] >= 2 and out["suggests"] in ("opacified", "fungal_ball", "sinusitis"):
        out["flags"].append(f"разрывов стенки: {wall['wall_gaps']} — вероятно разрушение кости")
        out["suggests"] = "wall_destruction"
    if (
        reference_volume_ml
        and contents["volume_ml"] > 1.5 * reference_volume_ml
        and 0.0 < wall["wall_thickness_mm"] < MUCOCELE_WALL_MM
    ):
        out["flags"].append("расширенная полость с истончённой стенкой — мукоцеле")
        out["suggests"] = "mucocele"
    out["outside_soft_tissue"] = _extracavitary_soft_tissue(cavity, volume, spacing)
    if out["outside_soft_tissue"]["fraction_in_break"] >= 0.3:
        out["flags"].append(
            f"мягкотканный компонент вне полости: {out['outside_soft_tissue']['fraction_in_break']:.0%} — "
            "стоит исключить опухоль"
        )
        out["suggests"] = "malignancy_hint"
    out["note"] = (
        "критерии по конспекту; локатор не различает пазухи и ячейки сосцевидного отростка. " + NOT_CLAIM
    )
    return out


def _extracavitary_soft_tissue(
    cavity: np.ndarray,
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    reach_mm: float = 4.0,
) -> dict:
    """Soft tissue growing out through a break in the wall.

    A normal sinus is circled by bone, and whatever lies beyond that bone is
    somebody else's tissue — so measuring soft tissue in a plain 4 mm ring
    around the cavity reports the orbit or the brain on every normal scan. The
    measurement therefore starts where the bone is *missing*: the gaps in the
    wall ring, and the soft tissue reachable outwards from them.

    That is the geometry malignancy has to produce, and it stays silent on a
    cavity whose wall is intact.
    """
    wall = ndi.binary_dilation(cavity, CONN3) & ~cavity
    gaps = wall & ~(volume > BONE_HU)
    if not gaps.any():
        return {"gap_ml": 0.0, "outside_soft_ml": 0.0, "fraction_in_break": 0.0, "wall_gaps": 0}
    _, gaps_n = ndi.label(gaps, structure=CONN3)
    outside = ndi.distance_transform_edt(
        ~cavity, sampling=(spacing[0], spacing[1], spacing[2])
    )
    reachable = ndi.binary_dilation(gaps, CONN3, iterations=max(1, round(reach_mm / max(float(spacing[2]), 1e-6))))
    shell = reachable & ~cavity & (outside > 0.0)
    soft = shell & (volume >= SOFT_HU[0]) & (volume <= SOFT_HU[1])
    return {
        "gap_ml": round(volume_ml(gaps, spacing), 3),
        "outside_soft_ml": round(volume_ml(soft, spacing), 3),
        "fraction_in_break": round(float(soft.sum()) / float(shell.sum()), 3) if shell.any() else 0.0,
        "wall_gaps": int(gaps_n),
    }


def sinus_report(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    cavities: np.ndarray | None = None,
    reference_volume_ml: float | None = None,
) -> dict:
    """Phase 5 block for the report.

    The cavities are located from HU, because a bone-enclosed air or soft-tissue
    region is findable without a model. A caller with a Phase 5 segmentation can
    pass `cavities` instead, and then nothing is located at all.

    A mucocele is a dilated cavity, which means a comparison against a normal
    one. Left without a reference the criterion cannot run, so when no volume is
    given the *other* cavities on this scan are used — with only one cavity
    there is nothing to compare against and the criterion stays off.
    """
    volume = np.asarray(volume)
    out: dict = {"phase": 5, "note": NOT_CLAIM}
    if cavities is None:
        labels, info = find_cavities(volume, spacing)
        out["locator"] = info
    else:
        labels = np.asarray(cavities).astype(np.int32)
        info = {"n_cavities": int(labels.max()), "cavities": [], "note": "маски поданы извне"}
    out["ok"] = bool(labels.max() > 0)
    if not out["ok"]:
        out["cavities"] = []
        out["abnormal"] = []
        out["reason"] = "полостей, окружённых костью, не найдено"
        return out
    ref_bone = _reference_bone_hu(volume)
    masks = {i: labels == i for i in range(1, int(labels.max()) + 1)}
    vols = {i: volume_ml(m, spacing) for i, m in masks.items()}
    findings = []
    for i, mask in masks.items():
        # no reference given, so fall back to the other cavities on this scan:
        # a mucocele is a dilated cavity, and dilation is only visible against
        # a normal one. A cavity is never compared against itself.
        ref = reference_volume_ml
        if ref is None and len(masks) >= 2:
            ref = float(np.median([v for j, v in vols.items() if j != i]))
        f = sinus_findings(mask, volume, spacing, reference_bone_hu=ref_bone, reference_volume_ml=ref)
        loc = next((c for c in info.get("cavities", []) if c["id"] == i), {})
        f["location"] = {k: loc.get(k) for k in ("kind", "centroid_zyx", "bone_fraction")}
        f["reference_volume_ml"] = round(ref, 2) if ref else None
        f["reference_source"] = (
            "задан извне" if reference_volume_ml else
            ("медиана остальных полостей этого исследования" if ref else "нет с чем сравнить — мукоцеле не оценивается")
        )
        findings.append(f)
    findings.sort(key=lambda d: -(d.get("contents", {}).get("volume_ml") or 0.0))
    out["n_cavities"] = len(findings)
    out["cavities"] = findings
    out["abnormal"] = [f["suggests"] for f in findings if f.get("suggests") not in (None, "normal_aerated")]
    out["reference_volume_ml"] = reference_volume_ml
    out["caveat"] = (
        "локатор по HU находит и ячейки сосцевидного отростка, и иногда орбиты; "
        "ориентация в загрузчике не хранится, поэтому названия пазух не присваиваются"
    )
    return out
