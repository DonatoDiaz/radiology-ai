"""Lesion extraction and semiology from a hemorrhage mask (Phase 3).

Turns a binary (or per-subtype) segmentation mask into discrete findings:
3D components with bounding boxes, volumes, and the morphology signs the
roadmap asks to encode into the report:

* biconvex and limited by sutures -> epidural
* concave / not limited, spreads along the convexity -> subdural
* follows the sulci / fissures          -> subarachnoid
* rounded inside the parenchyma         -> intraparenchymal

These are *heuristic* morphology descriptors, not a diagnosis — they mirror
the notes ("по конспекту") and exist to be checked against the classifier
output, not to replace it.
"""

from __future__ import annotations

import cv2
import numpy as np

# margin (in px) within which a lesion counts as "touching bone/cortex"
BONE_MARGIN_PX = 6
# solidity (area / convex-hull area) above which a component is convex (biconvex)
CONVEX_SOLIDITY = 0.72
# solidity below which the outline is concave — crescentic or serpiginous
CONCAVE_SOLIDITY = 0.50
# circularity (4*pi*area / perimeter^2) below which the outline is convoluted
CONVOLUTED_CIRCULARITY = 0.45


def find_components(mask2d: np.ndarray, min_area: int = 20) -> list[dict]:
    """Connected components of one 2D slice mask (8-connectivity)."""
    binary = (np.asarray(mask2d) > 0).astype(np.uint8)
    if binary.sum() == 0:
        return []
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out: list[dict] = []
    for i in range(1, n):  # 0 is the background
        x, y, w, h, area = stats[i].tolist()
        if area < min_area:
            continue
        comp = (labels[y : y + h, x : x + w] == i).astype(np.uint8)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_area = float(cv2.contourArea(cv2.convexHull(cnts[0]))) if cnts else float(area)
        perimeter = float(cv2.arcLength(cnts[0], True)) if cnts else 0.0
        # contourArea walks the pixel-centre polygon, so it reports slightly less
        # than the pixel count (a 48x48 block gives 47x47) — clamp at 1.0.
        solidity = min(1.0, float(area) / hull_area) if hull_area > 0 else 1.0
        circularity = (4.0 * np.pi * float(area)) / (perimeter**2) if perimeter > 0 else 0.0
        out.append(
            {
                "bbox_xyxy": [int(x), int(y), int(x + w), int(y + h)],
                "area_px": int(area),
                "centroid_xy": [round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)],
                "solidity": round(solidity, 3),
                "circularity": round(circularity, 3),
            }
        )
    return out


def _touches_border(comp: dict, shape_hw: tuple[int, int], margin: int = BONE_MARGIN_PX) -> dict:
    """Which borders of the slice a component reaches (inner/outer table)."""
    x1, y1, x2, y2 = comp["bbox_xyxy"]
    h, w = shape_hw
    return {
        "top": y1 <= margin,
        "bottom": y2 >= h - margin,
        "left": x1 <= margin,
        "right": x2 >= w - margin,
    }


def _crosses_midline(comp: dict, width: int) -> bool:
    """Does the component straddle the sagittal midline of the image?

    A subdural collection routinely crosses it; an epidural hematoma is
    confined to one side by the sutures, so it does not.
    """
    x1, _y1, x2, _y2 = comp["bbox_xyxy"]
    return x1 < width / 2.0 < x2


def lesion_semiotics(mask2d: np.ndarray) -> list[dict]:
    """Morphology signs per 2D component — the notes' semiology, computed.

    The discriminating features are how the collection relates to the inner
    table of the skull and whether it bulges away from it:

    * compact and convex against the bone, limited in extent -> epidural
      (двояковыпуклая, в пределах швов);
    * long and concave, hugging the convexity and crossing the midline ->
      subdural (ничем не отграничена, серповидная);
    * not reaching the bone, following sulci -> subarachnoid (по бороздам);
    * rounded inside the parenchyma -> intraparenchymal.
    """
    binary = (np.asarray(mask2d) > 0).astype(np.uint8)
    out: list[dict] = []
    for comp in find_components(binary):
        borders = _touches_border(comp, binary.shape)
        touches_any = any(borders.values())
        crosses_midline = _crosses_midline(comp, binary.shape[1])
        solidity = comp["solidity"]
        circularity = comp["circularity"]
        if (
            touches_any
            and not crosses_midline
            and solidity >= CONVEX_SOLIDITY
            and circularity >= CONVOLUTED_CIRCULARITY
        ):
            shape_hint = "biconvex_limited"          # двояковыпуклая, в пределах швов
            suggests = "epidural"
        elif touches_any:
            shape_hint = "crescentic_unconfined"     # серповидная, ничем не отграничена
            suggests = "subdural"
        elif circularity < CONVOLUTED_CIRCULARITY or solidity < CONCAVE_SOLIDITY:
            shape_hint = "sulcal_fissural"           # по бороздам/щелям
            suggests = "subarachnoid"
        else:
            shape_hint = "rounded_parenchymal"       # округлое в паренхиме
            suggests = "intraparenchymal"
        out.append(
            {
                **comp,
                "borders": borders,
                "crosses_midline": crosses_midline,
                "shape_hint": shape_hint,
                "suggests": suggests,
            }
        )
    return out


def find_lesions(
    mask3d: np.ndarray,
    spacing: tuple[float, float, float],
    min_area_px: int = 20,
    min_volume_ml: float = 0.1,
) -> list[dict]:
    """Group a 3D mask into lesions spanning consecutive slices.

    Consecutive slices whose boxes overlap in-plane are treated as one lesion,
    which is what makes an epidural/subdural collection a single finding
    instead of a stack of per-slice blobs.
    """
    volume = np.asarray(mask3d)
    if volume.ndim != 3:
        raise ValueError(f"expected a (Z, Y, X) mask, got shape {volume.shape}")
    z_sp, y_sp, x_sp = (float(s) for s in spacing)
    ml_per_px = z_sp * y_sp * x_sp / 1000.0

    open_groups: list[dict] = []  # lesions that may still grow
    closed: list[dict] = []
    for z in range(volume.shape[0]):
        comps = lesion_semiotics(volume[z]) if volume[z].any() else []
        if not comps:  # an empty slice breaks every open lesion
            closed.extend(open_groups)
            open_groups = []
            continue
        new_groups: list[dict] = []
        for comp in comps:
            x1, y1, x2, y2 = comp["bbox_xyxy"]
            best, best_iou = None, 0.0
            for g in open_groups:
                gx1, gy1, gx2, gy2 = g["bbox_xyxy"]
                inter = max(0, min(x2, gx2) - max(x1, gx1)) * max(0, min(y2, gy2) - max(y1, gy1))
                area = max(1, (x2 - x1) * (y2 - y1))
                iou = inter / area
                if iou > best_iou:
                    best, best_iou = g, iou
            if best is not None and best_iou >= 0.2:
                best["bbox_xyxy"] = [
                    min(best["bbox_xyxy"][0], x1),
                    min(best["bbox_xyxy"][1], y1),
                    max(best["bbox_xyxy"][2], x2),
                    max(best["bbox_xyxy"][3], y2),
                ]
                best["slices"].append(z)
                best["area_px"] += comp["area_px"]
                best["shape_hints"].append(comp["shape_hint"])
                best["suggests"].append(comp["suggests"])
                new_groups.append(best)
                open_groups.remove(best)
            else:
                new_groups.append(
                    {
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "slices": [z],
                        "area_px": comp["area_px"],
                        "shape_hints": [comp["shape_hint"]],
                        "suggests": [comp["suggests"]],
                        "z_start": z,
                        "z_end": z,
                    }
                )
        closed.extend(open_groups)  # a gap in z ends the lesion
        open_groups = new_groups
    closed.extend(open_groups)

    lesions: list[dict] = []
    for g in closed:
        vol_ml = g["area_px"] * ml_per_px
        if vol_ml < min_volume_ml:
            continue
        votes: dict[str, int] = {}
        for s in g["suggests"]:
            votes[s] = votes.get(s, 0) + 1
        ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
        z1, z2 = min(g["slices"]), max(g["slices"])
        x1, y1, x2, y2 = g["bbox_xyxy"]
        lesions.append(
            {
                "z_range": [z1, z2],
                "n_slices": len(g["slices"]),
                "bbox_zyx": [z1, y1, z2, y2],
                "bbox_xyxy": [x1, y1, x2, y2],
                "area_px": int(g["area_px"]),
                "volume_ml": round(vol_ml, 2),
                "suggests": ranked[0][0] if ranked else None,
                "suggestion_confidence": round(
                    ranked[0][1] / max(1, sum(votes.values())), 2
                ) if ranked else 0.0,
                "alternatives": {k: v for k, v in ranked[1:]},
                "shape_hints": sorted(set(g["shape_hints"])),
            }
        )
    lesions.sort(key=lambda d: d["volume_ml"], reverse=True)
    return lesions


def summarize_lesions(mask3d: np.ndarray, spacing: tuple[float, float, float], **kw) -> dict:
    """Report block for a mask: total volume plus per-lesion findings."""
    lesions = find_lesions(mask3d, spacing, **kw)
    volume = np.asarray(mask3d)
    z_sp, y_sp, x_sp = (float(s) for s in spacing)
    total_ml = float(volume.sum()) * z_sp * y_sp * x_sp / 1000.0
    by_type: dict[str, float] = {}
    for les in lesions:
        key = les.get("suggests") or "unclassified"
        by_type[key] = round(by_type.get(key, 0.0) + les["volume_ml"], 2)
    return {
        "n_lesions": len(lesions),
        "total_volume_ml": round(total_ml, 2),
        "volume_ml_by_suggested_type": by_type,
        "lesions": lesions,
    }
