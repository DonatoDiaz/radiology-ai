"""Skull-vault fracture detection for head CT (Phase 3, heuristic).

The whole calvarium sits above 700 HU, so counting dense voxels flags every
normal study and tells the reader nothing. A fracture is a *discontinuity in an
otherwise smooth ring*, and that is what this looks for, slice by slice:

* **lucent line** — a ray leaving the brain centre normally crosses the vault
  once; a fracture splits that bone into two runs with a thin dark gap between
  them;
* **displaced step** — one side of the outer table sits deeper (depressed) or
  higher (elevated) than its mirror across the midline over a wide sector;
* **free debris** — small bone fragments beside the vault (comminution).

Findings on consecutive slices that share an angular sector are grouped into
one 3D line, and :func:`associate_with_hematoma` reports the classic epidural
pairing (a fracture next to a biconvex hematoma).

Slices whose anatomy is not a clean vault — the skull base, where a ray
legitimately meets the petrous bone more than once — are *skipped and counted*
rather than guessed at. Sutures are the main mimic for the lucent line: they
are lucent too, so the sagittal suture at the midline is masked out
(:data:`SUTURE_GUARD_DEG`) and everything else is reported as suspicion only.

Heuristic morphology descriptors, not a diagnosis — same caveat as lesions.py.
"""

from __future__ import annotations

import cv2
import numpy as np

from vindr.ct.volume import CTSeries

# cortical bone; the conventional bone window starts at 500, this catches the
# partial-volume edge of a fracture line
HU_BONE_MIN = 400.0
# angular resolution of the vault profile
N_ANGLES = 180
# the sagittal suture sits on the midline, front and back: suppress +/- this
# many degrees around both, where a lucent line is expected anatomy
SUTURE_GUARD_DEG = 12.0
# a displaced outer table has to step at least this far to be more than shape
STEP_MIN_MM = 2.5
# ...and at least this many times the head's own left-right asymmetry
MIRROR_NOISE_FACTOR = 4.0
# ...and at least this share of the vault radius, for a big head
MIRROR_MIN_REL = 0.05
# ...and the step has to persist over this many degrees to be a line, not a notch
STEP_MIN_RUN_DEG = 20.0
# a lucent line has to persist over this many degrees
LUCENT_MIN_RUN_DEG = 12.0
# a fracture line is thin: wider gaps are partial volume, sutures or sinuses
GAP_MIN_MM = 0.6
GAP_MAX_MM = 3.5
# debris: free bone fragments beside the vault
DEBRIS_MIN_AREA_PX = 4
DEBRIS_MAX_AREA_PX = 400
DEBRIS_MARGIN_MM = 8.0
# slices below this bone fraction are not a vault (vertex, skull base)
MIN_RING_AREA_FRAC = 0.015
# ...and above this the slice is dominated by bone and the ring is not isolated
MAX_RING_AREA_FRAC = 0.45


def _circular_runs(flag: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """Contiguous circular runs of True, as [start, stop) index pairs."""
    n = int(flag.size)
    if n == 0 or not flag.any():
        return []
    # rotate so index 0 is False: no run then crosses the array boundary
    first_false = int(np.argmax(~flag))
    f = np.roll(flag, -first_false)
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if f[i]:
            j = i
            while j < n and f[j]:
                j += 1
            if j - i >= min_len:
                runs.append((i, j))
            i = j
        else:
            i += 1
    return [((a + first_false) % n, (b + first_false) % n) for a, b in runs]


def _circular_smooth(profile: np.ndarray, width: int) -> np.ndarray:
    """Moving average around a circle, so the ends do not jump."""
    if width <= 1:
        return profile.copy()
    half = width // 2
    acc = np.zeros_like(profile, dtype=np.float64)
    for k in range(-half, half + 1):
        acc += np.roll(profile, k)
    return acc / (2 * half + 1)


def _fill_angles(profile: np.ndarray) -> np.ndarray:
    """Linearly interpolate the angles where the ring has no bone at all."""
    good = ~np.isnan(profile)
    if not good.any():
        return np.full_like(profile, np.nan)
    if good.all():
        return profile
    n = profile.size
    idx = np.arange(n)
    filled = np.interp(idx, idx[good], profile[good], period=n)
    return filled


def _sample_rays(
    hu_plane: np.ndarray, centroid: tuple[float, float], n_angles: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample HU along one ray per angle, from the brain centre outwards.

    Returns (bone mask per angle, radius per sample, HU per sample).
    """
    h, w = hu_plane.shape
    cy, cx = centroid
    theta = np.arange(n_angles) * (2.0 * np.pi / n_angles)
    step = 0.5  # half-pixel sampling: thin fracture lines are only a few px
    r = np.arange(0.0, float(np.hypot(h, w)), step)
    ys = np.rint(cy + r[None, :] * np.sin(theta)[:, None]).astype(int)
    xs = np.rint(cx + r[None, :] * np.cos(theta)[:, None]).astype(int)
    inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
    vals = np.where(inside, hu_plane[np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)], -1000.0)
    r_grid = np.broadcast_to(r[None, :], vals.shape)
    return vals > HU_BONE_MIN, r_grid, vals


def _segment_counts(bone: np.ndarray) -> np.ndarray:
    """How many separate bone runs each ray crosses."""
    prev = np.zeros((bone.shape[0], 1), bool)
    starts = bone & ~np.hstack([prev, bone[:, :-1]])
    return starts.sum(1)


def _internal_gaps(bone_row: np.ndarray, step_mm: float) -> list[float]:
    """Widths (mm) of the dark gaps sitting strictly between bone runs."""
    idx = np.flatnonzero(bone_row)
    if idx.size < 2:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    gaps = []
    for b in breaks:
        a, c = int(idx[b]) + 1, int(idx[b + 1]) - 1
        if c >= a:
            gaps.append((c - a + 1) * step_mm)
    return gaps


def _largest_component(binary: np.ndarray) -> np.ndarray:
    """Largest 8-connected blob, holes filled (the vault is a closed shell)."""
    b = (np.asarray(binary) > 0).astype(np.uint8)
    if b.sum() == 0:
        return b.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    if n <= 1:
        return b.astype(bool)
    keep = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    comp = (labels == keep).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        comp = cv2.drawContours(np.zeros_like(comp), cnts, -1, 1, thickness=cv2.FILLED)
    return comp.astype(bool)


def analyze_slice(
    hu_plane: np.ndarray,
    spacing_xy: tuple[float, float],
    n_angles: int = N_ANGLES,
) -> dict:
    """Fracture evidence on one axial slice.

    ``None`` is returned when the slice is not a clean vault (skull base, very
    low vertex), which is the honest answer: the ray geometry is ambiguous.
    """
    hu = np.asarray(hu_plane, dtype=np.float32)
    bone = hu > HU_BONE_MIN
    if not bone.any():
        return {"ok": False, "reason": "no bone"}
    ring = _largest_component(bone)
    area_frac = float(ring.mean())
    if area_frac < MIN_RING_AREA_FRAC:
        return {"ok": False, "reason": "ring too small"}
    if area_frac > MAX_RING_AREA_FRAC:
        return {"ok": False, "reason": "slice dominated by bone"}

    ys, xs = np.nonzero(ring)
    cy, cx = float(ys.mean()), float(xs.mean())
    # a radial deviation is only meaningful in mm, and the two axes differ
    sp_y, sp_x = float(spacing_xy[0]), float(spacing_xy[1])

    bone_rays, r_grid, _ = _sample_rays(hu, (cy, cx), n_angles)
    segs = _segment_counts(bone_rays)
    covered = float((segs > 0).mean())
    if covered < 0.8:
        return {"ok": False, "reason": "vault not closed"}
    if int(np.median(segs)) > 1:
        # an intact vault is crossed once; more often means petrous bone, a
        # facial bone or the skull base, where "two bone runs" is normal
        return {"ok": False, "reason": "ray crosses bone more than once"}

    findings: list[dict] = []

    # --- lucent line: a thin dark gap splitting the vault bone
    guarded = _suture_guard(n_angles)
    # the rays are sampled every 0.5 px, so a gap's width in mm is samples * this
    sample_mm = 0.5 * (sp_y + sp_x) / 2.0
    lucent = np.zeros(n_angles, bool)
    for i in np.flatnonzero(segs >= 2):
        if guarded[i]:
            continue
        gaps = _internal_gaps(bone_rays[i], sample_mm)
        if any(GAP_MIN_MM <= g <= GAP_MAX_MM for g in gaps):
            lucent[i] = True
    for a, b in _circular_runs(lucent, max(1, round(LUCENT_MIN_RUN_DEG / 360.0 * n_angles))):
        findings.append(
            {
                "kind": "lucent_line",
                "angle_start_deg": round(a * 360.0 / n_angles, 1),
                "angle_end_deg": round(b * 360.0 / n_angles, 1),
                "run_deg": round((b - a) * 360.0 / n_angles, 1),
            }
        )

    # --- displaced step: one side of the vault sitting deeper than the other.
    # A plain radial deviation is measured from a centre that the defect itself
    # displaces, and that shift cancels most of the signal (a 6 mm depression
    # reads as 1.8 mm). Comparing each direction against its mirror across the
    # midline cancels the shift to first order and leaves the step behind.
    outer_px = np.where(bone_rays.any(1), (r_grid * bone_rays).max(1), np.nan)
    outer = _fill_angles(outer_px)
    vault_radius_mm = None
    if np.isfinite(outer).any():
        outer_mm = outer * (sp_y + sp_x) / 2.0
        vault_radius_mm = float(np.nanmedian(outer_mm))
        mirror = (n_angles // 2 - np.arange(n_angles)) % n_angles
        dev = outer_mm - outer_mm[mirror]
        # natural left-right asymmetry of a real head sets the noise floor
        noise = float(np.median(np.abs(dev)))
        thresh = max(
            STEP_MIN_MM,
            MIRROR_NOISE_FACTOR * noise,
            MIRROR_MIN_REL * vault_radius_mm,
        )
        step_flag = (np.abs(dev) > thresh) & (~guarded)
        for a, b in _circular_runs(step_flag, max(1, round(STEP_MIN_RUN_DEG / 360.0 * n_angles))):
            seg = dev[a:b]
            findings.append(
                {
                    "kind": "depressed" if float(np.mean(seg)) < 0 else "elevated",
                    "angle_start_deg": round(a * 360.0 / n_angles, 1),
                    "angle_end_deg": round(b * 360.0 / n_angles, 1),
                    "run_deg": round((b - a) * 360.0 / n_angles, 1),
                    "step_mm": round(float(np.abs(seg).max()), 2),
                    "mirror_noise_mm": round(noise, 2),
                }
            )

    # --- debris: bone fragments beside the vault (comminution)
    debris = _debris(bone, ring, (sp_y, sp_x))
    return {
        "ok": True,
        "slice_area_frac": round(area_frac, 4),
        "centroid_xy": [round(cx, 1), round(cy, 1)],
        "vault_radius_mm": round(vault_radius_mm, 1) if vault_radius_mm is not None else None,
        "n_ray_segments_median": np.median(segs),
        "n_debris": len(debris),
        "debris": debris[:5],
        "findings": findings,
    }


def _suture_guard(n_angles: int) -> np.ndarray:
    """True where a lucent line would be normal anatomy (midline sutures)."""
    deg = np.arange(n_angles) * 360.0 / n_angles
    half = SUTURE_GUARD_DEG
    return ((np.abs(deg) <= half) | (np.abs(deg - 180.0) <= half))


def _debris(bone: np.ndarray, ring: np.ndarray, spacing_xy: tuple[float, float]) -> list[dict]:
    """Bone blobs that touch neither the vault nor each other."""
    free = (bone & ~ring).astype(np.uint8)
    if free.sum() == 0:
        return []
    n, _labels, stats, cents = cv2.connectedComponentsWithStats(free, connectivity=8)
    sp_y, sp_x = spacing_xy
    ring_px = cv2.dilate(ring.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (DEBRIS_MIN_AREA_PX <= area <= DEBRIS_MAX_AREA_PX):
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        pad = DEBRIS_MARGIN_MM / max(sp_y, sp_x, 1e-6)
        lo_y, hi_y = max(0, y - int(pad)), min(ring.shape[0], y + h + int(pad))
        lo_x, hi_x = max(0, x - int(pad)), min(ring.shape[1], x + w + int(pad))
        near = ring_px[lo_y:hi_y, lo_x:hi_x]
        if not near.any():
            continue  # far from the vault: probably cervical bone or an artefact
        out.append({"area_px": int(area), "centroid_xy": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)]})
    return out


def _angles_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Overlap of two angular sectors in degrees."""
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)


def group_lines(slice_results: list[tuple[int, dict]], n_angles: int = N_ANGLES) -> list[dict]:
    """Group per-slice findings that persist on consecutive slices.

    A fracture is a *line* through the volume, so evidence that stops after one
    slice is a notch or an artefact rather than a fracture.
    """
    lines: list[dict] = []
    tol = max(1, round(LUCENT_MIN_RUN_DEG / 360.0 * n_angles)) * 360.0 / n_angles
    for z, res in slice_results:
        for f in res.get("findings", []):
            sector = (f["angle_start_deg"], f["angle_end_deg"])
            for ln in lines:
                # a line is continuous evidence: it has to reach the previous slice
                if ln["z_end"] != z - 1:
                    continue
                if _angles_overlap(*sector, *ln["sector"]) > tol:
                    ln["z_end"] = z
                    ln["n_slices"] += 1
                    ln["kinds"].add(f["kind"])
                    if f.get("step_mm"):
                        ln["max_step_mm"] = max(ln["max_step_mm"], f["step_mm"])
                    break
            else:
                lines.append(
                    {
                        "z_start": z,
                        "z_end": z,
                        "n_slices": 1,
                        "sector": sector,
                        "kinds": {f["kind"]},
                        "max_step_mm": float(f.get("step_mm", 0.0)),
                    }
                )
    out = []
    for ln in lines:
        out.append(
            {
                "z_start": ln["z_start"],
                "z_end": ln["z_end"],
                "n_slices": ln["n_slices"],
                "angle_start_deg": round(ln["sector"][0], 1),
                "angle_end_deg": round(ln["sector"][1], 1),
                "kinds": sorted(ln["kinds"]),
                "max_step_mm": round(ln["max_step_mm"], 2),
            }
        )
    return out


def skull_fractures(series: CTSeries, min_slices: int = 2) -> dict:
    """Study-level skull-vault fracture assessment.

    Returns the candidate lines, the slices that could be evaluated at all, and
    an explicit ``flag`` — never a diagnosis.
    """
    sp_y, sp_x = float(series.spacing[1]), float(series.spacing[2])
    evaluated: list[tuple[int, dict]] = []
    skip_reasons: list[str] = []
    debris_total = 0
    for z in range(series.n_slices):
        res = analyze_slice(series.volume[z], (sp_y, sp_x))
        if res.get("ok"):
            evaluated.append((z, res))
            debris_total += int(res.get("n_debris", 0))
        else:
            skip_reasons.append(str(res.get("reason", "unknown")))
    lines = [ln for ln in group_lines(evaluated) if ln["n_slices"] >= min_slices]
    return {
        "n_slices_evaluated": len(evaluated),
        "n_slices_skipped": len(skip_reasons),
        "skip_reasons": sorted(set(skip_reasons)),
        "n_candidate_lines": len(lines),
        "n_debris": debris_total,
        "lines": lines,
        "flag": bool(lines) or debris_total > 0,
        "note": "эвристика: разрыв контура свода, ступенька или осколки кости — не диагноз",
    }


def associate_with_hematoma(
    series: CTSeries, mask: np.ndarray, max_distance_mm: float = 15.0
) -> dict:
    """Flag a hematoma lying next to the vault — the epidural pairing.

    A biconvex hematoma touching bone means looking for the fracture that
    produced it, so the distance from the blood to the nearest bone matters.
    """
    vol = np.asarray(mask).astype(bool)
    if not vol.any():
        return {"n_adjacent_slices": 0, "min_distance_mm": None, "adjacent": False}
    sp_y, sp_x = float(series.spacing[1]), float(series.spacing[2])
    hits = []
    for z in range(min(vol.shape[0], series.n_slices)):
        if not vol[z].any():
            continue
        bone = series.volume[z] > HU_BONE_MIN
        if not bone.any():
            continue
        dist = cv2.distanceTransform((~bone).astype(np.uint8), cv2.DIST_L2, 3)
        d = float(dist[vol[z]].min())
        hits.append((z, d * (sp_y + sp_x) / 2.0))
    if not hits:
        return {"n_adjacent_slices": 0, "min_distance_mm": None, "adjacent": False}
    best = min(hits, key=lambda t: t[1])
    near = [z for z, d in hits if d <= max_distance_mm]
    return {
        "n_adjacent_slices": len(near),
        "min_distance_mm": round(best[1], 1),
        "closest_slice": best[0],
        "adjacent": bool(near),
        "note": "кровь у кости → искать перелом (эпидуральная ассоциация)",
    }
