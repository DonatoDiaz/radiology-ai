"""Tests for skull-fracture suspicion (vindr.ct.fracture).

The property that matters is not "does it find the line I drew" but the
opposite: a normal skull is a smooth ring, and the detector has to stay quiet
on it. Both directions are pinned here, along with the honest reporting of
slices the ray geometry could not judge.
"""

from __future__ import annotations

import numpy as np
import pytest

from vindr.ct import CTSeries
from vindr.ct.fracture import (
    GAP_MAX_MM,
    GAP_MIN_MM,
    HU_BONE_MIN,
    LUCENT_MIN_RUN_DEG,
    _circular_runs,
    _circular_smooth,
    _fill_angles,
    _internal_gaps,
    _segment_counts,
    analyze_slice,
    associate_with_hematoma,
    group_lines,
    skull_fractures,
)

SPACING = (5.0, 0.6, 0.6)


def make_skull(
    n: int = 10,
    size: int = 96,
    thickness_mm: float = 6.0,
    inner: float = 34.0,
    step_deg: tuple[float, float] | None = None,
    step_mm: float = 0.0,
    line_deg: tuple[float, float] | None = None,
    line_width_mm: float = 1.5,
    line_depth_mm: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """A smooth elliptical vault, optionally with a depressed step or a lucent line.

    Geometry is in millimetres so a test can ask for a gap of a given width:
    the detector's whole job is separating a thin fracture line from a wide
    partial-volume edge, and that distinction is only expressible in mm.

    Returns the volume and a boolean volume marking where the lesion was drawn,
    so a test can check the detector pointed at it.
    """
    sp = SPACING[1]
    thickness_px = thickness_mm / sp
    a, b = size * 0.40, size * 0.34  # outer-table semi-axes, px
    cy = cx = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size]
    dy, dx = yy - cy, xx - cx
    d = np.hypot(dy, dx)
    th = np.arctan2(dy, dx)
    sin_t, cos_t = np.sin(th), np.cos(th)
    # pixel radius of the outer table along each direction (an ellipse, not a circle)
    outer = 1.0 / np.sqrt(sin_t**2 / a**2 + cos_t**2 / b**2)
    inner_table = outer - thickness_px
    ang = (np.degrees(th) + 360.0) % 360.0

    vol = np.full((n, size, size), 10.0, np.float32)
    ring = (d >= inner_table) & (d <= outer)
    vol[:, ring] = 1200.0
    vol[:, d < inner_table] = inner
    lesion = np.zeros((n, size, size), bool)

    if step_deg is not None:
        lo, hi = step_deg
        sector = (ang >= lo) & (ang < hi)
        a2 = a - step_mm / sp  # the outer table steps inward
        outer2 = 1.0 / np.sqrt(sin_t**2 / a2**2 + cos_t**2 / b**2)
        old = sector & ring
        vol[:, old] = 10.0
        new = sector & (d >= outer2 - thickness_px) & (d <= outer2)
        vol[:, new] = 1200.0
        lesion[:, old & ~new] = True

    if line_deg is not None:
        lo, hi = line_deg
        sector = (ang >= lo) & (ang < hi)
        depth_px = line_depth_mm / sp
        width_px = line_width_mm / sp
        gap = (
            sector
            & (d >= inner_table + depth_px)
            & (d < inner_table + depth_px + width_px)
        )
        vol[:, gap] = 300.0  # lucent, but above air and below bone
        lesion[:, gap] = True
    return vol, lesion


def make_series(vol: np.ndarray, study_id: str = "skull") -> CTSeries:
    return CTSeries(volume=vol, study_id=study_id, spacing=SPACING)


# ----------------------------------------------------------------- helpers


def test_circular_runs_finds_one_run_and_wraps():
    flag = np.zeros(180, bool)
    flag[170:180] = True
    flag[0:10] = True  # same sector, split across the array boundary
    runs = _circular_runs(flag, min_len=5)
    assert len(runs) == 1, "a sector crossing 0 degrees is still one sector"
    (a, b) = runs[0]
    assert a == 170 and b == 10


def test_circular_runs_drops_runs_below_the_minimum():
    flag = np.zeros(180, bool)
    flag[50:53] = True  # 3 angles, below a 12-degree line
    assert _circular_runs(flag, min_len=10) == []


def test_circular_runs_on_an_all_true_ring_is_one_run():
    assert len(_circular_runs(np.ones(180, bool), min_len=1)) == 1


def test_circular_smooth_keeps_both_ends_continuous():
    p = np.sin(np.linspace(0, 2 * np.pi, 180, endpoint=False))  # no duplicated endpoint
    s = _circular_smooth(p, 9)
    assert np.isfinite(s).all()
    assert np.std(s) < np.std(p), "smoothing cannot add structure"
    # the seam must be an ordinary step, not a jump: compare it with the largest
    # step anywhere else in the smoothed signal
    step_max = float(np.abs(np.diff(s)).max())
    assert abs(s[0] - s[-1]) <= step_max * 1.5, "a wrapped average must not step at the seam"


def test_fill_angles_interpolates_a_gap_in_the_ring():
    p = np.linspace(10.0, 20.0, 180)
    p[80:90] = np.nan
    out = _fill_angles(p)
    assert np.isfinite(out).all()
    assert 13.0 < out[85] < 17.0


def test_fill_angles_gives_up_when_nothing_is_bone():
    assert np.isnan(_fill_angles(np.full(180, np.nan))).all()


def test_segment_counts_separates_runs():
    row = np.array([[0, 1, 1, 0, 1, 0, 0, 1]])
    assert _segment_counts(row.astype(bool))[0] == 3


def test_internal_gaps_measures_a_dark_gap_between_bone():
    bone = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0], bool)
    # two dark samples at a 0.3 mm step == a 0.6 mm gap
    gaps = _internal_gaps(bone, step_mm=0.3)
    assert gaps == [pytest.approx(0.6)]
    # the same gap at a finer step is wider, not the same: units must travel
    assert _internal_gaps(bone, step_mm=0.15) == [pytest.approx(0.3)]


def test_internal_gaps_is_empty_for_solid_bone():
    bone = np.ones(10, bool)
    assert _internal_gaps(bone, step_mm=0.3) == []


# ----------------------------------------------------------------- one slice


def test_clean_skull_gives_no_findings():
    """The property that matters: silence on a normal vault."""
    vol, _ = make_skull()
    res = analyze_slice(vol[0], (SPACING[1], SPACING[2]))
    assert res["ok"], res
    assert res["findings"] == [], "a smooth ring must not raise a fracture"
    assert res["n_debris"] == 0
    assert res["n_ray_segments_median"] == 1, "a ray crosses an intact vault once"
    # the median ray radius of an ellipse with semi-axes 38.4 and 32.6 px
    assert res["vault_radius_mm"] == pytest.approx(21.0, abs=2.0)


def test_lucent_line_is_found_on_the_sector_that_has_one():
    vol, lesion = make_skull(line_deg=(200.0, 240.0))
    res = analyze_slice(vol[0], (SPACING[1], SPACING[2]))
    assert res["ok"]
    kinds = {f["kind"] for f in res["findings"]}
    assert "lucent_line" in kinds, "a thin dark gap through the bone is a fracture line"
    found = [f for f in res["findings"] if f["kind"] == "lucent_line"]
    assert any(f["angle_start_deg"] <= 215.0 < f["angle_end_deg"] for f in found), (
        "the reported sector must cover the drawn line"
    )
    assert all(f["run_deg"] >= LUCENT_MIN_RUN_DEG for f in found)
    assert lesion.any(), "the fixture must actually draw a line"


def test_displaced_step_is_reported_with_its_depth():
    vol, lesion = make_skull(step_deg=(90.0, 140.0), step_mm=6.0)
    res = analyze_slice(vol[0], (SPACING[1], SPACING[2]))
    assert res["ok"]
    steps = [f for f in res["findings"] if f["kind"] in ("depressed", "elevated")]
    assert steps, "an inward step in the outer table is a depressed fracture"
    assert any(f["angle_start_deg"] <= 110.0 < f["angle_end_deg"] for f in steps)
    assert max(f["step_mm"] for f in steps) > 3.0, "the step must be measured in mm"
    assert lesion.any(), "the fixture must actually move the outer table"


def test_a_wide_gap_is_not_called_a_fracture_line():
    """A partial-volume hole or a sinus is wide; a fracture line is thin.

    The same sector drawn 4 mm wide must not be reported as a fracture line, or
    every sinus and every beam-hardening edge becomes a fracture.
    """
    assert GAP_MIN_MM < GAP_MAX_MM
    # a 9 mm thick vault so a 6 mm gap still *splits* the bone: without the width
    # guard this would read as a fracture line, with it the split is ignored
    vol, _ = make_skull(thickness_mm=9.0, line_deg=(200.0, 240.0), line_width_mm=6.0)
    res = analyze_slice(vol[0], (SPACING[1], SPACING[2]))
    assert res["ok"]
    kinds = {f["kind"] for f in res.get("findings", [])}
    assert "lucent_line" not in kinds, f"a 6 mm gap is not a fracture line: {kinds}"
    thin, _ = make_skull(thickness_mm=9.0, line_deg=(200.0, 240.0), line_width_mm=1.5)
    thin_kinds = {f["kind"] for f in analyze_slice(thin[0], (SPACING[1], SPACING[2]))["findings"]}
    assert "lucent_line" in thin_kinds, "the same vault with a 1.5 mm gap is one"


def test_midline_sector_is_guarded_as_a_suture():
    """The sagittal suture is lucent by nature and must not read as a fracture."""
    vol, _ = make_skull(line_deg=(356.0, 4.0))  # straddles 0 degrees
    res = analyze_slice(vol[0], (SPACING[1], SPACING[2]))
    assert res["ok"]
    assert res["findings"] == [], "a lucent line on the midline is the suture"


def test_slices_without_a_vault_are_skipped_not_guessed():
    vol = np.full((4, 64, 64), 10.0, np.float32)  # no bone at all
    res = analyze_slice(vol[0], (0.6, 0.6))
    assert res["ok"] is False
    assert res["reason"] == "no bone"


def test_slice_dominated_by_bone_is_skipped():
    vol = np.full((4, 64, 64), 1200.0, np.float32)  # everything is bone
    res = analyze_slice(vol[0], (0.6, 0.6))
    assert res["ok"] is False
    assert res["reason"] == "slice dominated by bone"


def test_skull_base_geometry_is_skipped_not_guessed():
    """A ray at the base legitimately meets the petrous bone twice."""
    vol, _ = make_skull()
    plane = vol[0].copy()
    cy, cx = plane.shape[0] / 2, plane.shape[1] / 2
    plane[int(cy) - 3 : int(cy) + 3, int(cx) - 12 : int(cx) + 12] = 1200.0  # midline bone
    res = analyze_slice(plane, (0.6, 0.6))
    assert res["ok"] is False
    assert res["reason"] == "ray crosses bone more than once"


# ----------------------------------------------------------------- a study


def test_a_normal_study_is_not_flagged():
    vol, _ = make_skull()
    out = skull_fractures(make_series(vol))
    assert out["n_candidate_lines"] == 0
    assert out["n_debris"] == 0
    assert out["flag"] is False


def test_a_fracture_running_through_the_volume_is_flagged_once():
    vol, _ = make_skull(line_deg=(200.0, 240.0))
    out = skull_fractures(make_series(vol))
    assert out["flag"] is True
    assert out["n_candidate_lines"] >= 1
    line = out["lines"][0]
    assert line["z_start"] == 0 and line["z_end"] == vol.shape[0] - 1
    assert line["n_slices"] == vol.shape[0], "one line, not one per slice"
    assert "lucent_line" in line["kinds"]


def test_a_single_slice_artifact_is_not_a_fracture():
    """Evidence that stops after one slice is a notch, not a line."""
    vol, _ = make_skull()
    vol[3, 30:70, 60:66] = 10.0  # a nick in the ring, one slice only
    out = skull_fractures(make_series(vol))
    assert out["n_candidate_lines"] == 0
    assert out["flag"] is False


def test_debris_next_to_the_vault_is_reported():
    """A free fragment just outside the outer table, i.e. in the subdural space."""
    vol, _ = make_skull()
    z = vol.shape[0] // 2
    outer_row = vol.shape[1] / 2 - vol.shape[1] * 0.40  # 47.5 - 38.4 = ~9
    vol[z, round(outer_row) - 6 : round(outer_row) - 2, 44:50] = 1200.0  # just outside
    out = skull_fractures(make_series(vol))
    assert out["n_debris"] >= 1, out
    assert out["flag"] is True


def test_far_away_bone_is_not_debris():
    vol, _ = make_skull()
    vol[4, 0:6, 0:6] = 1200.0  # dense but nowhere near the vault
    out = skull_fractures(make_series(vol))
    assert out["n_debris"] == 0


def test_study_reports_how_many_slices_it_could_judge():
    vol, _ = make_skull(n=6)
    vol[0] = 10.0  # an air slice above the vertex
    out = skull_fractures(make_series(vol))
    assert out["n_slices_evaluated"] + out["n_slices_skipped"] == 6
    assert "no bone" in out["skip_reasons"]


def test_group_lines_requires_consecutive_slices():
    results = [
        (0, {"findings": [{"kind": "lucent_line", "angle_start_deg": 0.0, "angle_end_deg": 30.0}]}),
        (2, {"findings": [{"kind": "lucent_line", "angle_start_deg": 0.0, "angle_end_deg": 30.0}]}),
    ]
    lines = group_lines(results)
    assert len(lines) == 2, "a gap in z breaks the line"
    assert all(ln["n_slices"] == 1 for ln in lines)


def test_group_lines_merges_the_same_sector():
    results = [
        (z, {"findings": [{"kind": "lucent_line", "angle_start_deg": 200.0, "angle_end_deg": 240.0}]})
        for z in range(4)
    ]
    lines = group_lines(results)
    assert len(lines) == 1
    assert lines[0]["n_slices"] == 4 and lines[0]["z_start"] == 0 and lines[0]["z_end"] == 3


def test_group_lines_merges_kinds_on_one_line():
    results = [
        (0, {"findings": [{"kind": "lucent_line", "angle_start_deg": 0.0, "angle_end_deg": 40.0}]}),
        (1, {"findings": [{"kind": "depressed", "angle_start_deg": 0.0, "angle_end_deg": 40.0,
                           "step_mm": 4.0}]}),
    ]
    lines = group_lines(results)
    assert lines[0]["kinds"] == ["depressed", "lucent_line"]
    assert lines[0]["max_step_mm"] == 4.0


# ----------------------------------------------------------------- hematoma pairing


def test_blood_next_to_the_vault_is_flagged_as_the_epidural_pairing():
    vol, _ = make_skull()
    mask = np.zeros(vol.shape, bool)
    cy = vol.shape[1] // 2
    mask[:, cy - 4 : cy + 4, 44:50] = True  # blood hugging the inner table
    out = associate_with_hematoma(make_series(vol), mask)
    assert out["adjacent"] is True
    assert out["n_adjacent_slices"] == vol.shape[0]
    assert out["min_distance_mm"] < 15.0


def test_parenchymal_blood_is_farther_from_bone_than_subdural_blood():
    """The signal is the *difference*: deep blood is much farther from the vault."""
    vol, _ = make_skull()
    cy = vol.shape[1] // 2
    deep = np.zeros(vol.shape, bool)
    deep[:, cy - 3 : cy + 3, cy - 3 : cy + 3] = True  # central, inside the brain
    near = np.zeros(vol.shape, bool)
    inner_table_col = round((vol.shape[1] - 1) / 2.0 + (vol.shape[2] * 0.34 - 10.0))
    near[:, cy - 4 : cy + 4, inner_table_col - 5 : inner_table_col] = True
    st = make_series(vol)
    d_deep = associate_with_hematoma(st, deep)["min_distance_mm"]
    d_near = associate_with_hematoma(st, near)["min_distance_mm"]
    assert d_deep > d_near * 1.5, f"deep={d_deep} near={d_near}"
    # this phantom's whole brain is within 15 mm of the vault, so the default
    # threshold is exercised on a smaller one to separate the two cases
    assert associate_with_hematoma(st, near, max_distance_mm=8.0)["adjacent"] is True
    assert associate_with_hematoma(st, deep, max_distance_mm=8.0)["adjacent"] is False


def test_no_mask_means_no_pairing():
    vol, _ = make_skull()
    out = associate_with_hematoma(make_series(vol), np.zeros(vol.shape, bool))
    assert out == {"n_adjacent_slices": 0, "min_distance_mm": None, "adjacent": False}


def test_hematoma_pairing_needs_bone_to_measure_against():
    vol = np.full((3, 64, 64), 10.0, np.float32)
    mask = np.zeros(vol.shape, bool)
    mask[:, 20:30, 20:30] = True
    out = associate_with_hematoma(make_series(vol), mask)
    assert out["adjacent"] is False and out["min_distance_mm"] is None


def test_threshold_is_above_the_soft_tissue_floor():
    """Sanity: bone above HU_BONE_MIN is what every measurement keys on."""
    assert HU_BONE_MIN > 300.0
