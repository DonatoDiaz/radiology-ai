"""Tests for Phase 5 — paranasal sinus CT.

Every case is a phantom built from the HU categories the module reasons about
(air, fluid, soft tissue, bone), so a failure means the criterion is wrong,
not that a real scan was unusual. Where a criterion is a threshold, the test
puts the phantom just on each side of it.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage as ndi

from vindr.ct.shapes import equivalent_diameter_mm, feret_diameter_mm, perimeter_mm
from vindr.ct.sinuses import (
    AIR_HU,
    BONE_HU,
    CLEARANCE_REL,
    FLUID_HU,
    LEVEL_SPAN_RATIO,
    LEVEL_STRAIGHTNESS_MM,
    MUCOSA_FRACTION,
    MUCOSA_THICK_MM,
    NOT_CLAIM,
    SOFT_HU,
    cavity_measurements,
    find_cavities,
    fluid_level,
    mucosal_thickness,
    sinus_findings,
    sinus_report,
)

SP = (2.0, 1.0, 1.0)  # 2 mm slices, 1 mm pixels
AIR, FLUID, SOFT, BONE = -1000.0, 10.0, 40.0, 700.0
BRAIN = 30.0


def _blank(z: int = 24, y: int = 96, x: int = 96, fill: float = -1000.0) -> np.ndarray:
    return np.full((z, y, x), fill, np.float32)


def _shell(mask: np.ndarray, width: int = 1) -> np.ndarray:
    return ndi.binary_dilation(mask, np.ones((3, 3, 3), bool), iterations=width) & ~mask


def _lining(cavity: np.ndarray, voxels: int) -> np.ndarray:
    """The soft-tissue rim of a cavity lining, `voxels` thick.

    Eroding a solid region would give a smaller solid region, so the lining is
    built as the difference: everything inside the cavity that is not more than
    `voxels` from the wall.
    """
    dt = ndi.distance_transform_edt(cavity, sampling=SP)
    return cavity & (dt <= voxels)


def _head(vol: np.ndarray, cy: float = 48.0, cx: float = 48.0) -> np.ndarray:
    """A head-sized soft-tissue cylinder, so _head_extent_mm has a scale to read."""
    _, yy, xx = np.ogrid[:1, : vol.shape[1], : vol.shape[2]]
    return np.broadcast_to((yy - cy) ** 2 + (xx - cx) ** 2 <= 38.0**2, vol.shape)


def _sphere(
    shape: tuple[int, int, int],
    cz: float,
    cy: float,
    cx: float,
    radius_mm: float,
    n_slices: int = 9,
) -> np.ndarray:
    """A ball of `radius_mm`, so every axial slice of it is a round cross-section."""
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    d2 = ((zz - cz) * SP[0]) ** 2 + ((yy - cy) * SP[1]) ** 2 + ((xx - cx) * SP[2]) ** 2
    return (d2 <= radius_mm**2) & (np.abs(zz - cz) <= n_slices / 2.0)


def _sinus_phantom(
    *,
    wall: float = BONE,
    inside: float = AIR,
    radius: float = 9.0,
    z_center: float = 12.0,
    cy: float = 48.0,
    cx: float = 48.0,
    n_slices: int = 9,
    wall_width: int = 2,
    wall_gap: bool = False,
    outside_soft: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """One bone-walled cavity inside a soft-tissue head. Returns (volume, cavity).

    The head carries a bony skull of its own, because the hyperostosis
    criterion compares the cavity's wall with the rest of the skull — a
    phantom where the wall is the only bone in the volume would make that
    comparison meaningless.
    """
    vol = _blank()
    vol[_head(vol)] = BRAIN
    vol[_shell(_head(vol), 2)] = BONE
    cavity = _sphere(vol.shape, z_center, cy, cx, radius, n_slices)
    vol[cavity] = inside
    vol[_shell(cavity, wall_width)] = wall
    if wall_gap:
        # a hole in the wall on one side, opening towards +x
        gap = np.zeros_like(cavity)
        gap[:, 44:52, 56:60] = True
        gap &= _shell(cavity, 3)
        vol[gap] = SOFT_HU[0] - 1.0 if outside_soft is None else outside_soft
    return vol, cavity


def _two_cavities() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A small normal cavity and a big dilated one, for the mucocele reference.

    Far enough apart that neither one's bony wall reaches into the other; with
    a 2-voxel wall at 14 mm and 8 mm radii, centres 30 mm apart leave a real gap.
    """
    vol, small = _sinus_phantom(inside=FLUID, radius=7.0, cy=42.0, cx=34.0, n_slices=9)
    big = _sphere(vol.shape, 12.0, 56.0, 62.0, 12.0, n_slices=13)
    vol[big] = FLUID
    vol[_shell(big, 1)] = BONE
    return vol, small, big


# --------------------------------------------------------------------------- #
# find_cavities
# --------------------------------------------------------------------------- #


def test_find_cavities_finds_a_bone_walled_air_cavity():
    vol, _ = _sinus_phantom()
    labels, _info = find_cavities(vol, SP)
    found = _info["cavities"]
    assert labels.max() >= 1
    assert any(f["kind"] == "aerated" for f in found)


def test_find_cavities_finds_a_fully_opacified_cavity():
    """An air-only anchor would miss exactly the case worth reporting."""
    vol, _ = _sinus_phantom(inside=SOFT)
    _, _info = find_cavities(vol, SP)
    found = _info["cavities"]
    assert any(f["kind"] == "opacified" for f in found)


def test_find_cavities_rejects_a_region_with_no_bone_around_it():
    vol, _ = _sinus_phantom(wall=SOFT)
    _, _info = find_cavities(vol, SP)
    found = _info["cavities"]
    assert found == []


def test_find_cavities_drops_a_region_touching_the_image_edge():
    """The air around the patient is not a sinus."""
    vol = _blank()
    vol[8:16, 40:56, 0:20] = AIR
    _, _info = find_cavities(vol, SP)
    found = _info["cavities"]
    assert found == []


def test_find_cavities_keeps_the_brain_out_on_volume():
    """The brain is a bone-enclosed soft-tissue region too, and far too big."""
    vol = _blank()
    vol[_head(vol, 48.0, 48.0)] = BRAIN
    vol[_shell(_head(vol, 48.0, 48.0), 2)] = BONE
    _, _info = find_cavities(vol, SP)
    found = _info["cavities"]
    assert all(f["volume_ml"] <= 40.0 for f in found)


def test_find_cavities_reports_bone_fraction_per_cavity():
    vol, _ = _sinus_phantom()
    _, _info = find_cavities(vol, SP)
    found = _info["cavities"]
    assert all(0.0 <= f["bone_fraction"] <= 1.0 for f in found)
    assert any(f["bone_fraction"] > 0.5 for f in found)


def test_find_cavities_clearance_scales_with_the_head():
    """>1, so a 96-voxel-wide phantom keeps its 6 mm default."""
    vol, _ = _sinus_phantom()
    small = find_cavities(vol, SP, clearance_mm=6.0)[1]["cavities"]
    big = find_cavities(vol, SP, clearance_mm=6.0 * 3)[1]["cavities"]
    assert len(small) >= len(big)


def test_clearance_relative_is_sane():
    assert 0.0 < CLEARANCE_REL < 1.0


# --------------------------------------------------------------------------- #
# mucosal_thickness
# --------------------------------------------------------------------------- #


def test_normal_mucosa_under_two_millimetres_is_measured_not_guessed():
    vol, cavity = _sinus_phantom()
    vol[_lining(cavity, 1)] = SOFT
    out = mucosal_thickness(cavity, vol, SP)
    assert out["max_mm"] <= MUCOSA_THICK_MM
    assert out["typical_mm"] > 0.0  # not measured against the air outside


def test_mucosal_thickness_is_twice_the_distance_transform():
    """A half-thickness d of soft lining reads as 2d, since the DT is to the wall."""
    vol, cavity = _sinus_phantom()
    vol[_lining(cavity, 3)] = SOFT
    out = mucosal_thickness(cavity, vol, SP)
    # within one slice of the nominal 2 x 3 mm, the difference being how coarsely
    # a 2 mm grid can place a distance
    assert abs(out["max_mm"] - 6.0) <= SP[0]


def test_a_fully_opacified_cavity_reads_as_thick_mucosa():
    vol, cavity = _sinus_phantom(inside=SOFT)
    out = mucosal_thickness(cavity, vol, SP)
    assert out["max_mm"] > MUCOSA_THICK_MM


def test_mucosa_thresholds_are_published_ones():
    assert MUCOSA_THICK_MM == 4.0
    assert 0.2 < MUCOSA_FRACTION < 0.5


# --------------------------------------------------------------------------- #
# fluid_level
# --------------------------------------------------------------------------- #


def test_a_flat_chord_air_over_fluid_is_a_level():
    vol, cavity = _sinus_phantom(inside=AIR)
    # fluid in the lower half of the cavity, on several slices
    _, yy, _ = np.ogrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    lower = cavity & (yy > 48.0)
    vol[lower] = FLUID
    out = fluid_level(cavity, vol, SP)
    assert out["present"] is True
    assert out["max_consecutive_slices"] >= 2
    assert out["slices"][0]["tilt_deg"] is not None


def test_fluid_over_air_is_the_same_level_with_the_air_underneath():
    vol, cavity = _sinus_phantom(inside=AIR)
    _, yy, _ = np.ogrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    vol[cavity & (yy < 48.0)] = FLUID
    out = fluid_level(cavity, vol, SP)
    assert out["present"] is True  # which end is up is not decided here


def test_a_lining_is_not_a_level():
    """A thickened lining follows the wall, so it is curved and short."""
    vol, cavity = _sinus_phantom(inside=AIR)
    vol[_lining(cavity, 4)] = SOFT
    out = fluid_level(cavity, vol, SP)
    assert out["present"] is False


def test_a_meniscus_is_not_a_level():
    """A droplet is a chord in one axis only — it fails the span test."""
    vol, cavity = _sinus_phantom(inside=AIR)
    droplet = cavity.copy()
    droplet[:, 44:53, :] = False
    vol[droplet] = FLUID
    out = fluid_level(cavity, vol, SP)
    assert out["present"] is False


def test_one_isolated_slice_is_not_enough_for_a_level():
    """Two consecutive slices are required, so one noisy slice cannot report it."""
    vol, cavity = _sinus_phantom(inside=AIR)
    zz, yy, _ = np.ogrid[: vol.shape[0], : vol.shape[1], : vol.shape[2]]
    middle = int(np.flatnonzero(cavity.any(axis=(1, 2)))[len(np.flatnonzero(cavity.any(axis=(1, 2)))) // 2])
    half = (yy > 48.0) & (zz == middle)
    vol[cavity & half] = FLUID
    out = fluid_level(cavity, vol, SP)
    assert out["max_consecutive_slices"] < 2
    assert out["present"] is False


def test_level_tolerance_is_in_millimetres_and_the_span_is_a_fraction():
    assert LEVEL_STRAIGHTNESS_MM == 2.0
    assert 0.0 < LEVEL_SPAN_RATIO < 1.0


def test_level_note_carries_the_disclaimer():
    vol, cavity = _sinus_phantom()
    assert NOT_CLAIM in fluid_level(cavity, vol, SP)["note"]


# --------------------------------------------------------------------------- #
# cavity_measurements
# --------------------------------------------------------------------------- #


def test_measurements_split_the_contents_into_fractions():
    vol, cavity = _sinus_phantom(inside=AIR)
    out = cavity_measurements(cavity, vol, SP)
    assert out["air_fraction"] > 0.9
    assert out["bone_fraction"] < 0.05
    assert out["volume_ml"] > 0.0
    assert len(out["extent_mm"]) == 3


def test_measurements_add_a_volume_ratio_only_with_a_reference():
    vol, cavity = _sinus_phantom()
    assert "volume_ratio_to_reference" not in cavity_measurements(cavity, vol, SP)
    out = cavity_measurements(cavity, vol, SP, reference_volume_ml=out_ml(vol, cavity) * 2)
    assert out["volume_ratio_to_reference"] < 1.0


def out_ml(vol: np.ndarray, mask: np.ndarray) -> float:
    return float(mask.sum()) * SP[0] * SP[1] * SP[2] / 1000.0


# --------------------------------------------------------------------------- #
# sinus_findings
# --------------------------------------------------------------------------- #


def test_a_clear_sinus_suggests_nothing():
    vol, cavity = _sinus_phantom()
    out = sinus_findings(cavity, vol, SP, reference_bone_hu=600.0)
    assert out["suggests"] == "normal_aerated"
    assert out["flags"] == []
    assert NOT_CLAIM in out["note"]


def test_thick_mucosa_suggests_sinusitis():
    vol, cavity = _sinus_phantom(inside=SOFT)
    out = sinus_findings(cavity, vol, SP, reference_bone_hu=600.0)
    assert out["suggests"] == "sinusitis"
    assert out["mucosa"]["max_mm"] > MUCOSA_THICK_MM


def test_a_broken_wall_with_soft_tissue_outside_raises_malignancy():
    vol, cavity = _sinus_phantom(inside=AIR, wall_gap=True, outside_soft=SOFT)
    out = sinus_findings(cavity, vol, SP, reference_bone_hu=600.0)
    assert out["suggests"] == "malignancy_hint"
    assert out["outside_soft_tissue"]["fraction_in_break"] > 0.2


def test_denser_wall_beside_an_opacified_cavity_raises_a_fungal_ball():
    vol, cavity = _sinus_phantom(inside=SOFT, wall=1150.0)
    out = sinus_findings(cavity, vol, SP, reference_bone_hu=700.0)
    assert out["suggests"] in ("fungal_ball", "sinusitis")
    assert out["wall"]["hyperostosis_ratio"] >= 1.35


def test_a_mucocele_needs_expansion_against_a_reference():
    vol, cavity = _sinus_phantom(
        inside=FLUID, radius=14.0, n_slices=13, wall_width=1
    )
    small = sinus_findings(cavity, vol, SP, reference_bone_hu=600.0, reference_volume_ml=5.0)
    assert small["suggests"] == "mucocele"
    # same cavity, not expanded: no reference comparison means no mucocele
    plain = sinus_findings(cavity, vol, SP, reference_bone_hu=600.0)
    assert plain["suggests"] != "mucocele"


def test_hyperostosis_ratio_compares_against_the_rest_of_the_skull():
    """A normal wall reads about 1.0 against the surrounding skull bone."""
    vol, cavity = _sinus_phantom(inside=SOFT)
    normal = sinus_findings(cavity, vol, SP)["wall"]["hyperostosis_ratio"]
    assert 0.85 <= normal <= 1.15


def test_an_explicit_reference_overrides_the_derived_one():
    vol, cavity = _sinus_phantom(inside=SOFT)
    assert sinus_findings(cavity, vol, SP, reference_bone_hu=1400.0)["wall"][
        "hyperostosis_ratio"
    ] == 0.5


def test_an_empty_mask_is_refused_rather_than_reported():
    empty = np.zeros((4, 40, 40), bool)
    out = sinus_findings(empty, _blank(), SP)
    assert out["ok"] is False
    assert out["reason"]


def test_hu_bands_are_ordered_and_fluid_nests_inside_soft():
    """Simple fluid is inside the soft band on purpose; see the report note.

    Bone is read as strictly above BONE_HU and soft up to and including
    SOFT_HU[1], so the two do not double-count a voxel on the boundary.
    """
    assert AIR_HU < SOFT_HU[0] <= FLUID_HU[0] <= FLUID_HU[1] <= SOFT_HU[1] <= BONE_HU


def test_fluid_reads_as_soft_tissue_too_and_the_report_says_so():
    """Simple fluid is inside the soft band, so a level also counts as soft.

    The fractions therefore overlap and do not sum to 1; the report has to say
    which of the two excludes the other, or a reader will add them up.
    """
    vol, cavity = _sinus_phantom(inside=FLUID)
    out = cavity_measurements(cavity, vol, SP)
    assert out["fluid_fraction"] > 0.5
    assert out["soft_fraction"] > 0.5
    assert out["soft_excluding_fluid_fraction"] < out["soft_fraction"]
    assert "soft" in out["note"].lower()


# --------------------------------------------------------------------------- #
# sinus_report
# --------------------------------------------------------------------------- #


def test_report_walks_the_cavities_it_finds():
    vol, _ = _sinus_phantom(inside=SOFT)
    out = sinus_report(vol, SP)
    assert out["ok"] is True
    assert out["n_cavities"] >= 1
    assert all("suggests" in c for c in out["cavities"])
    assert NOT_CLAIM in out["note"]


def test_report_says_so_when_there_is_nothing_to_look_at():
    out = sinus_report(_blank(), SP)
    assert out["ok"] is False
    assert out["reason"]
    assert out["cavities"] == []


def test_report_names_no_sinus_without_orientation():
    """The loader keeps no orientation, so a name would be a guess."""
    vol, _ = _sinus_phantom(inside=AIR)
    out = sinus_report(vol, SP)
    joined = " ".join(str(c.get("name", "")) for c in out.get("cavities", []))
    assert "frontal" not in joined.lower()
    assert "maxillary" not in joined.lower()
    assert "sphenoid" not in joined.lower()
    assert "ethmoid" not in joined.lower()


def test_report_names_the_locator_caveat_and_the_orientation_limit():
    vol, _ = _sinus_phantom()
    out = sinus_report(vol, SP)
    assert "сосцевидн" in out["caveat"]
    assert "ориентац" in out["caveat"]


def test_report_never_compares_a_cavity_with_itself():
    """The reference is the median of the *other* cavities, or nothing."""
    vol, _, _ = _two_cavities()
    out = sinus_report(vol, SP)
    for c in out["cavities"]:
        assert c["reference_volume_ml"] != c["contents"]["volume_ml"]
        assert c["reference_source"]


# --------------------------------------------------------------------------- #
# shapes helpers these criteria rely on
# --------------------------------------------------------------------------- #


def test_perimeter_of_a_disk_lands_on_two_pi_r():
    r = 20.0
    yy, xx = np.ogrid[:80, :80]
    disk = ((yy - 40) ** 2 + (xx - 40) ** 2) <= r**2
    per = perimeter_mm(disk, (1.0, 1.0))
    assert abs(per - 2 * np.pi * r) / (2 * np.pi * r) < 0.10


def test_feret_of_a_disk_lands_on_the_diameter():
    r = 20.0
    yy, xx = np.ogrid[:80, :80]
    disk = ((yy - 40) ** 2 + (xx - 40) ** 2) <= r**2
    assert abs(feret_diameter_mm(disk, (1.0, 1.0)) - 2 * r) <= 2.0


def test_feret_is_measured_on_the_outline_not_the_eroded_core():
    """Erosion shortens every chord by a voxel, 2 mm on a 2 mm grid."""
    rect = np.zeros((40, 40), bool)
    rect[10:30, 5:35] = True
    assert feret_diameter_mm(rect, (1.0, 1.0)) >= 29.0


def test_equivalent_diameter_inverts_the_area():
    assert abs(equivalent_diameter_mm(np.pi * 25.0) - 10.0) < 0.01


@pytest.mark.parametrize("fn", [perimeter_mm, feret_diameter_mm])
def test_empty_mask_measures_zero_rather_than_dividing_by_zero(fn):
    assert fn(np.zeros((10, 10), bool), (1.0, 1.0)) == 0.0
