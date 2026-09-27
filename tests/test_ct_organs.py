"""Tests for Phase 4 organ measurements."""

import numpy as np
import pytest

from vindr.ct.organs import (
    adrenal_findings,
    aorta_findings,
    aorta_mask,
    body_mask,
    contrast_hu_guess,
    equivalent_diameter_mm,
    feret_diameter_mm,
    kidney_findings,
    lung_mask,
    lymph_node_findings,
    mediastinal_mass_findings,
    short_axis_mm,
    spine_mask,
    vessel_encasement,
    volume_ml,
    washout,
)

SPACING = (2.0, 2.0, 2.0)  # 2 mm grid: 1 px == 2 mm in every direction


def disk(shape_hw, centre, radius_px, z_span=3):
    yy, xx = np.mgrid[0 : shape_hw[0], 0 : shape_hw[1]]
    sl = np.sqrt((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2) <= radius_px
    return np.repeat(sl[None], z_span, axis=0)


def sphere_mask(shape, centre, radius_px):
    zz, yy, xx = np.indices(shape)
    d = np.sqrt((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2)
    return d <= radius_px


# --- geometry primitives --------------------------------------------------- #


def test_feret_matches_the_diameter_of_a_disk():
    # 10 px radius on a 2 mm grid is a 40 mm diameter
    assert feret_diameter_mm(disk((60, 60), (30, 30), 10)[0], (2.0, 2.0)) == pytest.approx(40.0, abs=0.6)


def test_feret_of_nothing_is_zero():
    assert feret_diameter_mm(np.zeros((10, 10), bool), (2.0, 2.0)) == 0.0


def test_feret_separates_two_blobs_but_each_blob_alone_is_smaller():
    sl = disk((60, 60), (15, 30), 5, z_span=1)[0] | disk((60, 60), (45, 30), 5, z_span=1)[0]
    assert feret_diameter_mm(sl, (2.0, 2.0)) > 2 * feret_diameter_mm(disk((60, 60), (15, 30), 5, z_span=1)[0], (2.0, 2.0))


def test_equivalent_diameter_of_a_circle():
    assert equivalent_diameter_mm(np.pi * 100.0) == pytest.approx(20.0)


def test_volume_ml_uses_the_full_spacing():
    assert volume_ml(np.ones((2, 3, 4), bool), (2.0, 2.0, 2.0)) == pytest.approx(2 * 3 * 4 * 8 / 1000.0)


def test_short_axis_is_the_smaller_in_plane_extent():
    # 5 px long in z, 8 px in y, 20 px in x on a 2 mm grid
    m = np.zeros((5, 8, 20), bool)
    m[1:4, 2:6, 4:16] = True
    assert short_axis_mm(m, SPACING) == pytest.approx(8.0)


# --- anchors --------------------------------------------------------------- #


def test_body_mask_fills_the_lungs_as_holes():
    vol = np.full((4, 60, 60), 100.0)
    vol[:, 10:50, 10:50] = -800.0  # lungs inside the body
    vol[:, 0:2, :] = -1000.0  # air around the patient
    body = body_mask(vol)
    assert body[0, 30, 30], "the lung interior counts as body"
    assert not body[0, 0, 0], "air around the patient does not"


def test_lung_mask_keeps_air_inside_the_body_only():
    vol = np.full((4, 60, 60), 100.0)
    vol[:, 10:50, 10:50] = -800.0
    vol[:, 55:60, 55:60] = -1000.0  # a big air pocket outside the body
    lungs = lung_mask(vol, (1.0, 1.0, 1.0), min_ml=5.0)
    assert lungs[:, 30, 30].all(), "lungs are kept"
    assert not lungs[:, 57, 57].any(), "outside air is dropped"
    assert volume_ml(lungs, (1.0, 1.0, 1.0)) == pytest.approx(6.4, rel=0.2)  # 40x40x4 mm


def test_spine_mask_needs_density_and_midline_proximity():
    vol = np.full((4, 80, 80), 100.0)
    vol[:, 38:42, 38:42] = 900.0  # vertebral body, on the midline
    vol[:, 4:8, 70:74] = 900.0  # a rib, dense but far off centre
    sp = spine_mask(vol, SPACING, half_width_mm=45.0)
    assert sp[:, 40, 40].all()
    assert not sp[:, 6, 72].any()


def test_aorta_mask_on_a_contrast_study_finds_the_pool():
    vol = np.full((4, 60, 60), 50.0)  # plain soft tissue, below the window
    vol[:, 38:42, 38:42] = 900.0  # spine
    vol[:, 26:32, 30:36] = 350.0  # enhanced aorta, just left of the spine
    mask, info = aorta_mask(vol, SPACING)
    assert info["ok"] and info["assumed"] == "контрастированная аорта"
    assert mask[:, 29, 33].all() and not mask[:, 40, 40].any(), "the bone is not the aorta"
    assert contrast_hu_guess(vol, mask) == pytest.approx(350.0, abs=1.0)
    assert contrast_hu_guess(vol, np.zeros_like(mask)) is None


def test_aorta_mask_says_so_when_there_is_no_contrast():
    vol = np.full((4, 60, 60), 50.0)  # plain soft tissue, below the window
    vol[:, 38:42, 38:42] = 900.0
    vol[:, 26:32, 30:36] = 45.0  # unenhanced blood pool
    mask, info = aorta_mask(vol, SPACING)
    assert not info["ok"] and "контраст" in info["reason"]
    assert not mask.any()


def test_a_flap_does_not_make_the_locator_lose_half_the_aorta():
    # regression: a hypodense flap falls out of the enhancement window and used
    # to split the vessel, after which keeping the biggest component threw away
    # half the aorta and still reported a plausible caliber
    shape = (6, 60, 60)
    vol = np.full(shape, 60.0)
    vol[:, 28:32, 28:32] = 700.0  # spine
    yy, xx = np.mgrid[0:60, 0:60]
    disc = np.sqrt((yy - 30.0) ** 2 + (xx - 24.0) ** 2) <= 8  # 32 mm across
    vol[:, disc] = 350.0
    vol[1:5, 30, 15:35] = 60.0  # the flap, at 60 HU: below the window
    mask, info = aorta_mask(vol, SPACING)
    assert info["ok"]
    assert mask[:, 30, 24].all(), "the strip must not punch a hole through the mask"
    f = aorta_findings(mask, SPACING, hu_volume=vol)
    assert f["max_diameter_mm"] == pytest.approx(32.0, abs=1.0)
    assert f["flap"]["suspected"]


def test_a_genuinely_split_structure_keeps_both_parts():
    shape = (3, 40, 40)
    vol = np.full(shape, 60.0)
    vol[:, 18:22, 18:22] = 700.0
    for cx in (11, 29):  # two lumens of a bilobed vessel, far enough apart to stay separate
        yy, xx = np.mgrid[0:40, 0:40]
        vol[:, np.sqrt((yy - 20.0) ** 2 + (xx - cx) ** 2) <= 6] = 350.0
    mask, info = aorta_mask(vol, SPACING)
    assert info["ok"] and info["n_parts_kept"] == 2
    assert mask[:, 20, 14].any() and mask[:, 20, 28].any()


def test_aorta_mask_without_a_spine_gives_up():
    vol = np.full((4, 60, 60), 100.0)  # no dense midline structure
    mask, info = aorta_mask(vol, SPACING)
    assert not info["ok"] and not mask.any()


# --- aorta ----------------------------------------------------------------- #


def test_a_normal_aorta_is_not_an_aneurysm():
    f = aorta_findings(disk((80, 80), (40, 40), 5), SPACING)  # 20 mm
    assert f["max_diameter_mm"] == pytest.approx(20.0, abs=0.6)
    assert f["flags"] == []


def test_a_thirty_millimetre_aorta_is_flagged():
    f = aorta_findings(disk((80, 80), (40, 40), 8), SPACING)  # 32 mm
    assert "аневризма" in " ".join(f["flags"])
    assert "крупная" not in " ".join(f["flags"])


def test_a_forty_millimetre_aorta_is_urgent():
    f = aorta_findings(disk((80, 80), (40, 40), 11), SPACING)  # 44 mm
    assert any("крупная аневризма" in x for x in f["flags"])


def test_an_empty_aorta_mask_says_so():
    f = aorta_findings(np.zeros((3, 20, 20), bool), SPACING)
    assert f["flags"] == ["аорта не найдена"] and f["max_diameter_mm"] is None


def test_diameter_by_slice_picks_the_worst_slice():
    m = disk((80, 80), (40, 40), 4, z_span=3).copy()
    m[2] = disk((80, 80), (40, 40), 9, z_span=1)[0]
    f = aorta_findings(m, SPACING)
    assert f["worst_slice"] == 2
    assert f["max_diameter_mm"] == pytest.approx(36.0, abs=0.6)


def test_a_flap_that_splits_the_lumen_is_reported():
    # the dilated lumen is one mask; inside it a low-density band splits it
    # into two enhanced halves, which is the flap sign
    mask = np.zeros((3, 40, 40), bool)
    mask[:, 16:24, 10:30] = True
    hu = np.full((3, 40, 40), 40.0)
    hu[mask] = 350.0
    hu[:, 16:24, 20:24] = 60.0  # the flap: less enhanced than the lumen
    f = aorta_findings(mask, SPACING, hu_volume=hu)
    assert f["flap"]["suspected"]
    assert any("диссекция" in x for x in f["flags"])


def test_a_thin_flap_under_one_percent_is_still_found():
    # a real flap is a strip, not half the lumen
    mask = disk((40, 40), (20, 20), 14, z_span=4)
    hu = np.full(mask.shape, 40.0)
    hu[mask] = 350.0
    hu[:, 20, :] = 60.0  # one pixel across, crossing the whole lumen
    f = aorta_findings(mask, SPACING, hu_volume=hu)
    assert f["flap"]["suspected"]
    assert f["flap"]["max_consecutive_slices"] >= 2


def test_a_flap_on_one_slice_only_is_not_reported():
    mask = disk((40, 40), (20, 20), 14, z_span=4)
    hu = np.full(mask.shape, 40.0)
    hu[mask] = 350.0
    hu[1, 20, :] = 60.0  # a single slice: noise, not a structure
    f = aorta_findings(mask, SPACING, hu_volume=hu)
    assert not f["flap"]["suspected"]


def test_a_single_lumen_is_not_a_flap():
    mask = disk((40, 40), (20, 20), 8)
    hu = np.full(mask.shape, 40.0)
    hu[mask] = 350.0
    f = aorta_findings(mask, SPACING, hu_volume=hu)
    assert not f["flap"]["suspected"]


def test_flat_enhancement_cannot_produce_a_flap():
    mask = disk((40, 40), (20, 20), 8)
    hu = np.full(mask.shape, 350.0)  # no contrast gradient at all
    hu[mask] = 355.0
    f = aorta_findings(mask, SPACING, hu_volume=hu)
    assert not f["flap"]["suspected"]


def test_no_hu_volume_means_no_flap_claim():
    f = aorta_findings(disk((40, 40), (20, 20), 8), SPACING)
    assert "flap" not in f


# --- adrenal --------------------------------------------------------------- #


def adenoma_phantom(mean_hu=5.0, radius_px=5, shape=(6, 40, 40)):
    vol = np.full(shape, 100.0)
    m = sphere_mask(shape, (3, 20, 20), radius_px)
    vol[m] = mean_hu
    return m, vol


def test_a_lipid_rich_lesion_reads_as_an_adenoma():
    m, vol = adenoma_phantom(mean_hu=5.0)
    f = adrenal_findings(m, vol, SPACING)
    assert f["lipid_rich"]
    assert any("аденом" in x for x in f["flags"])


def test_a_soft_tissue_lesion_is_not_called_a_lipid_rich_adenoma():
    m, vol = adenoma_phantom(mean_hu=40.0)
    f = adrenal_findings(m, vol, SPACING)
    assert not f["lipid_rich"]
    assert f["median_hu"] == pytest.approx(40.0, abs=1.0)


def test_washout_is_reported_when_phases_are_given():
    m, vol = adenoma_phantom(mean_hu=40.0, radius_px=9)
    f = adrenal_findings(m, vol, SPACING, hu_by_phase={"plain": 80.0, "portal": 30.0})
    assert f["washout"]["kind"] == "absolute"
    assert f["washout"]["pct"] == pytest.approx(62.5)
    assert f["washout"]["passes_adenoma_cutoff"]


def test_a_large_lesion_is_pushed_off_the_bland_reading():
    m, vol = adenoma_phantom(mean_hu=5.0, radius_px=25)  # 100 mm across
    f = adrenal_findings(m, vol, SPACING)
    assert any("40" in x for x in f["flags"])


def test_a_heterogeneous_lesion_is_pushed_off_the_bland_reading():
    m, vol = adenoma_phantom(mean_hu=5.0)
    vol[m] = np.where(np.arange(m.sum()) % 7 == 0, 80.0, 5.0)
    f = adrenal_findings(m, vol, SPACING)
    assert f["hu_std"] > 25.0
    assert any("неоднород" in x for x in f["flags"])


def test_an_empty_adrenal_mask_says_so():
    assert adrenal_findings(np.zeros((4, 20, 20), bool), np.zeros((4, 20, 20)), SPACING)["ok"] is False


def test_absolute_washout_needs_a_denominator_above_the_noise():
    assert washout({"plain": 12.0, "portal": 5.0}) is None


def test_relative_washout_uses_portal_against_delayed():
    w = washout({"portal": 100.0, "delayed": 50.0}, kind="relative")
    assert w["pct"] == pytest.approx(50.0) and w["passes_adenoma_cutoff"]


def test_washout_of_nothing_is_none():
    assert washout({}) is None
    assert washout({"plain": None, "portal": None}) is None


# --- kidney ---------------------------------------------------------------- #


def kidney_phantom(wall_px=6, outer_px=20, z=6, shape=(6, 60, 60), centre=(30, 30), fluid_hu=None):
    """A kidney as a ring, because 2A/P of a ring is exactly the wall."""
    yy, xx = np.mgrid[0 : shape[1], 0 : shape[2]]
    d = np.sqrt((yy - centre[0]) ** 2 + (xx - centre[1]) ** 2)
    sl = (d <= outer_px) & (d >= outer_px - wall_px)
    mask = np.repeat(sl[None], z, axis=0)
    vol = np.full(mask.shape, 300.0)  # enhanced parenchyma
    if fluid_hu is not None:
        hole = d < outer_px - wall_px + 1
        vol[:, hole] = fluid_hu
    return mask, vol


def test_a_healthy_kidney_raises_no_flags():
    mask, vol = kidney_phantom(wall_px=6, outer_px=20)
    f = kidney_findings(mask, vol, SPACING)
    assert f["cortex_thickness_mm"] == pytest.approx(12.0, abs=2.0)  # 6 px on 2 mm
    assert f["flags"] == []


def test_a_thin_cortex_is_reported():
    mask, vol = kidney_phantom(wall_px=3, outer_px=20)  # 6 mm cortex
    f = kidney_findings(mask, vol, SPACING)
    assert any("истончение" in x for x in f["flags"])


def test_an_asymmetric_cortex_points_at_one_sided_disease():
    thin, vol = kidney_phantom(wall_px=3, outer_px=20)
    normal, _ = kidney_phantom(wall_px=8, outer_px=20)
    f = kidney_findings(thin, vol, SPACING, other_mask3d=normal)
    assert any("пиелонефрит" in x for x in f["flags"])


def test_a_small_kidney_next_to_a_normal_one_is_shrunk():
    small, vol = kidney_phantom(outer_px=12)
    normal, _ = kidney_phantom(outer_px=20)
    f = kidney_findings(small, vol, SPACING, other_mask3d=normal)
    assert any("нефросклероз" in x for x in f["flags"])


def test_equal_enhancement_raises_no_asymmetry_flag():
    mask, vol = kidney_phantom()
    f = kidney_findings(mask, vol, SPACING, portal_hu=180.0, other_portal_hu=182.0)
    assert f["enhancement_delta_hu"] == pytest.approx(2.0)
    assert not any("усилени" in x for x in f["flags"])


def test_a_fall_in_enhancement_is_reported():
    mask, vol = kidney_phantom()
    f = kidney_findings(mask, vol, SPACING, portal_hu=120.0, other_portal_hu=160.0)
    assert any("сниженное усиление" in x for x in f["flags"])


def test_a_dilated_fluid_collecting_system_is_reported():
    mask, vol = kidney_phantom(outer_px=20, fluid_hu=5.0)
    f = kidney_findings(mask, vol, SPACING)
    assert f["collecting_system_fluid_fraction"] > 0.15
    assert any("гидронефроз" in x for x in f["flags"])


def test_a_solid_kidney_has_no_collecting_system_fluid():
    mask, vol = kidney_phantom(outer_px=20)
    f = kidney_findings(mask, vol, SPACING)
    assert f["collecting_system_fluid_fraction"] < 0.15


def test_an_empty_kidney_mask_says_so():
    assert kidney_findings(np.zeros((4, 20, 20), bool), np.zeros((4, 20, 20)), SPACING)["ok"] is False


# --- mediastinum ----------------------------------------------------------- #


def test_node_size_grades_follow_the_published_rules():
    m = np.zeros((6, 40, 40), bool)
    m |= sphere_mask(m.shape, (3, 8, 8), 1)  # 6 mm short axis -> normal
    m |= sphere_mask(m.shape, (3, 32, 34), 3)  # 14 mm -> borderline
    m |= sphere_mask(m.shape, (3, 16, 26), 5)  # 22 mm -> pathologic
    f = lymph_node_findings(m, SPACING, min_voxels=2)
    grades = sorted(n["size_grade"] for n in f["nodes"])
    assert grades == ["borderline", "normal", "pathologic"]
    assert f["n_pathologic"] == 1
    assert f["nodes"][0]["short_axis_mm"] > f["nodes"][-1]["short_axis_mm"], "sorted worst first"


def test_a_single_lumen_graded_normal_is_not_counted_pathologic():
    m = sphere_mask((4, 30, 30), (2, 15, 15), 2)
    f = lymph_node_findings(m, SPACING, min_voxels=2)
    assert f["n_nodes"] == 1 and f["n_pathologic"] == 0
    assert "размер не решает" in f["note"]


def test_specks_are_ignored():
    m = np.zeros((4, 30, 30), bool)
    m[1, 5, 5] = True  # one voxel
    assert lymph_node_findings(m, SPACING, min_voxels=8)["n_nodes"] == 0


def test_a_fluid_mass_reads_as_cystic():
    shape = (6, 40, 40)
    m = sphere_mask(shape, (3, 20, 20), 8)
    vol = np.full(shape, 100.0)
    vol[m] = 8.0
    f = mediastinal_mass_findings(m, vol, SPACING)
    assert f["fluid_fraction"] > 0.9
    assert f["suggests"] == "cyst"


def test_fat_with_a_calcific_focus_reads_as_a_teratoma():
    shape = (6, 40, 40)
    m = sphere_mask(shape, (3, 20, 20), 8)
    vol = np.full(shape, 100.0)
    vol[m] = -60.0  # macroscopic fat
    vol[3, 20, 20] = 400.0
    vol[3, 20, 21] = 400.0  # a 2-voxel calcific focus, enough
    f = mediastinal_mass_findings(m, vol, SPACING)
    assert f["fat_fraction"] > 0.2 and f["calcification_voxels"] >= 2
    assert f["suggests"] == "teratoma"


def test_a_fat_mass_is_not_mistaken_for_a_cyst():
    shape = (6, 40, 40)
    m = sphere_mask(shape, (3, 20, 20), 8)
    vol = np.full(shape, 100.0)
    vol[m] = -60.0  # fat, and fat is not simple fluid
    f = mediastinal_mass_findings(m, vol, SPACING)
    assert f["fluid_fraction"] < 0.1
    assert f["suggests"] == "fat_mass"


def test_a_lone_high_hu_voxel_is_not_a_calcific_focus():
    shape = (6, 40, 40)
    m = sphere_mask(shape, (3, 20, 20), 8)
    vol = np.full(shape, 100.0)
    vol[m] = -60.0
    vol[3, 20, 20] = 400.0  # isolated: not a real calcification
    f = mediastinal_mass_findings(m, vol, SPACING)
    assert f["calcification_voxels"] == 0
    assert f["suggests"] == "fat_mass", "no calcification, so not a teratoma"


def test_a_heterogeneous_soft_tissue_mass_points_at_a_thymoma():
    shape = (6, 40, 40)
    m = sphere_mask(shape, (3, 20, 20), 10)
    vol = np.full(shape, 100.0)
    _, yy, xx = np.indices(shape)
    vol[m] = np.where((yy + xx) % 4 < 2, 30.0, 110.0)[m]
    f = mediastinal_mass_findings(m, vol, SPACING)
    assert f["hu_std"] > 35.0
    assert f["suggests"] == "thymoma"


def test_an_empty_mass_says_so():
    assert mediastinal_mass_findings(np.zeros((3, 10, 10), bool), np.zeros((3, 10, 10)), SPACING)["ok"] is False


def test_a_mass_around_a_vessel_encases_it():
    shape = (4, 40, 40)
    vessel = disk((40, 40), (20, 20), 3, z_span=4)
    mass = sphere_mask(shape, (2, 20, 20), 12) & ~vessel
    r = vessel_encasement(mass, vessel)
    assert r["vessel_fraction_surrounded"] > 0.9
    assert "лимфома" in r["note"]


def test_a_mass_beside_a_vessel_does_not_encase_it():
    shape = (4, 40, 40)
    vessel = disk((40, 40), (12, 12), 3, z_span=4)
    mass = sphere_mask(shape, (2, 30, 30), 6)
    r = vessel_encasement(mass, vessel)
    assert r["vessel_fraction_surrounded"] < 0.5


def test_encasement_needs_both_masks():
    assert vessel_encasement(np.zeros((2, 5, 5), bool), np.ones((2, 5, 5), bool))["ok"] is False
