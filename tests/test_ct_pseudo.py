"""Tests for weakly-supervised pseudo-masks (vindr.ct.pseudo).

The pipeline turns slice-level labels into masks, so the tests pin the parts
that decide whether a pseudo-label is trustworthy: the confidence guards, the
density band, and the fact that a mask comes back on the *native* series grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from vindr.ct import CTSeries, build_head_ct_model
from vindr.ct.pseudo import (
    _native_to_grid_z,
    _resize_mask_to,
    cam_for_slices,
    cam_sharpness,
    cam_to_seed,
    mask_to_volume,
    pseudo_mask_study,
    refine_with_grabcut,
    restrict_to_blood_density,
    write_pseudo_dataset,
)

SPACING = (5.0, 0.8, 0.8)


def make_phantom(n: int = 12, size: int = 64, bleed_slices: range = range(5, 9)) -> tuple[CTSeries, np.ndarray]:
    """A head phantom: skull, brain at 30 HU, a 70 HU bleed on some slices."""
    yy, xx = np.mgrid[0:size, 0:size]
    vol = np.full((n, size, size), -1000.0, np.float32)
    vol[:, yy + xx < 15] = 1000.0
    vol[:, (yy - size // 2) ** 2 + (xx - size // 2) ** 2 < (size // 2 - 2) ** 2] = 30.0
    truth = np.zeros((n, size, size), bool)
    for z in bleed_slices:
        truth[z, (yy - 38) ** 2 + (xx - 28) ** 2 < 60] = True
        vol[z, truth[z]] = 70.0
    return CTSeries(volume=vol, study_id="phantom", spacing=SPACING), truth


# ----------------------------------------------------------------- guards


def test_cam_sharpness_separates_flat_from_focused():
    flat = np.full((32, 32), 0.5, np.float32)
    focused = np.zeros((32, 32), np.float32)
    focused[8:16, 8:16] = 1.0
    assert cam_sharpness(flat) == pytest.approx(1.0, abs=1e-3)
    assert cam_sharpness(focused) > 3.0


def test_cam_to_seed_rejects_a_flat_cam():
    """An untrained net gives a near-uniform CAM; that is not a lesion."""
    assert cam_to_seed(np.full((32, 32), 0.5, np.float32)) is None
    assert cam_to_seed(np.full((32, 32), 0.5, np.float32), min_sharpness=1.5) is None


def test_cam_to_seed_marks_a_definite_core():
    cam = np.zeros((32, 32), np.float32)
    cam[4:12, 4:12] = 1.0
    seed = cam_to_seed(cam, quantile=0.5, min_sharpness=1.0, core_fraction=0.25)
    assert set(np.unique(seed)) <= {0, 1, 2}
    assert (seed >= 2).any(), "the CAM peak must become definite foreground for GrabCut"
    assert (seed > 0).sum() >= (seed >= 2).sum()


def test_cam_to_seed_scales_with_the_lesion():
    """A quantile seed keeps a small bleed on a big slice."""
    cam = np.zeros((64, 64), np.float32)
    cam[30:34, 30:34] = 1.0  # 16 px out of 4096
    small = cam_to_seed(cam, quantile=0.99, min_sharpness=1.0)
    big = cam_to_seed(cam, quantile=0.90, min_sharpness=1.0)
    assert 0 < (small > 0).sum() < (big > 0).sum() < cam.size, "a quantile seed is a pixel budget"


def test_mask_to_volume_keeps_the_z_axis_and_alignment():
    filled = [None, np.zeros((4, 5), bool), np.ones((4, 5), bool), np.ones((4, 5), bool), None]
    vol = mask_to_volume(filled)
    assert vol is not None
    assert vol.shape == (5, 4, 5)  # one plane per input slice, empties preserved
    assert vol[2].all() and not vol[0].any()


def test_mask_to_volume_rejects_isolated_slices():
    assert mask_to_volume([np.ones((4, 4), bool), None, None], min_slices=2) is None
    assert mask_to_volume([None, None], min_slices=1) is None
    assert mask_to_volume([], min_slices=1) is None


def test_restrict_to_blood_density_separates_clot_from_brain():
    series, truth = make_phantom()
    hu = series.volume[5]
    inside = restrict_to_blood_density(np.ones_like(truth[5]), hu, (50.0, 110.0))
    assert inside.sum() > 0
    assert inside[truth[5]].all(), "every clotted voxel is inside the acute band"
    assert not inside[hu == 30.0].any(), "brain tissue is below the band"


def test_restrict_to_blood_density_band_is_configurable():
    """Chronic collections sit near 25 HU, so the band cannot be hard-coded."""
    series, _ = make_phantom()
    hu = series.volume[5].copy()
    hu[50:60, 50:60] = 25.0
    region = np.zeros(hu.shape, bool)
    region[50:60, 50:60] = True
    assert not restrict_to_blood_density(region, hu, (50.0, 110.0))[50:60, 50:60].any()
    assert restrict_to_blood_density(region, hu, (15.0, 40.0))[50:60, 50:60].all()


def test_refine_with_grabcut_never_explodes():
    series, _ = make_phantom()
    seed = np.zeros(series.shape[-2:], np.uint8)
    seed[30:50, 20:50] = 2
    refined = refine_with_grabcut(series.volume[5], seed, iters=2)
    assert refined.shape == seed.shape
    assert refined.sum() <= 5 * (seed > 0).sum(), "a 5x growth is noise, not a lesion"


def test_refine_with_grabcut_survives_an_empty_seed():
    series, _ = make_phantom()
    empty = np.zeros(series.shape[-2:], np.uint8)
    assert not refine_with_grabcut(series.volume[0], empty).any()


def test_every_native_slice_is_examined():
    """Regression: a coarser grid must not skip native slices.

    Mapping grid -> native by nearest leaves native z in {0,2,4,7,9,11} of 12
    unchecked, which drops a bleed confined to z 5-8.
    """
    zmap = _native_to_grid_z(6, 12)
    assert len(zmap) == 12, "every native slice needs a grid CAM"
    assert zmap.min() >= 0 and zmap.max() < 6, "indices address the grid"
    # the phantom bleed lives on native z 5-8 and must not fall on one grid slice
    assert len(set(zmap[5:9].tolist())) > 1
    assert sorted(set(zmap.tolist())) == [0, 1, 2, 3, 4, 5]


def test_native_to_grid_z_is_identity_on_matching_grids():
    assert _native_to_grid_z(5, 5).tolist() == [0, 1, 2, 3, 4]
    assert _native_to_grid_z(1, 7).tolist() == [0] * 7


def test_resize_mask_to_returns_the_native_grid():
    mask = np.zeros((4, 8, 8), bool)
    mask[:, 2:4, 2:4] = True
    out = _resize_mask_to(mask, (12, 16, 16))
    assert out.shape == (12, 16, 16)
    assert out.any()
    assert set(np.unique(out)) <= {False, True}, "nearest-neighbour must not blend labels"


# ----------------------------------------------------------------- CAM


def test_cam_shape_and_range_for_both_model_kinds():
    series, _ = make_phantom()
    for kind in ("slice", "study"):
        cams, probs = cam_for_slices(
            build_head_ct_model(kind=kind, width=8), series, target_shape=None
        )
        assert cams.shape == (series.n_slices, *series.shape[-2:])
        assert 0.0 <= float(cams.min()) and float(cams.max()) <= 1.0
        assert probs.shape == (series.n_slices,)
        assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_cam_hi_res_is_finer_than_plain_cam():
    """Plain CAM is a 4x4 grid; the hi-res path must not be."""
    make_phantom()  # the fixture, not the model, defines the spatial size
    from vindr.ct.pseudo import _encode_hires

    net = build_head_ct_model(kind="slice", width=8)  # a slice model is the net itself
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        plain = net.encoder(x)
        hi = _encode_hires(net, x)
    assert hi.shape[-1] > plain.shape[-1]
    assert hi.shape[1] == plain.shape[1], "same channels, coarser pooling skipped"


# ----------------------------------------------------------------- end to end


def stub_cam_on_bleed(monkeypatch, centre=(0.594, 0.4375), radius_frac=0.09, prob=0.9):
    """Pin the CAM onto the phantom bleed.

    The plumbing is what is under test here, not CAM quality: an untrained net
    points at a random place, so whether its seed happens to land on the lesion
    is luck. This makes the CAM deterministic and centred on real blood.

    ``centre``/``radius_frac`` are fractions of the grid, so the peak lands on
    the same anatomy whatever grid the caller asks the net to run on.
    """
    from vindr.ct import pseudo

    def fake(model, series, **_kw):
        n, h, w = series.n_slices, series.shape[-2], series.shape[-1]
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = centre[0] * (h - 1), centre[1] * (w - 1)
        r = max(1.0, radius_frac * h)
        plane = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r**2))
        plane = plane / plane.max()
        return np.repeat(plane[None], n, 0).astype(np.float32), np.full(n, prob, np.float32)

    monkeypatch.setattr(pseudo, "cam_for_slices", fake)
    return fake


def test_pseudo_mask_study_returns_a_native_grid_mask(monkeypatch):
    series, truth = make_phantom()
    stub_cam_on_bleed(monkeypatch)
    rep = pseudo_mask_study(
        build_head_ct_model(kind="slice", width=8), series, target_shape=(6, 32, 32),
        prob_threshold=0.0, min_sharpness=1.0, use_grabcut=False,
    )
    assert rep is not None
    assert rep["mask"].shape == series.shape, "the mask must come home to the image grid"
    assert rep["grid_zyx"] == [6, 32, 32], "the network grid is reported, not silently lost"
    assert rep["image_zyx"] == list(series.shape)
    assert 0.0 < rep["max_prob"] <= 1.0
    assert rep["z_range"][0] <= rep["z_range"][1]
    assert 0 < rep["n_masked_slices"] <= series.n_slices
    assert (rep["mask"] & truth).any(), "the band must keep real clotted voxels"


def test_density_filter_reads_native_voxels(monkeypatch):
    """Regression: the HU band must not run on the resampled grid.

    Averaging a 70 HU bleed against 30 HU brain and air on a 2x-coarser grid
    drops it below the acute band, so a band computed there deletes the very
    haemorrhage it should confirm. On native voxels the same phantom survives.
    """
    from vindr.ct.volume import resample_to

    series, truth = make_phantom()
    stub_cam_on_bleed(monkeypatch)
    coarse = resample_to(series, (6, 32, 32))
    on_coarse = (coarse.volume[(coarse.volume >= 50) & (coarse.volume <= 110)]).size
    on_native = ((series.volume >= 50) & (series.volume <= 110)).sum()
    assert on_native > 0 and on_coarse < on_native, "the fixture must expose the dilution"

    rep = pseudo_mask_study(
        build_head_ct_model(kind="slice", width=8), series, target_shape=(6, 32, 32),
        prob_threshold=0.0, min_sharpness=1.0, use_grabcut=False,
    )
    assert rep is not None and (rep["mask"] & truth).any()


def test_pseudo_mask_study_declines_without_enough_evidence():
    series, _ = make_phantom(n=8, bleed_slices=range(0))  # no bleed at all
    model = build_head_ct_model(kind="slice", width=8)
    assert pseudo_mask_study(
        model, series, target_shape=(4, 32, 32), prob_threshold=1.01
    ) is None


def test_pseudo_mask_study_reports_evidence_for_filtering(monkeypatch):
    series, _ = make_phantom()
    stub_cam_on_bleed(monkeypatch)
    rep = pseudo_mask_study(
        build_head_ct_model(kind="slice", width=8), series, target_shape=None,
        prob_threshold=0.0, min_sharpness=1.0, use_grabcut=False,
    )
    assert rep is not None
    for key in ("max_prob", "mean_prob_masked", "n_dropped_slices", "spacing_zyx", "n_masked_slices"):
        assert key in rep, f"training needs {key} to filter pseudo-labels"


# ----------------------------------------------------------------- storage


def test_write_pseudo_dataset_links_images_and_logs_evidence(tmp_path: Path, monkeypatch):
    series, _ = make_phantom()
    stub_cam_on_bleed(monkeypatch)
    model = build_head_ct_model(kind="slice", width=8)
    rep = pseudo_mask_study(
        model, series, target_shape=None, prob_threshold=0.0, min_sharpness=1.0, use_grabcut=False
    )
    src = tmp_path / "images" / "phantom"
    src.mkdir(parents=True)
    np.save(src / "image.npy", series.volume)

    written = write_pseudo_dataset([rep], tmp_path / "pseudo", image_root=tmp_path / "images")
    assert len(written) == 1
    assert (written[0] / "mask.npy").exists()
    assert (written[0] / "image.npy").is_symlink(), "images are linked, not copied"

    meta = json.loads((tmp_path / "pseudo" / "pseudo_labels.json").read_text())
    assert "phantom" in meta
    assert "mask" not in meta["phantom"], "the array stays out of the JSON"
    assert meta["phantom"]["max_prob"] > 0


def test_pseudo_dataset_is_readable_by_the_training_loader(tmp_path: Path, monkeypatch):
    """The whole point: the written tree must feed load_masked_studies."""
    from vindr.ct import load_masked_studies

    stub_cam_on_bleed(monkeypatch)
    for i in range(2):
        series, _ = make_phantom()
        series.study_id = f"s{i}"
        model = build_head_ct_model(kind="slice", width=8)
        rep = pseudo_mask_study(
            model, series, target_shape=None, prob_threshold=0.0, min_sharpness=1.0, use_grabcut=False
        )
        src = tmp_path / "images" / f"s{i}"
        src.mkdir(parents=True)
        np.save(src / "image.npy", series.volume)
        write_pseudo_dataset([rep], tmp_path / "pseudo", image_root=tmp_path / "images")

    pairs = load_masked_studies(tmp_path / "pseudo")
    assert {st.study_id for st, _ in pairs} == {"s0", "s1"}
    for st, mask in pairs:
        assert mask.shape == st.shape


def test_link_images_skips_existing_masks(tmp_path: Path):
    from vindr.ct.pseudo import _link_images

    src = tmp_path / "src"
    src.mkdir()
    (src / "image.npy").touch()
    (src / "mask.npy").touch()
    (src / "labels.json").touch()
    (src / "a_mask.png").touch()
    dst = tmp_path / "dst"
    dst.mkdir()
    assert _link_images(src, dst) == 1
    assert (dst / "image.npy").exists()
    assert not (dst / "mask.npy").exists()
    assert not (dst / "a_mask.png").exists()


def test_unreadable_source_dir_is_tolerated(tmp_path: Path):
    from vindr.ct.pseudo import _link_images

    assert _link_images(tmp_path / "missing", tmp_path) == 0


def test_write_pseudo_dataset_skips_none_reports(tmp_path: Path):
    assert write_pseudo_dataset([None, None], tmp_path / "empty") == []


def test_grabcut_failure_falls_back_to_the_seed(monkeypatch):
    """A degenerate seed makes GrabCut raise; the seed is still a usable label."""
    series, _ = make_phantom()
    seed = np.zeros(series.shape[-2:], np.uint8)
    seed[30:40, 30:40] = 2

    def boom(*_a, **_k):
        raise cv2.error("bad seed")

    monkeypatch.setattr(cv2, "grabCut", boom)
    out = refine_with_grabcut(series.volume[5], seed)
    assert out.shape == seed.shape
    assert out.sum() > 0
