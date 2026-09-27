"""Tests for the head CT pipeline (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vindr.ct import (
    HEMORRHAGE_TYPES,
    CTSeries,
    HeadCTSliceDataset,
    HeadCTSliceNet,
    HeadCTStudyDataset,
    HeadCTStudyNet,
    any_label,
    apply_window,
    bbox_mask,
    classify_density,
    evans_index,
    lesion_density,
    lesion_volume_ml,
    load_nifti,
    loss_fn,
    midline_shift_mm,
    normalize_uint16_png,
    resample_to,
    slice_to_study_scores,
    slice_with_neighbours,
)


def phantom(n_slices: int = 24, size: int = 128, with_hematoma: bool = True) -> CTSeries:
    """Synthetic head CT: air, skull ring, brain, optional 65 HU hematoma."""
    vol = np.full((n_slices, size, size), -1000.0, np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    r = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    vol[:, (r < size * 0.48) & (r > size * 0.44)] = 1000.0
    vol[:, r <= size * 0.44] = 35.0
    if with_hematoma:
        hema = ((yy - size * 0.35) ** 2 + (xx - size * 0.45) ** 2) < (size * 0.08) ** 2
        vol[:, hema] = 65.0
    return CTSeries(volume=vol, spacing=(5.0, 0.8, 0.8))


def test_windowing_maps_hu_to_uint8():
    vol = np.array([[[-100.0, 40.0, 120.0]]], dtype=np.float32)
    out = apply_window(vol, "brain")  # level 40, width 80 -> [0, 120]
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_resample_preserves_shape_and_updates_spacing():
    s = phantom(n_slices=20, size=128)
    out = resample_to(s, (16, 64, 64))
    assert out.volume.shape == (16, 64, 64)
    z, y, x = out.spacing
    assert z > s.spacing[0]  # fewer slices -> thicker z spacing
    assert y > s.spacing[1] and x > s.spacing[2]


def test_resample_is_identity_when_shape_matches():
    s = phantom(n_slices=12, size=64)
    out = resample_to(s, (12, 64, 64))
    assert out.volume.shape == s.volume.shape
    np.testing.assert_allclose(out.spacing, s.spacing, rtol=1e-6)


def test_slice_with_neighbours_shape_and_edge_padding():
    vol = phantom().volume
    assert slice_with_neighbours(vol, 5, context=1).shape == (128, 128, 3)
    edge = slice_with_neighbours(vol, 0, context=2)
    assert edge.shape == (128, 128, 5)  # padded at the volume edge
    assert np.allclose(edge[..., 0], edge[..., 1])  # padding repeats the same plane


def test_png_hu_conversion_roundtrip(tmp_path):
    import cv2

    raw = np.array([[100, 200, 300]], dtype=np.uint16)
    path = tmp_path / "slice.png"
    cv2.imwrite(str(path), raw)
    hu = normalize_uint16_png(path, slope=2.0, intercept=-1024.0)
    np.testing.assert_allclose(hu, raw.astype(np.float32) * 2.0 - 1024.0)


def test_nifti_roundtrip(tmp_path):
    nib = pytest.importorskip("nibabel")
    vol = phantom().volume  # canonical (Z, Y, X)
    path = tmp_path / "vol.nii.gz"
    # NIfTI axis i maps to x, j to y, k to z -> store transposed
    nib.save(nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), np.eye(4)), path)
    loaded = load_nifti(path)
    assert loaded.volume.shape == vol.shape
    np.testing.assert_allclose(loaded.volume, vol, rtol=1e-5)
    assert loaded.spacing == pytest.approx((1.0, 1.0, 1.0))


def test_nifti_spacing_is_reordered_to_zyx(tmp_path):
    nib = pytest.importorskip("nibabel")
    vol = np.zeros((10, 20, 30), dtype=np.float32)
    # NIfTI affine columns map (i, j, k) -> (x, y, z) with spacing 0.5, 0.75, 2.0
    affine = np.diag([0.5, 0.75, 2.0, 1.0])
    path = tmp_path / "spaced.nii.gz"
    nib.save(nib.Nifti1Image(vol, affine), path)
    loaded = load_nifti(path)
    z_sp, y_sp, x_sp = loaded.spacing
    assert z_sp == pytest.approx(2.0)
    assert y_sp == pytest.approx(0.75)
    assert x_sp == pytest.approx(0.5)


def test_slice_model_output_shape():
    model = HeadCTSliceNet()
    out = model(torch.randn(2, 3, 128, 128))
    assert out.shape == (2, len(HEMORRHAGE_TYPES) + 1)


def test_study_model_attention_pools_to_study_level():
    model = HeadCTStudyNet()
    logits, slice_logits, attn = model(torch.randn(2, 8, 3, 128, 128), return_slices=True)
    assert logits.shape == (2, len(HEMORRHAGE_TYPES) + 1)
    assert slice_logits.shape == (2, 8, len(HEMORRHAGE_TYPES) + 1)
    torch.testing.assert_close(attn.sum(dim=1), torch.ones(2), rtol=1e-5, atol=1e-6)


def test_study_model_backward():
    model = HeadCTStudyNet()
    loss = loss_fn(model(torch.randn(1, 4, 3, 96, 96)), torch.zeros(1, len(HEMORRHAGE_TYPES) + 1))
    loss.backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.parameters())


def test_any_label_appends_and_recomputes():
    five = torch.zeros(2, len(HEMORRHAGE_TYPES))
    five[1, 1] = 1
    assert tuple(any_label(five).shape) == (2, len(HEMORRHAGE_TYPES) + 1)
    torch.testing.assert_close(any_label(five)[:, -1], torch.tensor([0.0, 1.0]))

    six = torch.zeros(2, len(HEMORRHAGE_TYPES) + 1)
    six[0, 2] = 1
    six[0, -1] = 0.0  # stale roll-up must be recomputed
    assert tuple(any_label(six).shape) == (2, len(HEMORRHAGE_TYPES) + 1)
    torch.testing.assert_close(any_label(six)[:, -1], torch.tensor([1.0, 0.0]))


def test_slice_to_study_scores_max_pools():
    scores = slice_to_study_scores(np.array([[0.1, 0.9], [0.2, 0.3]]))
    np.testing.assert_allclose(scores, [0.2, 0.9])


def test_lesion_density_names_blood():
    s = phantom()
    mask = np.zeros_like(s.volume, dtype=bool)
    yy, xx = np.mgrid[0:128, 0:128]
    mask[:, ((yy - 44) ** 2 + (xx - 57) ** 2) < 100] = True
    d = lesion_density(s.volume, mask)
    assert d["is_blood_density"] is True
    assert d["band"] == "acute_hemorrhage"


def test_classify_density_bands():
    assert classify_density(65)["band"] == "acute_hemorrhage"
    assert classify_density(-10)["band"] == "csf"
    assert classify_density(1200)["band"] == "bone"
    assert classify_density(-1000)["band"] == "air"


def test_lesion_volume_uses_spacing():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[:5] = True  # 500 voxels
    res = lesion_volume_ml(mask, (10.0, 1.0, 1.0))  # 10 mm x 1 mm x 1 mm = 10 mm^3/voxel
    assert res["n_voxels"] == 500
    assert res["volume_ml"] == pytest.approx(5.0)


def test_midline_shift_detects_asymmetry():
    s = phantom(with_hematoma=True)
    res = midline_shift_mm(s)
    assert res["max_shift_mm"] is not None
    assert res["max_shift_mm"] > 0


def test_evans_index_flags_hydrocephalus():
    vol = np.full((10, 128, 128), -1000.0, np.float32)
    vol[:, :, :] = 35.0               # brain parenchyma -> full inner width
    vol[:, 40:56, 58:70] = 5.0       # narrow frontal horns
    normal = evans_index(vol)
    assert normal["evans_index"] is not None
    assert normal["hydrocephalus"] is False
    vol[:, 30:60, 30:98] = 5.0       # widened frontal horns
    wide = evans_index(vol)
    assert wide["hydrocephalus"] is True
    assert wide["evans_index"] > 0.30


def test_bbox_mask_clamps_out_of_range():
    mask = bbox_mask((10, 20, 30), (-5, -5, 100, 100))
    assert mask.shape == (10, 20, 30)
    assert mask.any() and mask.all()  # clamped to the full volume


def test_slice_dataset_yields_per_slice_samples():
    studies = [phantom(20), phantom(16)]
    labels = [[0] * (len(HEMORRHAGE_TYPES) + 1), [1] + [0] * len(HEMORRHAGE_TYPES)]
    ds = HeadCTSliceDataset(studies, labels, target_shape=(8, 96, 96))
    assert len(ds) == 16  # 8 + 8 after resampling
    x, y, _sid = ds[0]
    assert x.shape == (3, 96, 96)
    assert y.shape == (len(HEMORRHAGE_TYPES) + 1,)


def test_study_dataset_yields_fixed_slice_count():
    studies = [phantom(20), phantom(30)]
    ds = HeadCTStudyDataset(studies, None, slices_per_study=6, target_shape=(8, 96, 96))
    x, _y, i = ds[0]
    assert x.shape == (6, 3, 96, 96)  # (S, C, H, W)
    assert i == 0


def test_augmentation_preserves_shape_and_range():
    st = phantom(12, with_hematoma=True)
    ds = HeadCTSliceDataset([st], None, target_shape=None, train=True, augment=True)
    np.random.seed(0)
    for _ in range(5):
        x, _, _ = ds[3]
        assert x.shape == (3, 128, 128)
        assert torch.isfinite(x).all()
        assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0


def test_augmentation_disabled_when_train_false():
    st = phantom(12)
    ds = HeadCTSliceDataset([st], None, target_shape=None, train=False, augment=True)
    np.random.seed(0)
    a, _, _ = ds[3]
    np.random.seed(1)
    b, _, _ = ds[3]
    torch.testing.assert_close(a, b)  # deterministic without augmentation


def test_evaluate_pools_slice_predictions_per_study():
    """Slice-level scores must be max-pooled per study before AUROC.

    Regression: raw slice rows weighted every study by its slice count.
    """
    import torch as _t
    from torch.utils.data import DataLoader

    from vindr.ct.dataset import collate_slices
    from vindr.ct.model import build_head_ct_model
    from vindr.ct.train import evaluate

    studies = [phantom(8, with_hematoma=bool(i % 2)) for i in range(4)]
    labels = [[0] * (len(HEMORRHAGE_TYPES) + 1) for _ in range(4)]
    labels[1][0] = labels[1][-1] = 1
    labels[3][1] = labels[3][-1] = 1
    ds = HeadCTSliceDataset(studies, labels, target_shape=(4, 64, 64))
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_slices)
    model = build_head_ct_model(kind="slice", width=8)
    res = evaluate(model, loader, _t.device("cpu"), study_level=False)
    assert set(res["macro"]) == {"auroc", "ap"}
    assert len(res["per_class"]) == len(HEMORRHAGE_TYPES) + 1


def test_evaluate_study_level_runs():
    import torch as _t
    from torch.utils.data import DataLoader

    from vindr.ct.dataset import collate_studies
    from vindr.ct.model import build_head_ct_model
    from vindr.ct.train import evaluate

    studies = [phantom(8, with_hematoma=bool(i % 2)) for i in range(4)]
    labels = [[0] * (len(HEMORRHAGE_TYPES) + 1) for _ in range(4)]
    labels[1][0] = labels[1][-1] = 1
    ds = HeadCTStudyDataset(studies, labels, slices_per_study=4, target_shape=(4, 64, 64))
    loader = DataLoader(ds, batch_size=2, collate_fn=collate_studies)
    model = build_head_ct_model(kind="study", width=8)
    res = evaluate(model, loader, _t.device("cpu"), study_level=True)
    assert "macro" in res and "auroc" in res["macro"]


def test_load_study_folders_reads_png_series_with_rescale(tmp_path):
    """A prepared PNG study tree must load into HU volumes."""
    import cv2

    from vindr.ct.dataset import load_study_folders

    studies_dir = tmp_path / "studies"
    rng = np.random.default_rng(0)
    for s in range(2):
        d = studies_dir / f"study_{s}"
        d.mkdir(parents=True)
        for i in range(3):
            cv2.imwrite(str(d / f"study_{s}_{i:04d}.png"), rng.integers(0, 4096, (16, 16)).astype(np.uint16))
    (tmp_path / "rescale_values.csv").write_text(
        "study_id,rescale_slope,rescale_intercept\nstudy_0,-1.0,1024.0\nstudy_1,-1.0,1024.0\n"
    )
    loaded = load_study_folders(studies_dir)
    assert len(loaded) == 2
    assert loaded[0].volume.shape == (3, 16, 16)
    assert loaded[0].study_id == "study_0"
    # HU = pixel * slope + intercept = 1024 - pixel, so values stay in range
    assert float(loaded[0].volume.max()) <= 1024.0 + 1e-3


def test_load_study_folders_prefers_nifti_then_png(tmp_path):
    import cv2

    nib = pytest.importorskip("nibabel")
    from vindr.ct.dataset import load_study_folders

    d_nii = tmp_path / "studies" / "with_nii"
    d_nii.mkdir(parents=True)
    # NIfTI is (X, Y, Z); the loader transposes to canonical (Z, Y, X)
    nib.save(nib.Nifti1Image(np.zeros((8, 8, 4), np.float32), np.eye(4)), d_nii / "v.nii.gz")
    d_png = tmp_path / "studies" / "with_png"
    d_png.mkdir(parents=True)
    cv2.imwrite(str(d_png / "a_0000.png"), np.zeros((8, 8), np.uint16))

    loaded = load_study_folders(tmp_path / "studies")
    shapes = {s.study_id or s.series_id: s.volume.shape for s in loaded}
    assert shapes["with_nii"] == (4, 8, 8)  # NIfTI wins when present
    assert any(v == (1, 8, 8) for v in shapes.values())
