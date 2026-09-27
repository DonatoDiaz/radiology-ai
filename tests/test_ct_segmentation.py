"""Tests for head CT segmentation and lesion semiology (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pytest
import torch

from vindr.ct import (
    CTSeries,
    HeadCTSegDataset,
    HeadCTSegNet,
    build_seg_model,
    collate_seg,
    dice_loss,
    dice_score,
    find_lesions,
    iou_score,
    lesion_semiotics,
    load_masked_studies,
    seg_loss,
)
from vindr.ct.train_seg import split_pairs

SPACING = (5.0, 0.8, 0.8)  # (z, y, x) mm


def _grid(h: int = 128, w: int = 128):
    return np.mgrid[0:h, 0:w]


# ---------------------------------------------------------------- model


@pytest.mark.parametrize(
    "shape", [(2, 3, 224, 224), (1, 3, 200, 168), (2, 5, 128, 128), (1, 3, 100, 100)]
)
def test_seg_net_keeps_spatial_shape(shape):
    """Padding for the skip connections must be cropped back off exactly."""
    out = HeadCTSegNet(in_channels=shape[1], width=8)(torch.randn(*shape))
    assert out.shape == (shape[0], 1, shape[2], shape[3])


def test_seg_net_multiclass_and_gradients():
    model = HeadCTSegNet(in_channels=3, num_classes=3, width=8)
    x = torch.randn(2, 3, 64, 64)
    y = (torch.rand(2, 3, 64, 64) > 0.8).float()
    logits = model(x)
    assert logits.shape == y.shape
    seg_loss(logits, y).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_seg_loss_and_metrics_on_degenerate_predictions():
    target = torch.zeros(1, 1, 32, 32)
    target[..., 8:20, 8:20] = 1.0
    perfect = (target * 20.0) - 10.0  # sigmoid -> exactly the target
    empty = torch.full_like(target, -10.0)  # sigmoid -> all below threshold
    assert dice_score(perfect.sigmoid(), target) == pytest.approx(1.0, abs=1e-3)
    assert dice_score(empty.sigmoid(), target) < 0.05
    assert iou_score(empty.sigmoid(), target) < 0.05
    assert float(dice_score(perfect, target)) == pytest.approx(1.0, abs=1e-3)  # logits accepted too
    assert float(seg_loss(perfect, target)) < float(seg_loss(empty, target))


def test_dice_loss_rewards_overlap():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1.0
    overlap = torch.full_like(target, -5.0)
    overlap[..., 6:14, 6:14] = 5.0  # 6x6 of the 8x8 target predicted, shifted by 2px
    nothing = torch.full_like(target, -5.0)
    assert float(dice_loss(overlap, target)) < float(dice_loss(nothing, target))
    assert float(seg_loss(overlap, target)) < float(seg_loss(nothing, target))


def test_build_seg_model_defaults():
    model = build_seg_model()
    assert isinstance(model, HeadCTSegNet)
    assert model(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)


# ---------------------------------------------------------------- semiology


def _volume(shape=(20, 128, 128)):
    return np.zeros(shape, bool)


def test_semiotics_epidural_compact_against_bone():
    """Biconvex, against the inner table, confined to one side -> epidural."""
    mask = _volume()
    mask[4:10, 30:70, 2:40] = True  # left border, compact, x stays under midline
    (lesion,) = find_lesions(mask, SPACING)
    assert lesion["suggests"] == "epidural"
    assert lesion["shape_hints"] == ["biconvex_limited"]


def test_semiotics_subdural_crescent_crossing_midline():
    yy, xx = _grid()
    mask = _volume()
    crescent = (
        ((yy - 64) ** 2 / 58**2 + (xx - 64) ** 2 / 62**2 < 1)
        & ~((yy - 64) ** 2 / 50**2 + (xx - 64) ** 2 / 54**2 < 1)
        & (xx < 70)
    )
    for z in range(5, 12):
        mask[z] = crescent
    (lesion,) = find_lesions(mask, SPACING)
    assert lesion["suggests"] == "subdural"
    assert lesion["shape_hints"] == ["crescentic_unconfined"]


def test_semiotics_intraparenchymal_round():
    yy, xx = _grid()
    mask = _volume()
    for z in range(8, 13):
        mask[z] = (yy - 64) ** 2 + (xx - 64) ** 2 < 324
    (lesion,) = find_lesions(mask, SPACING)
    assert lesion["suggests"] == "intraparenchymal"


def test_semiotics_subarachnoid_serpiginous():
    yy, xx = _grid()
    sulci = (np.abs(((xx - 64) % 26) - 13) < 0.8) & (yy > 20) & (yy < 118) & (np.abs(np.sin((xx - 64) / 13.0)) > 0.3)
    mask = _volume()
    for z in range(6, 11):
        mask[z] = sulci
    found = find_lesions(mask, SPACING)
    assert found, "thin serpiginous traces must survive the area filter"
    assert {lesion["suggests"] for lesion in found} == {"subarachnoid"}


def test_lesion_semiotics_reports_borders_and_midline():
    mask = np.zeros((128, 128), bool)
    mask[30:70, 2:40] = True
    (finding,) = lesion_semiotics(mask)
    assert finding["borders"]["left"] is True
    assert finding["borders"]["right"] is False
    assert finding["crosses_midline"] is False
    assert 0.0 < finding["solidity"] <= 1.0
    assert finding["circularity"] > 0.0


def test_solidity_never_exceeds_one():
    """contourArea reports the pixel-centre polygon, so it can undershoot area."""
    mask = np.zeros((128, 128), bool)
    mask[10:50, 10:50] = True  # a solid block: convex hull == the shape
    (finding,) = lesion_semiotics(mask)
    assert finding["solidity"] <= 1.0


# ---------------------------------------------------------------- grouping


def test_lesions_merge_consecutive_slices_but_not_across_a_gap():
    yy, xx = _grid()
    sphere = (yy - 64) ** 2 + (xx - 64) ** 2 < 225
    continuous = _volume()
    for z in range(4, 10):
        continuous[z] = sphere
    assert len(find_lesions(continuous, SPACING)) == 1
    assert find_lesions(continuous, SPACING)[0]["z_range"] == [4, 9]

    gapped = _volume()
    for z in list(range(2, 5)) + list(range(10, 13)):
        gapped[z] = sphere
    assert len(find_lesions(gapped, SPACING)) == 2  # the empty slices break it


def test_lesion_volume_uses_spacing():
    mask = np.zeros((4, 100, 100), bool)
    mask[:, 10:60, 10:60] = True
    (lesion,) = find_lesions(mask, (10.0, 1.0, 1.0))
    assert lesion["volume_ml"] == pytest.approx(50 * 50 * 4 * 10 / 1000.0)


def test_lesions_sorted_by_volume_and_empty_mask_is_safe():
    mask = _volume()
    mask[1:3, 10:20, 10:20] = True
    mask[8:12, 40:80, 40:80] = True
    found = find_lesions(mask, SPACING)
    assert [lesion["volume_ml"] for lesion in found] == sorted(
        (lesion["volume_ml"] for lesion in found), reverse=True
    )
    assert find_lesions(np.zeros((5, 8, 8), bool), SPACING) == []


def test_lesion_semiotics_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\(Z, Y, X\)"):
        find_lesions(np.zeros((8, 8), bool), SPACING)


# ---------------------------------------------------------------- datasets


def _write_masked_study(root: Path, name: str, n: int = 12, shape=(64, 64), mask_style: str = "nifti"):
    yy, xx = _grid(*shape)
    vol = np.full((n, *shape), 300.0, np.float32)
    vol[:, yy + xx < 20] = 900.0
    mask = np.zeros((n, *shape), np.uint8)
    mask[3:8, 10:34, 8:38] = 1
    d = root / name
    d.mkdir(parents=True)
    if mask_style == "nifti":
        for z in range(n):
            cv2.imwrite(str(d / f"img_{z:03d}.png"), np.clip(vol[z], 0, 3000).astype(np.uint16))
        nib.save(nib.Nifti1Image(mask.transpose(2, 1, 0).astype(np.uint8), np.eye(4)), d / "mask.nii.gz")
    elif mask_style == "png_folder":
        nib.save(nib.Nifti1Image(vol.transpose(2, 1, 0), np.eye(4)), d / "image.nii.gz")
        (d / "masks").mkdir()
        for z in range(n):
            cv2.imwrite(str(d / "masks" / f"{z:03d}.png"), mask[z] * 255)
    elif mask_style == "suffix":
        nib.save(nib.Nifti1Image(vol.transpose(2, 1, 0), np.eye(4)), d / "image.nii.gz")
        for z in range(n):
            cv2.imwrite(str(d / f"slice_{z:03d}_mask.png"), mask[z] * 255)
    return d


def test_load_masked_studies_accepts_the_documented_layouts(tmp_path):
    _write_masked_study(tmp_path, "s_nifti", mask_style="nifti")
    _write_masked_study(tmp_path, "s_png", mask_style="png_folder")
    _write_masked_study(tmp_path, "s_suffix", mask_style="suffix")
    pairs = load_masked_studies(tmp_path)
    assert {st.study_id for st, _ in pairs} == {"s_nifti", "s_png", "s_suffix"}
    for st, mask in pairs:
        assert mask.shape[:1] == (st.n_slices,)
        assert mask.shape[-2:] == st.shape[-2:]


def test_load_masked_studies_skips_unmasked_and_mismatched(tmp_path):
    _write_masked_study(tmp_path, "good", mask_style="nifti")
    d = tmp_path / "no_mask"
    d.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 12), np.float32), np.eye(4)), d / "image.nii.gz")
    short = tmp_path / "short_mask"
    short.mkdir()
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 12), np.float32), np.eye(4)), short / "image.nii.gz")
    nib.save(nib.Nifti1Image(np.zeros((64, 64, 4), np.uint8), np.eye(4)), short / "mask.nii.gz")
    assert [st.study_id for st, _ in load_masked_studies(tmp_path)] == ["good"]


def test_seg_dataset_skips_empty_slices_and_pairs_masks(tmp_path):
    _write_masked_study(tmp_path, "s1", mask_style="nifti")
    ds = HeadCTSegDataset(load_masked_studies(tmp_path), target_shape=None)
    assert len(ds) == 5  # only slices 3..7 carry a lesion
    x, y = ds[0]
    assert x.shape[0] == 3  # 2.5D: slice + 1 neighbour per side
    assert y.shape == (1, *ds.items[0][1].shape[1:3])
    assert 0.0 < float(y.mean()) < 1.0


def test_seg_dataset_raises_without_labelled_slices(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    series = CTSeries(volume=np.zeros((6, 32, 32), np.float32), study_id="empty")
    with pytest.raises(ValueError, match="no labelled slices"):
        HeadCTSegDataset([(series, np.zeros((6, 32, 32), np.float32))])


def test_seg_dataset_resamples_masks_nearest_neighbour(tmp_path):
    _write_masked_study(tmp_path, "s1", n=12, shape=(64, 64), mask_style="nifti")
    pairs = load_masked_studies(tmp_path)
    ds = HeadCTSegDataset(pairs, target_shape=(8, 32, 32))
    _x, y = ds[0]
    assert tuple(y.shape) == (1, 32, 32)
    # nearest-neighbour keeps the mask binary instead of blending in partial values
    assert set(torch.unique(y).tolist()) <= {0.0, 1.0}


def test_collate_seg_batches_pairs():
    pairs = [(CTSeries(volume=np.zeros((6, 32, 32), np.float32), study_id="a"), np.ones((6, 32, 32), np.float32))]
    ds = HeadCTSegDataset(pairs, target_shape=None)
    xb, yb = collate_seg([ds[0], ds[0], ds[0]])
    assert xb.shape == (3, 3, 32, 32)
    assert yb.shape == (3, 1, 32, 32)


def test_split_pairs_keeps_studies_disjoint():
    pairs = [(str(i), None) for i in range(10)]
    train, val = split_pairs(pairs, 0.2, seed=0)
    assert len(val) == 2 and len(train) == 8
    assert not {a for a, _ in train} & {b for b, _ in val}
    assert split_pairs(pairs, 0.2, seed=0) == (train, val)  # deterministic
