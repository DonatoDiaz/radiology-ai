"""Tests for head CT inference (vindr.ct.predict)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from vindr.ct.predict import load_model, load_series, predict_series


def make_study(tmp_path: Path, n: int = 12, with_hematoma: bool = True) -> Path:
    d = tmp_path / "study_demo"
    d.mkdir(parents=True, exist_ok=True)
    vol = np.full((n, 96, 96), -1000.0, np.float32)
    yy, xx = np.mgrid[0:96, 0:96]
    r = np.sqrt((yy - 48) ** 2 + (xx - 48) ** 2)
    vol[:, (r < 46) & (r > 42)] = 1000.0
    vol[:, r <= 42] = 35.0
    if with_hematoma:
        vol[:, (yy - 38) ** 2 + (xx - 52) ** 2 < 40] = 65.0
    np.save(d / "volume.npy", vol)
    return d


def make_ckpt(tmp_path: Path, kind: str) -> Path:
    from vindr.ct.model import HEMORRHAGE_TYPES, build_head_ct_model

    model = build_head_ct_model(kind=kind, width=8)
    ckpt = {
        "model_state": model.state_dict(),
        "model": kind,
        "labels": [*HEMORRHAGE_TYPES, "any"],
        "in_channels": 3,
        "width": 8,
        "target_shape": [8, 64, 64],
    }
    path = tmp_path / f"{kind}.pt"
    torch.save(ckpt, path)
    return path


def test_load_series_reads_npy_backed_nifti(tmp_path):
    nib = pytest.importorskip("nibabel")
    d = make_study(tmp_path)
    vol = np.load(d / "volume.npy")
    nib.save(nib.Nifti1Image(np.transpose(vol, (2, 1, 0)), np.eye(4)), d / "volume.nii.gz")
    series = load_series(d)
    assert series.volume.shape == vol.shape  # canonical (Z, Y, X) restored
    np.testing.assert_allclose(series.volume, vol, rtol=1e-5)


def test_load_series_rejects_unknown_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("x")
    with pytest.raises(ValueError):
        load_series(f)


def test_predict_study_mode_outputs_all_labels(tmp_path):
    from vindr.ct.model import HEMORRHAGE_TYPES

    d = make_study(tmp_path)
    ckpt = make_ckpt(tmp_path, "study")
    series = load_series(d)
    model, blob = load_model(ckpt, torch.device("cpu"))
    res = predict_series(model, series, blob, torch.device("cpu"))

    assert res["kind"] == "study"
    assert set(res["labels"]) == {*HEMORRHAGE_TYPES, "any"}
    assert res["n_slices"] == 12
    assert all(0.0 <= v <= 1.0 for v in res["probabilities"].values())
    assert set(res["positive"]) == set(res["labels"])
    assert json.dumps(res)  # must be serialisable


def test_predict_slice_mode_ranks_top_slices(tmp_path):
    d = make_study(tmp_path)
    ckpt = make_ckpt(tmp_path, "slice")
    series = load_series(d)
    model, blob = load_model(ckpt, torch.device("cpu"))
    res = predict_series(model, series, blob, torch.device("cpu"), threshold=0.0)

    assert res["kind"] == "slice"
    assert "top_slices" in res
    # sorted by descending 'any' score
    scores = [s["any"] for s in res["top_slices"]]
    assert scores == sorted(scores, reverse=True)
    assert all(0 <= s["z"] < res["n_slices"] for s in res["top_slices"])


def test_load_model_rebuilds_weights(tmp_path):
    d = make_study(tmp_path)
    ckpt = make_ckpt(tmp_path, "study")
    series = load_series(d)
    model, blob = load_model(ckpt, torch.device("cpu"))
    # same weights -> identical prediction
    a = predict_series(model, series, blob, torch.device("cpu"))
    model2, blob2 = load_model(ckpt, torch.device("cpu"))
    b = predict_series(model2, series, blob2, torch.device("cpu"))
    assert a["probabilities"] == b["probabilities"]
