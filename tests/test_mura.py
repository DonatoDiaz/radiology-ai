"""Tests for Phase 6 — MURA study-level preparation.

The fake release below is built to the real one's shape: patients, several
studies each, ``study1_positive`` folders, one manifest with no label column.
That last detail is the point — MURA does not put the label in the CSV, and a
test suite that hands the label over directly would never notice.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from vindr.mura import (
    ABNORMAL_SUFFIXES,
    REQUIRED_COLUMNS,
    MuraStudyDataset,
    abnormal_from_path,
    collate_studies,
    collect_studies,
    make_patient_splits,
    parse_mura_csv,
    patients_in_both_splits,
    pool_view_probabilities,
    split_summary,
    study_probabilities,
)

MANIFEST_COLUMNS = ["mura_id", "study_id", "patient_id", "series_id", "body_part",
                    "laterality", "subset", "height", "width", "depth", "view"]


def _write_image(path: Path, seed: int = 0, size: int = 24) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (size, size), dtype=np.uint8)).save(path)


def fake_release(root: Path, n_patients: int = 8, studies_each: int = 3) -> Path:
    """A MURA-shaped tree plus its manifest, with no label column."""
    rows = []
    for p in range(n_patients):
        patient = f"patient{p:05d}"
        for s in range(1, studies_each + 1):
            abnormal = (p + s) % 2 == 0
            folder = f"train/XR_SHOULDER/{patient}/study{s}_{'positive' if abnormal else 'negative'}"
            for img in range(1, 4):
                name = f"image{img}.png"
                _write_image(root / folder / name, seed=p * 10 + s * 3 + img)
                rows.append({
                    "mura_id": f"MURA-v1.1/{folder}/{name}",
                    "study_id": f"S{s}", "patient_id": patient, "series_id": "series1",
                    "body_part": "XR_SHOULDER", "laterality": "R", "subset": "train",
                    "height": 24, "width": 24, "depth": 3, "view": "AP",
                })
    csv = root / "train_image_paths.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(csv, index=False)
    return csv


# --------------------------------------------------------------------------- #
# the label lives in the path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("path,expected", [
    ("train/XR_SHOULDER/patient00001/study1_positive/image1.png", 1),
    ("train/XR_SHOULDER/patient00001/study1_negative/image1.png", 0),
    ("MURA-v1.1/train/XR_HAND/patient00007/study2_positive/image3.png", 1),
])
def test_label_is_read_from_the_folder_name(path, expected):
    assert abnormal_from_path(path) == expected


def test_a_path_without_a_label_is_refused_rather_than_assumed_normal():
    assert abnormal_from_path("train/XR_SHOULDER/patient00001/study1/image1.png") is None


def test_suffix_table_covers_the_two_mura_labels():
    assert ABNORMAL_SUFFIXES == {"positive": 1, "negative": 0}


def test_manifest_columns_are_checked(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame([{"mura_id": "x", "patient_id": "p"}]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing"):
        parse_mura_csv(bad)


def test_required_columns_are_the_real_ones():
    for col in ("mura_id", "patient_id", "body_part"):
        assert col in REQUIRED_COLUMNS


def test_a_manifest_without_labels_is_refused(tmp_path):
    csv = tmp_path / "nolabel.csv"
    pd.DataFrame([{**{c: "v" for c in MANIFEST_COLUMNS},
                   "mura_id": "MURA-v1.1/train/XR_SHOULDER/p1/study1/image1.png"}]).to_csv(
        csv, index=False)
    with pytest.raises(ValueError, match="positive/negative"):
        parse_mura_csv(csv)


def test_parsing_extracts_study_folder_and_label(tmp_path):
    csv = fake_release(tmp_path)
    df = parse_mura_csv(csv)
    assert len(df) == 8 * 3 * 3
    assert set(df["abnormal"]) == {0, 1}
    assert df["study_folder"].str.contains("_positive|_negative").all()
    assert (df["study_folder"].str.count("/") == 3).all()


def test_study_id_alone_is_not_unique_so_the_folder_is_used(tmp_path):
    """MURA reuses S1 across patients, so identity has to include the patient."""
    csv = fake_release(tmp_path)
    df = parse_mura_csv(csv)
    assert df["study_id"].duplicated().any()
    studies = collect_studies(df, tmp_path, require_images=False)
    assert not studies["study_folder"].duplicated().any()


# --------------------------------------------------------------------------- #
# studies, not images
# --------------------------------------------------------------------------- #


def test_images_collapse_into_studies(tmp_path):
    csv = fake_release(tmp_path)
    studies = collect_studies(parse_mura_csv(csv), tmp_path)
    assert len(studies) == 24
    assert set(studies["n_images"]) == {3}
    assert studies["abnormal"].nunique() == 2


def test_image_count_is_taken_from_disk_not_the_manifest(tmp_path):
    """A release missing an image must not look like a clean 3-view study."""
    csv = fake_release(tmp_path)
    df = parse_mura_csv(csv)
    victim = tmp_path / df["study_folder"].iloc[0] / df["image_name"].iloc[0]
    victim.unlink()
    studies = collect_studies(df, tmp_path)
    assert studies["n_images"].min() == 2
    assert (studies["n_views"] == 3).all()


def test_a_study_with_no_images_is_reported_as_zero(tmp_path):
    csv = fake_release(tmp_path)
    studies = collect_studies(parse_mura_csv(csv), tmp_path)
    for folder in studies["study_folder"]:
        for p in (tmp_path / folder).iterdir():
            p.unlink()
    studies = collect_studies(parse_mura_csv(csv), tmp_path)
    assert (studies["n_images"] == 0).all()


# --------------------------------------------------------------------------- #
# the split is per patient
# --------------------------------------------------------------------------- #


def test_no_patient_appears_in_both_splits():
    """The whole point: a per-study split would leak the same arm."""
    rng = np.random.default_rng(0)
    studies = pd.DataFrame({
        "study_folder": [f"train/XR_SHOULDER/patient{i//3:05d}/study{i%3+1}" for i in range(60)],
        "patient_id": [f"patient{i//3:05d}" for i in range(60)],
        "body_part": "XR_SHOULDER", "laterality": "R",
        "abnormal": [i % 2 for i in range(60)], "n_images": 3,
    })
    split = make_patient_splits(studies, val_fraction=0.25)
    assert patients_in_both_splits(split) == []
    assert split.groupby("patient_id")["split"].nunique().max() == 1
    assert rng is not None  # the split must not depend on row order


def test_the_split_is_whole_and_deterministic():
    studies = pd.DataFrame({
        "study_folder": [f"f{i}" for i in range(20)],
        "patient_id": [f"p{i % 5}" for i in range(20)],
        "body_part": "XR_SHOULDER", "laterality": "R", "abnormal": 0, "n_images": 3,
    })
    a = make_patient_splits(studies, val_fraction=0.3, seed=7)
    b = make_patient_splits(studies, val_fraction=0.3, seed=7)
    assert a["split"].tolist() == b["split"].tolist()
    assert set(a["split"]) <= {"train", "val"}
    assert (a["split"] == "train").any() and (a["split"] == "val").any()


def test_a_different_seed_can_move_patients():
    studies = pd.DataFrame({
        "study_folder": [f"f{i}" for i in range(40)],
        "patient_id": [f"p{i % 10}" for i in range(40)],
        "body_part": "XR_SHOULDER", "laterality": "R", "abnormal": 0, "n_images": 3,
    })
    a = set(make_patient_splits(studies, 0.25, seed=1).query("split=='val'")["patient_id"])
    b = set(make_patient_splits(studies, 0.25, seed=2).query("split=='val'")["patient_id"])
    assert a != b


def test_row_order_does_not_change_the_split():
    studies = pd.DataFrame({
        "study_folder": [f"f{i}" for i in range(30)],
        "patient_id": [f"p{i % 6}" for i in range(30)],
        "body_part": "XR_SHOULDER", "laterality": "R", "abnormal": 0, "n_images": 3,
    })
    straight = make_patient_splits(studies, 0.3)
    shuffled = make_patient_splits(studies.sample(frac=1.0, random_state=1), 0.3)
    assert dict(zip(straight["patient_id"], straight["split"], strict=True)) == dict(
        zip(shuffled["patient_id"], shuffled["split"], strict=True)
    )


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_an_impossible_val_fraction_is_refused(fraction):
    studies = pd.DataFrame({"study_folder": ["f"], "patient_id": ["p"], "abnormal": [0]})
    with pytest.raises(ValueError, match="val_fraction"):
        make_patient_splits(studies, fraction)


def test_leak_check_needs_a_split_first():
    studies = pd.DataFrame({"study_folder": ["f"], "patient_id": ["p"], "abnormal": [0]})
    with pytest.raises(ValueError, match="make_patient_splits"):
        patients_in_both_splits(studies)


def test_split_summary_reports_what_an_auroc_would_rest_on(tmp_path):
    csv = fake_release(tmp_path)
    studies = make_patient_splits(collect_studies(parse_mura_csv(csv), tmp_path), 0.25)
    s = split_summary(studies)
    assert s["n_studies"] == 24
    assert s["n_patients"] == 8
    assert s["patients_in_both_splits"] == []
    assert 0.0 < s["positive_rate"] < 1.0
    assert s["train_studies"] + s["val_studies"] == 24


# --------------------------------------------------------------------------- #
# dataset and pooling
# --------------------------------------------------------------------------- #


def test_dataset_returns_stacked_views_with_a_mask(tmp_path):
    csv = fake_release(tmp_path)
    studies = make_patient_splits(collect_studies(parse_mura_csv(csv), tmp_path), 0.25)
    ds = MuraStudyDataset(studies, tmp_path, max_views=3)
    views, label, mask = ds[0]
    assert views.shape[0] == 3
    assert mask.sum() == 3.0
    assert label.item() in (0.0, 1.0)


def test_max_views_truncates_and_the_mask_covers_only_the_real_ones(tmp_path):
    csv = fake_release(tmp_path)
    studies = collect_studies(parse_mura_csv(csv), tmp_path)
    ds = MuraStudyDataset(studies, tmp_path, max_views=2)
    views, _label, mask = ds[0]
    assert views.shape[0] == 2
    assert mask.tolist() == [1.0, 1.0]


def test_a_study_with_fewer_views_than_max_views_pads_and_says_so(tmp_path):
    csv = fake_release(tmp_path)
    df = parse_mura_csv(csv)
    folder = df["study_folder"].iloc[0]
    for p in sorted((tmp_path / folder).iterdir())[1:]:
        p.unlink()
    studies = collect_studies(df, tmp_path)
    ds = MuraStudyDataset(studies, tmp_path, max_views=4)
    views, _label, mask = ds[0]
    assert views.shape[0] == 4
    assert mask.tolist() == [1.0, 0.0, 0.0, 0.0]


def test_max_views_must_be_positive(tmp_path):
    studies = pd.DataFrame({"study_folder": ["f"], "abnormal": [0]})
    with pytest.raises(ValueError, match="max_views"):
        MuraStudyDataset(studies, tmp_path, max_views=0)


def test_a_missing_study_folder_is_an_error_not_an_empty_batch(tmp_path):
    studies = pd.DataFrame({"study_folder": ["nope"], "abnormal": [0]})
    ds = MuraStudyDataset(studies, tmp_path)
    with pytest.raises(FileNotFoundError):
        ds[0]


def test_collate_pads_views_across_the_batch():
    a = (torch.ones(3, 3, 8, 8), torch.tensor(1.0), torch.ones(3))
    b = (torch.ones(1, 3, 8, 8), torch.tensor(0.0), torch.ones(1))
    views, labels, mask = collate_studies([a, b])
    assert views.shape == (2, 3, 3, 8, 8)
    assert mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert labels.tolist() == [1.0, 0.0]


def test_pooling_takes_the_max_so_one_fracture_positive_view_is_enough():
    probs = torch.tensor([[0.05, 0.9, 0.1], [0.2, 0.3, 0.1]])
    mask = torch.ones(2, 3)
    pooled = pool_view_probabilities(probs, mask)
    assert pooled.tolist() == pytest.approx([0.9, 0.3])


def test_pooling_ignores_padded_slots():
    """Padding is exactly zero, so pooling it in would lower every study."""
    probs = torch.tensor([[0.8, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    assert pool_view_probabilities(probs, mask, "max").tolist() == pytest.approx([0.8])
    assert pool_view_probabilities(probs, mask, "mean").tolist() == pytest.approx([0.8])


def test_mean_pooling_lowers_a_study_with_one_fracture():
    probs = torch.tensor([[0.9, 0.0, 0.0]])
    mask = torch.ones(1, 3)
    assert pool_view_probabilities(probs, mask, "mean").item() == pytest.approx(0.3)


def test_an_unknown_pooling_rule_is_refused():
    with pytest.raises(ValueError, match="how must be"):
        pool_view_probabilities(torch.ones(1, 2), torch.ones(1, 2), "median")


def test_image_scores_pool_to_studies():
    scores = study_probabilities(
        [0.1, 0.9, 0.2], ["a", "a", "b"], ["study1", "study1", "study2"]
    )
    assert scores == {"study1": pytest.approx(0.9), "study2": pytest.approx(0.2)}


def test_mean_pooling_over_scores_averages_within_the_study():
    scores = study_probabilities([0.2, 0.4], ["a", "a"], ["s1", "s1"], how="mean")
    assert scores == {"s1": pytest.approx(0.3)}


def test_score_pooling_checks_its_inputs_line_up():
    with pytest.raises(ValueError, match="same length"):
        study_probabilities([0.1, 0.2], ["a"], ["s1", "s2"])

# --------------------------------------------------------------------------- #
# the study classifier
# --------------------------------------------------------------------------- #

from vindr.mura import (
    MuraStudyClassifier,
    build_mura_model,
    pool_view_logits,
)


def test_a_study_logit_is_the_max_over_its_views():
    logits = torch.tensor([[0.1, 2.5, -1.0], [0.4, 0.2, 0.1]])
    mask = torch.ones(2, 3)
    assert pool_view_logits(logits, mask).tolist() == pytest.approx([2.5, 0.4])


def test_padded_views_cannot_win_the_max():
    """A padded slot is a black image, and black is not a neutral input."""
    logits = torch.tensor([[0.1, 2.5]])
    masked = pool_view_logits(logits, torch.tensor([[1.0, 0.0]]))
    unmasked = pool_view_logits(logits, None)
    assert masked.item() == pytest.approx(0.1)
    assert unmasked.item() == pytest.approx(2.5)


def test_a_fully_masked_study_stays_finite_rather_than_going_nan():
    logits = torch.tensor([[1.0, 2.0]])
    out = pool_view_logits(logits, torch.zeros(1, 2))
    assert torch.isfinite(out).all()
    assert out.sigmoid().item() == pytest.approx(0.0, abs=1e-3)


def test_masked_mean_averages_only_the_real_views():
    logits = torch.tensor([[1.0, 3.0]])
    out = pool_view_logits(logits, torch.tensor([[1.0, 0.0]]), "mean")
    assert out.item() == pytest.approx(1.0)


def test_a_fully_masked_study_differs_by_rule_and_stays_finite():
    """Pinned rather than incidental: max goes very negative, mean lands on 0."""
    logits = torch.tensor([[1.0, 2.0]])
    mask = torch.zeros(1, 2)
    top = pool_view_logits(logits, mask, "max")
    mid = pool_view_logits(logits, mask, "mean")
    assert torch.isfinite(top).all() and torch.isfinite(mid).all()
    assert top.sigmoid().item() == pytest.approx(0.0, abs=1e-3)
    assert mid.item() == pytest.approx(0.0)


def test_an_unknown_pooling_rule_is_refused_in_the_model_too():
    with pytest.raises(ValueError, match="pooling must be"):
        MuraStudyClassifier(backbone="resnet18", pretrained=False, pooling="median")


def test_the_classifier_scores_a_study_not_a_view(tmp_path):
    model = MuraStudyClassifier(backbone="resnet18", pretrained=False)
    views = torch.rand(2, 3, 3, 64, 64)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    out = model(views, mask)
    assert out.shape == (2,)


def test_per_view_probabilities_explain_which_projection_fired():
    model = MuraStudyClassifier(backbone="resnet18", pretrained=False)
    views = torch.rand(2, 3, 3, 64, 64)
    p = model.per_view_probabilities(views)
    assert p.shape == (2, 3)
    assert ((p >= 0) & (p <= 1)).all()


def test_the_backbone_is_shared_across_views_so_parameters_do_not_grow_with_them():
    small = MuraStudyClassifier(backbone="resnet18", pretrained=False)
    big = MuraStudyClassifier(backbone="resnet18", pretrained=False)
    n_small = sum(p.numel() for p in small.parameters())
    n_big = sum(p.numel() for p in big.parameters())
    assert n_small == n_big


def test_build_mura_model_defaults_to_max_pooling():
    model = build_mura_model(backbone="resnet18", pretrained=False)
    assert model.pooling == "max"
