"""Tests for scripts/prepare_head_ct.py (RSNA ICH -> study tree)."""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prepare_head_ct", ROOT / "scripts" / "prepare_head_ct.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["prepare_head_ct"] = mod
    spec.loader.exec_module(mod)
    return mod


prepare = _load_module()


def _write_fake_rsna(root: Path, n_studies: int = 6) -> tuple[Path, Path, Path]:
    """Create a miniature RSNA-style 16-bit PNG dataset."""
    import cv2

    png_dir = root / "png"
    png_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)
    types = list(prepare.HEMORRHAGE_TYPES)
    slice_rows, rescale_rows = [], []
    for s in range(n_studies):
        study = f"id_{s:032x}"
        n = int(rng.integers(3, 6))
        pos = s % 2 == 0
        label = ",".join(rng.choice(types, size=1)) if pos else ""
        for i in range(n):
            sid = f"{study}_{i:04d}"
            cv2.imwrite(
                str(png_dir / f"{sid}.png"), rng.integers(0, 4096, (32, 32)).astype(np.uint16)
            )
            slice_rows.append({"ID": sid, "hemorrhage_type": label})
            rescale_rows.append({"ID": sid, "rescale_slope": -1.0, "rescale_intercept": 1024.0})

    labels_csv = root / "slice_labels.csv"
    with open(labels_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "hemorrhage_type"])
        w.writeheader()
        w.writerows(slice_rows)
    rescale_csv = root / "rescale_values.csv"
    with open(rescale_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["ID", "rescale_slope", "rescale_intercept"])
        w.writeheader()
        w.writerows(rescale_rows)
    return png_dir, labels_csv, rescale_csv


def test_split_id_strips_slice_index():
    assert prepare.split_id("id_abc_0001") == "id_abc"
    assert prepare.split_id("id_abc_0012") == "id_abc"
    assert prepare.split_id("no_index") == "no_index"
    assert prepare.split_id("study_2023_0007") == "study_2023"


def test_read_slice_labels_groups_per_study(tmp_path):
    _png, labels, _rescale = _write_fake_rsna(tmp_path)
    grouped = prepare.read_slice_labels(labels)
    # only studies that carry a hemorrhage_type are present (negatives are absent)
    assert len(grouped) == 3
    for names in grouped.values():
        assert names and names <= set(prepare.HEMORRHAGE_TYPES)


def test_read_rescale_takes_median(tmp_path):
    _png, _labels, rescale = _write_fake_rsna(tmp_path)
    table = prepare.read_rescale(rescale)
    assert len(table) == 6
    for slope, intercept in table.values():
        assert slope == pytest.approx(-1.0)
        assert intercept == pytest.approx(1024.0)


def test_read_rescale_missing_file_returns_empty(tmp_path):
    assert prepare.read_rescale(tmp_path / "nope.csv") == {}


def test_index_pngs_orders_slices_numerically(tmp_path):
    png, _labels, _rescale = _write_fake_rsna(tmp_path)
    by_study = prepare.index_pngs(png)
    for files in by_study.values():
        nums = [int(f.stem.rpartition("_")[2]) for f in files]
        assert nums == sorted(nums), "slices must be ordered by slice index, not name"


def test_prepare_end_to_end(tmp_path, monkeypatch):
    png, labels, rescale = _write_fake_rsna(tmp_path, n_studies=6)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prepare_head_ct.py", "--png-dir", str(png), "--labels", str(labels),
        "--rescale", str(rescale), "--out", str(out), "--val-fraction", "0.34",
    ])
    prepare.main()

    studies_csv = out / "study_labels.csv"
    assert studies_csv.exists()
    with open(studies_csv) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    assert list(rows[0].keys()) == ["study_id", *prepare.HEMORRHAGE_TYPES, "any"]
    for r in rows:
        types = [r[t] for t in prepare.HEMORRHAGE_TYPES]
        assert r["any"] == ("1" if any(t == "1" for t in types) else "0")

    assert (out / "splits.txt").exists()
    splits = dict(
        line.split()[:2] for line in (out / "splits.txt").read_text().splitlines()
    )
    assert len(splits) == 6
    assert set(splits.values()) == {"train", "val"}
    assert (out / "rescale_values.csv").exists()

    # the produced tree must be loadable by the trainer
    from vindr.ct.dataset import load_study_folders

    loaded = load_study_folders(out / "studies")
    assert len(loaded) == 6
    assert all(s.n_slices > 0 for s in loaded)
    assert all(s.volume.ndim == 3 for s in loaded)


def test_prepare_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    png, labels, _rescale = _write_fake_rsna(tmp_path, n_studies=4)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prepare_head_ct.py", "--png-dir", str(png), "--labels", str(labels),
        "--out", str(out), "--dry-run",
    ])
    prepare.main()
    assert "dry run" in capsys.readouterr().out
    assert not out.exists()


def test_prepare_positive_only_filter(tmp_path, monkeypatch):
    png, labels, _rescale = _write_fake_rsna(tmp_path, n_studies=8)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "prepare_head_ct.py", "--png-dir", str(png), "--labels", str(labels),
        "--out", str(out), "--positive-only", "--max-studies", "3",
    ])
    prepare.main()
    with open(out / "study_labels.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert all(r["any"] == "1" for r in rows)


def test_prepare_rejects_contradictory_filters(tmp_path, monkeypatch):
    png, labels, _rescale = _write_fake_rsna(tmp_path, n_studies=2)
    monkeypatch.setattr(sys, "argv", [
        "prepare_head_ct.py", "--png-dir", str(png), "--labels", str(labels),
        "--out", str(tmp_path / "o"), "--positive-only", "--negative-only",
    ])
    with pytest.raises(SystemExit):
        prepare.main()
