"""Tests for Phase 6 — GRAZPEDWRI Pascal VOC to YOLO conversion.

The fake release below deliberately mixes the two coordinate conventions the
real one has shipped in, because that mix is invisible in a test that only ever
sees one of them: a box read with the wrong units is still a box, it is just
1000x too small, and the trainer will happily train on it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vindr.grazpedwri import (
    BOX_KEYS,
    NON_FRACTURE_HINTS,
    assign_splits,
    build_class_map,
    convert_release,
    groups_in_both_splits,
    image_files,
    image_size_from_xml,
    is_fracture,
    parse_annotation,
    voc_box_to_yolo,
)


def _xml(path: Path, objects: list[tuple[str, tuple[float, float, float, float]]],
         size: tuple[int, int] | None = (200, 100)) -> Path:
    w, h = size or (0, 0)
    parts = ["<annotation>"]
    if size:
        parts.append(f"<size><width>{w}</width><height>{h}</height><depth>8</depth></size>")
    for name, (xmin, ymin, xmax, ymax) in objects:
        box = "".join(
            f"<{k}>{v}</{k}>" for k, v in zip(BOX_KEYS, (xmin, ymin, xmax, ymax), strict=True)
        )
        parts.append(f"<object><name>{name}</name><bndbox>{box}</bndbox></object>")
    parts.append("</annotation>")
    path.write_text("".join(parts))
    return path


def fake_release(root: Path, n: int = 6) -> Path:
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (100, 200), dtype=np.uint8)).save(
            image_dir / f"img{i}.png"
        )
        if i % 3 == 0:
            _xml(image_dir / f"img{i}.xml",
                 [("radius fracture", (10.0, 20.0, 60.0, 70.0))])
        elif i % 3 == 1:
            _xml(image_dir / f"img{i}.xml",
                 [("radius fracture", (10.0, 20.0, 60.0, 70.0)),
                  ("text", (5.0, 5.0, 40.0, 15.0))])
        else:
            _xml(image_dir / f"img{i}.xml", [])
    return image_dir


# --------------------------------------------------------------------------- #
# reading the XML
# --------------------------------------------------------------------------- #


def test_names_and_boxes_come_back(tmp_path):
    xml = _xml(tmp_path / "a.xml", [("radius fracture", (10.0, 20.0, 60.0, 70.0))])
    names, boxes = parse_annotation(xml)
    assert names == ["radius fracture"]
    assert boxes == [("radius fracture", 10.0, 20.0, 60.0, 70.0)]


def test_image_size_is_read_when_the_writer_recorded_it(tmp_path):
    xml = _xml(tmp_path / "a.xml", [], size=(640, 480))
    assert image_size_from_xml(xml) == (640, 480)


def test_a_missing_size_is_none_rather_than_zero(tmp_path):
    xml = _xml(tmp_path / "a.xml", [], size=None)
    assert image_size_from_xml(xml) is None


def test_malformed_xml_yields_no_objects_rather_than_raising(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("<annotation><object><name>x</name>")
    names, boxes = parse_annotation(xml)
    assert names == [] and boxes == []


def test_an_object_without_a_box_is_skipped(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("<annotation><object><name>text</name></object></annotation>")
    assert parse_annotation(xml) == ([], [])


def test_an_empty_name_is_skipped(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("<annotation><object><name>  </name><bndbox>"
                   "<xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax>"
                   "</bndbox></object></annotation>")
    assert parse_annotation(xml) == ([], [])


def test_reversed_corners_are_dropped_not_normalized(tmp_path):
    """Dividing by a negative width would put the box on the wrong side."""
    xml = _xml(tmp_path / "a.xml", [("radius fracture", (60.0, 70.0, 10.0, 20.0))])
    _names, boxes = parse_annotation(xml)
    assert boxes == []


def test_a_non_numeric_box_is_skipped(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text("<annotation><object><name>radius fracture</name><bndbox>"
                   "<xmin>a</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax>"
                   "</bndbox></object></annotation>")
    _names, boxes = parse_annotation(xml)
    assert boxes == []


# --------------------------------------------------------------------------- #
# the coordinate trap
# --------------------------------------------------------------------------- #


def test_pixel_boxes_are_read_as_pixels():
    yolo = voc_box_to_yolo((10, 20, 60, 70), 200, 100)
    assert yolo == pytest.approx((0.175, 0.45, 0.25, 0.5))


def test_normalized_boxes_are_not_read_as_pixels():
    """A 0-1 box read as pixels comes out a thousand times too small."""
    yolo = voc_box_to_yolo((0.05, 0.2, 0.3, 0.7), 200, 100)
    assert yolo == pytest.approx((0.175, 0.45, 0.25, 0.5))


def test_the_two_conventions_agree_once_normalized():
    pixels = voc_box_to_yolo((10, 20, 60, 70), 200, 100)
    units = voc_box_to_yolo((0.05, 0.2, 0.3, 0.7), 200, 100)
    assert pixels == pytest.approx(units)


def test_the_convention_can_be_forced():
    """Forcing the wrong one gives a different answer, so the inference has teeth."""
    assert voc_box_to_yolo((10, 20, 60, 70), 200, 100, normalized=True) != pytest.approx(
        voc_box_to_yolo((10, 20, 60, 70), 200, 100, normalized=False)
    )


def test_a_whole_image_box_fills_the_image():
    """VOC is 1-based and inclusive, so width is W, not W-1."""
    yolo = voc_box_to_yolo((1, 1, 200, 100), 200, 100, normalized=False)
    assert yolo[2] == pytest.approx(199 / 200)
    assert yolo[3] == pytest.approx(99 / 100)


def test_an_out_of_bounds_box_is_clipped_to_the_visible_part():
    """Clipping beats shrinking around the centre: this box is plainly visible."""
    yolo = voc_box_to_yolo((-20, -10, 400, 300), 200, 100, normalized=False)
    assert 0.0 <= yolo[0] <= 1.0 and 0.0 <= yolo[1] <= 1.0
    assert yolo[2] == pytest.approx(0.995)
    assert yolo[3] == pytest.approx(0.99)


def test_a_box_hanging_off_one_edge_keeps_its_visible_half():
    yolo = voc_box_to_yolo((1, 1, 101, 50), 200, 100, normalized=False)
    assert yolo[2] == pytest.approx(100 / 200)
    assert yolo[3] == pytest.approx(49 / 100)


def test_a_box_that_leaves_the_image_entirely_is_dropped():
    assert voc_box_to_yolo((500, 500, 600, 600), 200, 100, normalized=False) is None


def test_a_zero_sized_image_cannot_produce_a_box():
    assert voc_box_to_yolo((1, 1, 10, 10), 0, 0) is None


# --------------------------------------------------------------------------- #
# classes
# --------------------------------------------------------------------------- #


def test_non_fracture_objects_are_recognized():
    assert not is_fracture("text")
    assert not is_fracture("normal")
    assert is_fracture("radius fracture")


@pytest.mark.parametrize("hint", sorted(NON_FRACTURE_HINTS))
def test_every_non_fracture_hint_is_actually_filtered(hint):
    assert not is_fracture(f"{hint} annotation")


def test_class_ids_are_sorted_so_two_runs_agree():
    a = build_class_map(["text", "radius fracture", "ulna fracture"])
    b = build_class_map(["ulna fracture", "text", "radius fracture"])
    assert a == b
    assert list(a.values()) == [0, 1, 2]


def test_class_ids_are_contiguous_from_zero():
    assert build_class_map(["c", "a", "b"]) == {"a": 0, "b": 1, "c": 2}


# --------------------------------------------------------------------------- #
# splitting
# --------------------------------------------------------------------------- #


def test_splits_are_per_group_when_groups_are_known():
    images = [Path(f"img{i}.png") for i in range(20)]
    groups = {f"img{i}.png": f"patient{i // 3}" for i in range(20)}
    splits = assign_splits(images, groups, val_fraction=0.25)
    assert groups_in_both_splits(splits, groups) == []


def test_a_per_image_split_leaks_and_says_so():
    images = [Path(f"img{i}.png") for i in range(20)]
    groups = {f"img{i}.png": f"patient{i // 3}" for i in range(20)}
    splits = assign_splits(images, None, val_fraction=0.25)
    assert groups_in_both_splits(splits, groups)


def test_group_leak_check_needs_groups_to_mean_anything():
    images = [Path(f"img{i}.png") for i in range(10)]
    splits = assign_splits(images, None)
    assert groups_in_both_splits(splits, {}) == []


def test_the_split_is_deterministic():
    images = [Path(f"img{i}.png") for i in range(15)]
    a = assign_splits(images, None, seed=3)
    b = assign_splits(images, None, seed=3)
    assert a == b


def test_an_impossible_val_fraction_is_refused():
    with pytest.raises(ValueError, match="val_fraction"):
        assign_splits([Path("a.png")], None, val_fraction=0.0)


def test_image_files_are_sorted_for_determinism(tmp_path):
    d = tmp_path / "images"
    d.mkdir()
    for n in ("c.png", "a.png", "b.jpg"):
        Image.new("L", (8, 8)).save(d / n)
    (d / "notes.txt").write_text("ignore me")
    assert [p.name for p in image_files(d)] == ["a.png", "b.jpg", "c.png"]


# --------------------------------------------------------------------------- #
# the whole conversion
# --------------------------------------------------------------------------- #


def test_a_release_converts_and_writes_a_dataset_yaml(tmp_path):
    image_dir = fake_release(tmp_path, n=40)
    stats = convert_release(image_dir, tmp_path / "out")
    assert stats["n_images"] == 40
    assert stats["n_classes"] == 1
    assert (tmp_path / "out" / "data.yaml").exists()
    assert (tmp_path / "out" / "labels").exists()


def test_both_split_directories_exist_even_when_one_is_empty(tmp_path):
    """ultralytics resolves `val:` whether or not anything landed in it."""
    image_dir = fake_release(tmp_path, n=6)
    stats = convert_release(image_dir, tmp_path / "out")
    for split in ("train", "val"):
        assert (tmp_path / "out" / "images" / split).is_dir()
        assert (tmp_path / "out" / "labels" / split).is_dir()
    assert set(stats["empty_splits"]) <= {"train", "val"}
    assert (stats["empty_splits"] == []) == (stats["val_images"] > 0)


def test_an_empty_split_is_reported(tmp_path):
    image_dir = fake_release(tmp_path, n=6)
    stats = convert_release(image_dir, tmp_path / "out", val_fraction=0.5, seed=1)
    assert isinstance(stats["empty_splits"], list)


def test_negative_radiographs_get_an_empty_label_file(tmp_path):
    """Otherwise a study with no fracture vanishes from the training set."""
    image_dir = fake_release(tmp_path)
    convert_release(image_dir, tmp_path / "out")
    label_dir = tmp_path / "out" / "labels"
    negatives = [p for p in label_dir.rglob("*.txt") if p.read_text() == ""]
    assert len(negatives) == 2


def test_text_annotations_are_dropped_by_default(tmp_path):
    image_dir = fake_release(tmp_path)
    stats = convert_release(image_dir, tmp_path / "out")
    assert "text" not in stats["classes"]
    assert stats["n_boxes"] == 4


def test_non_fractures_can_be_kept_when_asked(tmp_path):
    image_dir = fake_release(tmp_path)
    stats = convert_release(image_dir, tmp_path / "out2", fractures_only=False)
    assert stats["n_classes"] == 2


def test_an_image_without_xml_becomes_an_empty_label(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("L", (100, 200)).save(image_dir / "lonely.png")
    stats = convert_release(image_dir, tmp_path / "out")
    assert stats["n_boxes"] == 0
    assert stats["n_negative_images"] == 1


def test_box_geometry_survives_the_round_trip(tmp_path):
    image_dir = fake_release(tmp_path)
    convert_release(image_dir, tmp_path / "out")
    label_dir = tmp_path / "out" / "labels"
    lines = [ln for p in label_dir.rglob("*.txt") for ln in p.read_text().splitlines() if ln]
    for ln in lines:
        _cls, cx, cy, bw, bh = ln.split()
        assert 0.0 <= float(cx) <= 1.0 and 0.0 <= float(cy) <= 1.0
        assert 0.0 < float(bw) <= 1.0 and 0.0 < float(bh) <= 1.0


def test_the_dataset_yaml_names_match_the_label_ids(tmp_path):
    import yaml

    image_dir = fake_release(tmp_path)
    convert_release(image_dir, tmp_path / "out")
    data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    assert set(data["names"]) == set(range(len(data["names"])))
    assert data["train"] and data["val"]


def test_the_yaml_points_at_directories_that_hold_images(tmp_path):
    """`train:` is globbed for images, so naming the label dir trains on nothing."""
    import yaml

    image_dir = fake_release(tmp_path, n=40)
    stats = convert_release(image_dir, tmp_path / "out")
    data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    for split in ("train", "val"):
        assert not (stats["empty_splits"] and split in stats["empty_splits"])
        img_dir = Path(data["path"]) / data[split]
        assert img_dir.is_dir(), f"{data[split]} does not exist"
        found = [p for p in img_dir.iterdir() if p.suffix.lower() in (".png", ".jpg")]
        assert found, f"{img_dir} holds no images"


def test_every_image_has_a_label_in_the_same_split(tmp_path):
    import yaml

    image_dir = fake_release(tmp_path)
    convert_release(image_dir, tmp_path / "out")
    data = yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())
    for split in ("train", "val"):
        img_dir = Path(data["path"]) / data[split]
        label_dir = tmp_path / "out" / "labels" / split
        stems = {p.stem for p in img_dir.iterdir() if p.suffix.lower() == ".png"}
        labels = {p.stem for p in label_dir.glob("*.txt")}
        assert stems == labels


def test_images_are_linked_not_copied(tmp_path):
    image_dir = fake_release(tmp_path)
    convert_release(image_dir, tmp_path / "out")
    linked = [p for p in (tmp_path / "out" / "images").rglob("*.png")]
    assert linked and all(p.is_symlink() for p in linked)
    assert all(p.stat().st_size > 0 for p in linked)


def test_image_linking_can_be_turned_off(tmp_path):
    image_dir = fake_release(tmp_path)
    stats = convert_release(image_dir, tmp_path / "out", link_images=False)
    assert stats["images_linked"] is False
    linked = list((tmp_path / "out" / "images").rglob("*.png"))
    assert linked == []


def test_converting_twice_does_not_fail_on_existing_links(tmp_path):
    image_dir = fake_release(tmp_path)
    convert_release(image_dir, tmp_path / "out")
    stats = convert_release(image_dir, tmp_path / "out")
    assert stats["n_images"] == 6


def test_an_empty_image_dir_is_an_error(tmp_path):
    (tmp_path / "images").mkdir()
    with pytest.raises(FileNotFoundError, match="no radiographs"):
        convert_release(tmp_path / "images", tmp_path / "out")


def test_a_patient_level_split_reports_no_leakage(tmp_path):
    image_dir = fake_release(tmp_path)
    groups = {f"img{i}.png": f"patient{i // 2}" for i in range(6)}
    stats = convert_release(image_dir, tmp_path / "out", groups=groups)
    assert stats["groups_in_both_splits"] == []
    assert stats["split_basis"] == "groups"