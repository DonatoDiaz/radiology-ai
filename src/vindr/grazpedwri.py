"""Phase 6 — pediatric fracture boxes from GRAZPEDWRI (Pascal VOC XML to YOLO).

GRAZPEDWRI ships one XML per radiograph in Pascal VOC form rather than COCO, so
Phase 2's `coco2yolo.py` does not apply and this converter is separate. The
model side is reused unchanged: ultralytics YOLO consumes what this writes, via
the same `configs/det_*.yaml` shape as Phase 2.

Three things about the release drive the code:

* **The box coordinates are not consistent across the release.** Some copies
  carry pixel coordinates, some carry values already normalized to 0–1. Both look
  like four numbers under `<bndbox>`, so :func:`parse_annotation` infers which
  from the magnitude and normalizes either. Getting this wrong does not fail
  loudly — it produces boxes a few pixels wide that still "train", which is the
  worst kind of bug to notice later.
* **Object names are anatomical free text** (`radius fracture`, `text`, and
  annotations that are not fractures at all). The class list is therefore built
  from the data and printed, never assumed, so a reviewer sees what the model is
  actually being taught before spending GPU hours on it.
* **Pediatric patients contribute several studies**, the same leakage shape as
  MURA in `vindr.mura`. With the release's study CSV the split is per patient;
  without it, :func:`assign_splits` falls back to per-image and the caller is
  told, because a per-image split on multi-study patients inflates mAP50.

Nothing here is validated: no GRAZPEDWRI study has been trained on.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

NOT_CLAIM = "не диагноз: критерий для проверки врачом"

#: VOC coordinate names, in the order this module stores them
BOX_KEYS = ("xmin", "ymin", "xmax", "ymax")

#: an object name containing any of these is not a fracture finding
NON_FRACTURE_HINTS = ("text", "normal", "none", "unremarkable", "artifact")


def _box_is_normalized(values: list[float]) -> bool:
    """True when the four coordinates look already scaled to 0-1.

    A pixel box on any real radiograph exceeds 1 somewhere, so a maximum at or
    below 1 can only be normalized data. This is the one inference that has to
    be made from the payload itself, since VOC carries no units.
    """
    return max(values) <= 1.0


def parse_annotation(
    xml_path: str | Path,
    image_size: tuple[int, int] | None = None,
) -> tuple[list[str], list[tuple[str, float, float, float, float]]]:
    """Return (object names, boxes) from one Pascal VOC XML.

    Boxes come back as `(name, xmin, ymin, xmax, ymax)` with 1-based inclusive
    VOC coordinates kept as they are, because the conversion to YOLO's
    center/size happens in :func:`voc_box_to_yolo` where the image size is
    known. A box with `xmax <= xmin` is dropped rather than normalized: VOC
    writers sometimes emit reversed corners, and dividing by a negative width
    would hand the trainer a box on the wrong side of the image.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return [], []
    names: list[str] = []
    boxes: list[tuple[str, float, float, float, float]] = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        box_el = obj.find("bndbox")
        if name_el is None or box_el is None:
            continue
        name = (name_el.text or "").strip()
        if not name:
            continue
        try:
            values = [float(box_el.find(k).text) for k in BOX_KEYS]
        except (AttributeError, TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in values):
            continue
        xmin, ymin, xmax, ymax = values
        if xmax <= xmin or ymax <= ymin:
            continue
        names.append(name)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return names, boxes


def voc_box_to_yolo(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
    normalized: bool | None = None,
) -> tuple[float, float, float, float] | None:
    """Convert one VOC box to YOLO center/size, or None if it cannot be done.

    VOC is 1-based and inclusive, YOLO is 0-based and relative: a box covering
    the whole image is `xmin=1, xmax=W`, which is a width of `W-1`, not `W`.

    A box reaching outside the radiograph is clipped to the visible part first.
    Clipping matters for more than tidiness — shrinking such a box around its
    centre instead collapses it to nothing as soon as the centre itself falls
    outside, which throws away an object that is plainly visible. The lower
    bound is 1 rather than 0 for pixel coordinates because VOC has no row 0.

    `normalized` forces the units; left to None it is inferred as in
    :func:`_box_is_normalized`. A box with no visible part at all is dropped.
    """
    xmin, ymin, xmax, ymax = box
    if normalized is None:
        normalized = _box_is_normalized([xmin, ymin, xmax, ymax])
    if normalized:
        w, h = 1.0, 1.0
        lo_x, lo_y = 0.0, 0.0
    else:
        w, h = float(width), float(height)
        lo_x, lo_y = 1.0, 1.0
    if w <= 0 or h <= 0:
        return None
    xmin, xmax = max(xmin, lo_x), min(xmax, w)
    ymin, ymax = max(ymin, lo_y), min(ymax, h)
    if xmax <= xmin or ymax <= ymin:
        return None
    cx, cy = ((xmin + xmax) / 2) / w, ((ymin + ymax) / 2) / h
    bw, bh = (xmax - xmin) / w, (ymax - ymin) / h
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    bw = min(bw, 2 * min(cx, 1 - cx))
    bh = min(bh, 2 * min(cy, 1 - cy))
    if bw <= 0 or bh <= 0:
        return None
    return cx, cy, bw, bh


def image_size_from_xml(xml_path: str | Path) -> tuple[int, int] | None:
    """`<size><width>/<height>` from the XML, when the writer recorded it."""
    try:
        size = ET.parse(xml_path).getroot().find("size")
    except ET.ParseError:
        return None
    if size is None:
        return None
    try:
        return int(float(size.find("width").text)), int(float(size.find("height").text))
    except (AttributeError, TypeError, ValueError):
        return None


def is_fracture(name: str) -> bool:
    """Whether an object name denotes a fracture rather than a note or normal."""
    low = name.lower()
    return not any(hint in low for hint in NON_FRACTURE_HINTS)


def build_class_map(names: list[str]) -> dict[str, int]:
    """Sorted, contiguous class ids — sorted so ids do not depend on file order.

    Two runs over the same annotations have to produce the same ids, otherwise a
    `data.yaml` saved by one run silently disagrees with the weights from
    another.
    """
    return {name: i for i, name in enumerate(sorted(set(names)))}


def image_files(image_dir: str | Path) -> list[Path]:
    """Radiographs with a matching XML, sorted for determinism."""
    image_dir = Path(image_dir)
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    return sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)


def assign_splits(
    images: list[Path],
    groups: dict[str, str] | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, str]:
    """Split by `groups[image]` when a study mapping is given, else per image.

    The key is the patient or study id when one is known and the image path
    otherwise, so the caller cannot accidentally get a per-image split by
    passing an empty dict — an empty `groups` falls back to the image name,
    which is the documented weaker behaviour.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    out: dict[str, str] = {}
    for p in images:
        key = (groups or {}).get(p.name, p.name)
        h = int(hashlib.sha256(f"{key}{seed}".encode()).hexdigest(), 16)
        out[p.name] = "val" if h % 10000 < val_fraction * 10000 else "train"
    return out


def groups_in_both_splits(splits: dict[str, str], groups: dict[str, str]) -> list[str]:
    """Keys landing on both sides — empty unless the split was per image."""
    if not groups:
        return []
    by_key: dict[str, set[str]] = {}
    for name, split in splits.items():
        by_key.setdefault(groups.get(name, name), set()).add(split)
    return sorted(k for k, v in by_key.items() if len(v) > 1)


def convert_release(
    image_dir: str | Path,
    out_dir: str | Path,
    groups: dict[str, str] | None = None,
    val_fraction: float = 0.15,
    seed: int = 42,
    fractures_only: bool = True,
    link_images: bool = True,
) -> dict:
    """Convert a GRAZPEDWRI-style directory into a YOLO dataset.

    The output is the layout ultralytics resolves: `<out>/images/<split>` for
    radiographs and `<out>/labels/<split>` for the matching `.txt` files, with
    `data.yaml` naming `path: <out>` and `train: images/train`. That last part
    is the whole reason the images are linked into the tree — ultralytics globs
    the directory `train:` points at for images, so a `data.yaml` naming the
    label directory collects zero images and trains on nothing, silently.

    Images are symlinked rather than copied: the release is tens of gigabytes
    and duplicating it to write a label file is not a trade worth making. Pass
    `link_images=False` when the filesystem has no usable symlinks.

    Negative radiographs are written as empty label files, which is what keeps a
    study with no fracture in the training set instead of silently vanishing.
    """
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    images = image_files(image_dir)
    if not images:
        raise FileNotFoundError(f"no radiographs under {image_dir}")

    splits = assign_splits(images, groups, val_fraction, seed)
    leaked = groups_in_both_splits(splits, groups or {})
    all_names: list[str] = []
    records: dict[str, list[str]] = {}

    for img in images:
        xml = img.with_suffix(".xml")
        if not xml.exists():
            xml = image_dir / "annotations" / f"{img.stem}.xml"
        if not xml.exists():
            records[img.name] = []
            continue
        _names, boxes = parse_annotation(xml)
        size = image_size_from_xml(xml)
        lines: list[tuple[str, float, float, float, float]] = []
        for name, xmin, ymin, xmax, ymax in boxes:
            if fractures_only and not is_fracture(name):
                continue
            width, height = size if size else _pill_size(img)
            if width is None or height is None:
                continue
            yolo = voc_box_to_yolo((xmin, ymin, xmax, ymax), width, height)
            if yolo is None:
                continue
            all_names.append(name)
            lines.append((name, *yolo))
        records[img.name] = lines

    class_map = build_class_map(all_names)
    n_boxes = 0
    n_negative = 0
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
    for name, lines in records.items():
        split = splits[name]
        label_dir = out_dir / "labels" / split
        label_dir.mkdir(parents=True, exist_ok=True)
        text = "\n".join(
            f"{class_map[n]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}" for n, cx, cy, bw, bh in lines
        )
        (label_dir / f"{Path(name).stem}.txt").write_text(text + ("\n" if text else ""))
        n_boxes += len(lines)
        if not lines:
            n_negative += 1
        if link_images:
            img_dst = out_dir / "images" / split / name
            img_dst.parent.mkdir(parents=True, exist_ok=True)
            if not img_dst.exists():
                img_dst.symlink_to((image_dir / name).resolve())

    names_yaml = {i: n for n, i in class_map.items()}
    data = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": names_yaml,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))

    n_train = sum(1 for s in splits.values() if s == "train")
    n_val = sum(1 for s in splits.values() if s == "val")
    return {
        "n_images": len(images),
        "n_boxes": n_boxes,
        "n_negative_images": n_negative,
        "n_classes": len(class_map),
        "classes": names_yaml,
        "train_images": n_train,
        "val_images": n_val,
        "empty_splits": [s for s, n in (("train", n_train), ("val", n_val)) if n == 0],
        "groups_in_both_splits": leaked,
        "split_basis": "groups" if groups else "image",
        "images_linked": link_images,
    }


def _pill_size(image_path: Path) -> tuple[int | None, int | None]:
    """(width, height) from the image itself, or (None, None) if unreadable."""
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(image_path) as im:
            return im.size
    except (OSError, UnidentifiedImageError):
        return None, None