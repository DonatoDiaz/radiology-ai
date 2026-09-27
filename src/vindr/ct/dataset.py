"""Datasets for head CT (Phase 3).

Two granularities:

* ``HeadCTSliceDataset``   — one 2.5D sample per slice (cheap, slice-level labels);
* ``HeadCTStudyDataset``   — a fixed set of slices per study (attention-pooled).

Both take a :class:`CTSeries` provider so the same code serves DICOM folders,
16-bit PNG series and NIfTI volumes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from vindr.ct.volume import (
    CTSeries,
    apply_window,
    load_nifti,
    resample_to,
    slice_with_neighbours,
)

log = logging.getLogger(__name__)

# per-slice window mix so the net sees grey matter, acute blood and bone
WINDOW_MIX: tuple[str, ...] = ("brain", "subdural", "stroke")


def _to_tensor(slice_stack: np.ndarray, window: str) -> torch.Tensor:
    """(H, W, 2c+1) HU planes -> (2c+1, H, W) float tensor in [0, 1]."""
    planes = [apply_window(slice_stack[..., i], window) for i in range(slice_stack.shape[-1])]
    arr = np.stack(planes).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


def _rotate_plane(plane: np.ndarray, angle: float, fill: float = -1000.0) -> np.ndarray:
    """Rotate a single HU plane in-plane, filling the border with ``fill`` (air).

    Masks pass ``fill=0.0`` so the rotated border stays background.
    """
    if abs(angle) < 1e-3:
        return plane
    h, w = plane.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    rotated = cv2.warpAffine(plane, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=fill)
    return rotated.reshape(h, w, -1) if plane.ndim == 3 else rotated


class HeadCTSliceDataset(Dataset):
    """One 2.5D sample per axial slice, with the slice's hemorrhage labels."""

    def __init__(
        self,
        studies: Sequence[CTSeries],
        labels: Sequence[Sequence[float]] | None = None,
        context: int = 1,
        target_shape: tuple[int, int, int] | None = (32, 224, 224),
        window: str | tuple[str, ...] = WINDOW_MIX,
        train: bool = False,
        augment: bool = False,
    ):
        self.train = train
        self.augment = augment
        self.context = context
        self.target_shape = target_shape
        self.windows = (window,) if isinstance(window, str) else tuple(window)
        self.series: list[CTSeries] = []
        self.index: list[tuple[int, int]] = []  # (study_idx, slice_z)
        for si, st in enumerate(studies):
            st = resample_to(st, target_shape) if target_shape else st
            self.series.append(st)
            for z in range(st.n_slices):
                self.index.append((si, z))
        self.labels = labels

    def __len__(self) -> int:
        return len(self.index)

    def _augment(self, planes: np.ndarray) -> np.ndarray:
        """In-plane augmentation of a (H, W, C) stack of HU planes.

        Slice-axis jitter is *not* done here: the channels are already the
        neighbouring slices, so it is applied to the sampled z in __getitem__.
        """
        if not self.augment or not self.train:
            return planes
        if np.random.rand() < 0.5:  # horizontal flip along the width axis
            planes = planes[:, ::-1, :]
        if np.random.rand() < 0.2:  # small in-plane rotation, +/-8 degrees
            angle = float(np.random.uniform(-8.0, 8.0))
            planes = np.stack(
                [_rotate_plane(planes[..., c], angle) for c in range(planes.shape[-1])], axis=-1
            )
        return planes

    def __getitem__(self, i: int):
        si, z = self.index[i]
        st = self.series[si]
        if self.augment and self.train and np.random.rand() < 0.3:  # z-jitter
            z = int(np.clip(z + np.random.choice([-1, 1]), 0, st.n_slices - 1))
        planes = slice_with_neighbours(st.volume, z, context=self.context)
        planes = self._augment(planes)
        window = self.windows[i % len(self.windows)]
        x = _to_tensor(planes, window)
        if self.labels is None:
            y = torch.zeros(1)
        else:
            y = torch.tensor(np.asarray(self.labels[si], dtype=np.float32))
        return x, y, si


class HeadCTStudyDataset(Dataset):
    """Fixed number of slices per study for attention-pooled study-level labels."""

    def __init__(
        self,
        studies: Sequence[CTSeries],
        labels: Sequence[Sequence[float]] | None = None,
        slices_per_study: int = 24,
        context: int = 1,
        target_shape: tuple[int, int, int] | None = (32, 224, 224),
        window: str | tuple[str, ...] = "brain",
        train: bool = False,
    ):
        self.train = train
        self.slices_per_study = slices_per_study
        self.context = context
        self.window = window
        self.target_shape = target_shape
        self.series = [resample_to(s, target_shape) if target_shape else s for s in studies]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.series)

    def __getitem__(self, i: int):
        st = self.series[i]
        n = st.n_slices
        k = min(self.slices_per_study, n)
        zs = np.linspace(0, n - 1, k).round().astype(int) if n > 1 else np.zeros(k, dtype=int)
        planes = np.stack([slice_with_neighbours(st.volume, int(z), self.context) for z in zs])
        windowed = np.stack([apply_window(planes[j], self.window) for j in range(len(zs))])
        # (S, H, W, C) -> (S, C, H, W), scaled to [0, 1]
        x = torch.from_numpy(np.ascontiguousarray(windowed.transpose(0, 3, 1, 2)).astype(np.float32) / 255.0)
        y = torch.zeros(1) if self.labels is None else torch.tensor(np.asarray(self.labels[i], dtype=np.float32))
        return x, y, i


def collate_slices(batch):
    xs, ys, gs = zip(*batch)
    return torch.stack(xs), torch.stack(ys), torch.tensor(gs)


def collate_studies(batch):
    """Pad the slice dimension so a batch of studies with different slice counts fits."""
    xs, ys, gs = zip(*batch)
    max_s = max(x.shape[0] for x in xs)
    out = []
    for x in xs:
        if x.shape[0] < max_s:
            pad = x[-1:].expand(max_s - x.shape[0], *x.shape[1:])
            x = torch.cat([x, pad], dim=0)
        out.append(x)
    return torch.stack(out), torch.stack(ys), torch.tensor(gs)


def _rescale_for_study(root: Path, study: str, table: dict[str, tuple[float, float]] | None):
    """Median (slope, intercept) for a study, from <root>/rescale_values.csv."""
    if not table:
        return 1.0, 0.0
    hit = table.get(study)
    return hit if hit is not None else (1.0, 0.0)


def load_study_folders(
    root: str | Path, rescale_csv: str | Path | None = None
) -> list[CTSeries]:
    """Load every study under ``root`` (one subfolder = one study).

    Per-folder formats, checked in this order: NIfTI, 16-bit PNG series,
    a single ``.npy`` HU volume, DICOM series. ``rescale_csv`` is the optional
    per-study ``rescale_values.csv`` from ``scripts/prepare_head_ct.py``.
    """
    from vindr.ct.volume import load_dicom_series, load_nifti, load_png_series

    root = Path(root)
    table: dict[str, tuple[float, float]] = {}
    rescale_path = Path(rescale_csv) if rescale_csv else root.parent / "rescale_values.csv"
    if rescale_path.exists():
        with open(rescale_path, newline="") as fh:
            import csv

            for row in csv.DictReader(fh):
                keys = {k.lower(): v for k, v in row.items() if k}
                study = keys.get("study_id") or keys.get("id")
                slope = keys.get("rescale_slope", keys.get("slope"))
                if not study or slope is None:
                    continue
                try:
                    table[str(study)] = (
                        float(slope),
                        float(keys.get("rescale_intercept", keys.get("intercept", 0.0)) or 0.0),
                    )
                except ValueError:
                    continue
        log.info("loaded rescale table for %d studies from %s", len(table), rescale_path)

    studies: list[CTSeries] = []
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        nii = list(entry.glob("*.nii*"))
        pngs = sorted(entry.glob("*.png"))
        npy = sorted(p for p in entry.glob("*.npy") if "mask" not in p.name.lower())
        try:
            if nii:
                st = load_nifti(nii[0])
            elif pngs:
                slope, intercept = _rescale_for_study(root, entry.name, table)
                st = load_png_series(
                    [(p, slope, intercept) for p in pngs], study_id=entry.name
                )
            elif npy:
                st = CTSeries(volume=np.load(npy[0]), study_id=entry.name)
            else:
                st = load_dicom_series(entry)
        except Exception as exc:  # noqa: BLE001  # skip unreadable study folders
            log.warning("skipping %s: %s", entry.name, exc)
            continue
        # the folder name is the study key used by study_labels.csv
        st.study_id = st.study_id or entry.name
        st.series_id = st.series_id or entry.name
        studies.append(st)
    return studies


def make_provider(root: str | Path) -> Callable[[], list[CTSeries]]:
    """Return a zero-arg callable that (re)loads all studies — for lazy workers."""
    return lambda: load_study_folders(root)


def _load_mask(entry: Path) -> np.ndarray | None:
    """Read a per-study mask as ``(Z, Y, X)`` or ``(Z, Y, X, C)`` float32 in [0, 1].

    Accepted layouts inside a study folder: ``mask.nii.gz``, ``mask.npy``,
    ``masks/*.png`` (sorted, non-zero = lesion) and ``<image>_mask.png``.
    """
    nifti = [p for p in (entry / "mask.nii", entry / "mask.nii.gz") if p.exists()]
    if nifti:
        arr = load_nifti(nifti[0]).volume
        return arr.astype(np.float32) if arr.ndim == 3 else None
    npy = entry / "mask.npy"
    if npy.exists():
        arr = np.load(npy)
        return arr.astype(np.float32) if arr.ndim in (3, 4) else None
    folder = entry / "masks"
    pngs = sorted(folder.glob("*.png")) if folder.is_dir() else []
    if not pngs:
        pngs = sorted(entry.glob("*_mask.png"))
    if pngs:
        planes = []
        for p in pngs:
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.ndim == 3:  # a multi-channel mask, one image per class
                raw = raw.transpose(2, 0, 1)
            planes.append(raw)
        if planes:
            return (np.stack(planes).astype(np.float32) > 0).astype(np.float32)
    return None


def load_masked_studies(
    root: str | Path, rescale_csv: str | Path | None = None
) -> list[tuple[CTSeries, np.ndarray]]:
    """Load studies that come with a segmentation mask.

    Same per-folder formats as :func:`load_study_folders`; studies without a
    readable mask are skipped, so a partially annotated dataset still trains.
    """
    from vindr.ct.volume import load_dicom_series, load_nifti, load_png_series

    root = Path(root)
    rescale_path = Path(rescale_csv) if rescale_csv else root.parent / "rescale_values.csv"
    table: dict[str, tuple[float, float]] = {}
    if rescale_path.exists():
        import csv

        with open(rescale_path, newline="") as fh:
            for row in csv.DictReader(fh):
                keys = {k.lower(): v for k, v in row.items() if k}
                study = keys.get("study_id") or keys.get("id")
                slope = keys.get("rescale_slope", keys.get("slope"))
                if study and slope is not None:
                    table[str(study)] = (float(slope), float(keys.get("intercept", 0.0) or 0.0))

    out: list[tuple[CTSeries, np.ndarray]] = []
    for entry in sorted(p for p in root.iterdir() if p.is_dir()):
        mask = _load_mask(entry)
        if mask is None:
            log.warning("skipping %s: no readable mask", entry.name)
            continue
        nii = [p for p in entry.glob("*.nii*") if "mask" not in p.name]
        pngs = sorted(p for p in entry.glob("*.png") if not p.name.endswith("_mask.png"))
        npy = [p for p in entry.glob("*.npy") if "mask" not in p.name.lower()]
        try:
            if nii:
                st = load_nifti(nii[0])
            elif pngs:
                slope, intercept = table.get(entry.name, (1.0, 0.0))
                st = load_png_series([(p, slope, intercept) for p in pngs], study_id=entry.name)
            elif npy:
                st = CTSeries(volume=np.load(npy[0]), study_id=entry.name)
            else:
                st = load_dicom_series(entry)
        except Exception as exc:  # noqa: BLE001  # skip unreadable study folders
            log.warning("skipping %s: %s", entry.name, exc)
            continue
        st.study_id = st.study_id or entry.name
        st.series_id = st.series_id or entry.name
        if mask.shape[0] != st.n_slices:
            log.warning(
                "skipping %s: mask has %d slices, image has %d",
                entry.name, mask.shape[0], st.n_slices,
            )
            continue
        if mask.ndim == 3 and mask.shape[-2:] != st.shape[-2:]:
            log.warning("skipping %s: mask shape %s != image %s", entry.name, mask.shape, st.shape)
            continue
        out.append((st, mask))
    if not out:
        log.warning("no masked studies found under %s", root)
    return out


class HeadCTSegDataset(Dataset):
    """One 2.5D sample per axial slice, paired with the slice's mask.

    ``masks`` maps study_id to a ``(Z, Y, X)`` or ``(Z, Y, X, C)`` array; the
    neighbours feed the input as channels while the target stays single-slice,
    so a multi-class mask is segmented per class.
    """

    def __init__(
        self,
        studies: Sequence[tuple[CTSeries, np.ndarray]],
        context: int = 1,
        target_shape: tuple[int, int, int] | None = (32, 224, 224),
        window: str | tuple[float, float] = "brain",
        train: bool = False,
        augment: bool = False,
    ):
        self.train = train
        self.augment = augment
        self.context = context
        self.target_shape = target_shape
        self.window = window
        self.items: list[tuple[CTSeries, np.ndarray, int, int]] = []
        for st, mask in studies:
            if target_shape:
                st = resample_to(st, target_shape)
                mask = _resample_mask(mask, st.shape)
            if mask.shape[0] != st.n_slices:
                # a mask shorter/longer than the series means misaligned inputs;
                # keep the common prefix rather than indexing out of range
                n = min(mask.shape[0], st.n_slices)
                log.warning(
                    "mask has %d slices but image has %d — truncating both to %d (%s)",
                    mask.shape[0], st.n_slices, n, st.study_id,
                )
                st = CTSeries(
                    volume=st.volume[:n], patient_id=st.patient_id, study_id=st.study_id,
                    series_id=st.series_id, spacing=st.spacing,
                )
                mask = mask[:n]
            if mask.ndim == 3:
                mask = mask[..., None]
            for z in range(st.n_slices):
                # skip empty slices: they dominate head CT and only teach the
                # net to predict nothing, which is already the easy answer
                if mask[z].max() <= 0:
                    continue
                self.items.append((st, mask, z, mask.shape[-1]))
        if not self.items:
            raise ValueError("no labelled slices found — check the mask folder layout")
        self.n_classes = self.items[0][3]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        st, mask, z, _c = self.items[i]
        planes = slice_with_neighbours(st.volume, z, context=self.context)
        target = mask[z].astype(np.float32)
        if self.augment and self.train:
            if np.random.rand() < 0.5:  # left-right flip is anatomically valid
                planes = planes[:, ::-1, :]
                target = target[:, ::-1]
            if np.random.rand() < 0.2:
                angle = float(np.random.uniform(-8.0, 8.0))
                planes = np.stack(
                    [_rotate_plane(planes[..., c], angle) for c in range(planes.shape[-1])], axis=-1
                )
                target = np.stack(
                    [_rotate_plane(target[..., c], angle, fill=0.0) for c in range(target.shape[-1])],
                    axis=-1,
                ).astype(np.float32)
        planes = np.ascontiguousarray(planes)
        target = np.ascontiguousarray(target)
        x = _to_tensor(planes, self.window)
        y = torch.from_numpy(target.transpose(2, 0, 1))  # (C, Y, X)
        return x, y


def _resample_mask(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbour resample of a mask onto the image grid (no new labels)."""
    n_z = min(mask.shape[0], shape[0])
    planes = [
        cv2.resize(mask[z], (shape[2], shape[1]), interpolation=cv2.INTER_NEAREST)
        for z in range(n_z)
    ]
    return np.stack(planes).astype(np.float32)


def collate_seg(batch):
    """Stack (x, y) pairs; all samples already share one spatial size."""
    xs = torch.stack([b[0] for b in batch])
    ys = torch.stack([b[1] for b in batch])
    return xs, ys
