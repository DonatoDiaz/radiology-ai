"""Head CT volume loading and windowing (Phase 3, brain).

Supports three input flavours, all mapped to a Hounsfield-unit (HU) volume:

* DICOM series (pydicom) — the clinical source of truth.
* 16-bit PNG series + ``rescale_values.csv`` (RSNA IICH, MIT) — pixel value is
  mapped with the series' ``rescale_slope`` / ``rescale_intercept``.
* NIfTI volume (nibabel) — already in HU.

Windowing follows the conventional radiology presets (level / width in HU).
"""

from __future__ import annotations

import csv
import itertools
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# (level, width) in HU — conventional radiology presets
WINDOWS: dict[str, tuple[float, float]] = {
    "brain": (40.0, 80.0),          # grey matter / white matter contrast
    "subdural": (50.0, 130.0),      # acute blood at the convexity
    "stroke": (40.0, 100.0),        # early infarct / edema
    "bone": (500.0, 2000.0),        # skull, fractures
    "angio": (200.0, 600.0),        # vessels
    "lung": (-600.0, 1500.0),       # chest (Phase 6/7 reuse)
    "abdomen": (50.0, 400.0),       # soft abdomen (Phase 4 reuse)
    "sinus": (300.0, 700.0),        # bony sinuses (Phase 5 reuse)
}

# Blood on non-contrast CT, HU — used by the hemorrhage density check.
HU_BLOOD = (40.0, 100.0)
HU_ACUTE_BLOOD = (50.0, 90.0)
HU_EDEMA = (5.0, 30.0)
HU_HEMORRHAGE_LATER = (10.0, 40.0)  # chronic/subacute stages


@dataclass
class CTSeries:
    """A head CT volume: HU array (Z, Y, X) plus acquisition metadata."""

    volume: np.ndarray          # (slices, height, width) float32 in HU
    patient_id: str = ""
    study_id: str = ""
    series_id: str = ""
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)  # (z, y, x) in mm

    @property
    def n_slices(self) -> int:
        return int(self.volume.shape[0])

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.volume.shape)  # type: ignore[return-value]


def apply_window(volume: np.ndarray, window: str | tuple[float, float]) -> np.ndarray:
    """Map HU to uint8 using a named preset or an explicit (level, width)."""
    level, width = WINDOWS[window] if isinstance(window, str) else window
    lo, hi = level - width / 2.0, level + width / 2.0
    out = (np.asarray(volume, dtype=np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8)


def normalize_uint16_png(path: str | Path, slope: float = 1.0, intercept: float = 0.0) -> np.ndarray:
    """Read a 16-bit PNG slice and convert stored values to HU."""
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"cannot read slice: {path}")
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr.astype(np.float32) * float(slope) + float(intercept)


def read_rescale_table(csv_path: str | Path) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Parse ``rescale_values.csv`` -> {(patient, study, series): (slope, intercept)}.

    Accepts the common column spellings used by the RSNA PNG conversions.
    """
    table: dict[tuple[str, str, str], tuple[float, float]] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            keys = {k.lower(): v for k, v in row.items() if k}
            pid = keys.get("patient_id") or keys.get("patientid") or keys.get("patient")
            sid = keys.get("study_id") or keys.get("studyid") or keys.get("study")
            ser = keys.get("series_id") or keys.get("seriesid") or keys.get("series")
            slope = keys.get("rescale_slope") or keys.get("slope")
            inter = keys.get("rescale_intercept") or keys.get("intercept")
            if not (pid and sid and ser and slope is not None):
                continue
            table[(str(pid), str(sid), str(ser))] = (float(slope), float(inter or 0.0))
    return table


def load_png_series(
    slices: list[tuple[Path, float, float]],
    patient_id: str = "",
    study_id: str = "",
    series_id: str = "",
    slice_spacing: float = 1.0,
    pixel_spacing: tuple[float, float] = (1.0, 1.0),
) -> CTSeries:
    """Build a HU volume from (path, slope, intercept) triples, z-sorted by name.

    ``slope``/``intercept`` follow the DICOM convention
    ``HU = stored_pixel * rescale_slope + rescale_intercept``, which is what the
    RSNA PNG conversion stores in ``rescale_values.csv``.
    """
    ordered = sorted(slices, key=lambda s: s[0].name)
    planes = [normalize_uint16_png(p, slope, intercept) for p, slope, intercept in ordered]
    volume = np.stack(planes).astype(np.float32)
    lo, hi = float(volume.min()), float(volume.max())
    if hi < -500 or lo > 2000:
        log.warning(
            "study %s: HU range %.0f..%.0f looks wrong — check rescale_slope/intercept "
            "(expected roughly -1024 air to +1000+ bone)",
            study_id or series_id or "<unnamed>", lo, hi,
        )
    return CTSeries(
        volume=volume,
        patient_id=patient_id,
        study_id=study_id,
        series_id=series_id,
        spacing=(float(slice_spacing), float(pixel_spacing[0]), float(pixel_spacing[1])),
    )


def load_dicom_series(folder: str | Path) -> CTSeries:
    """Load a DICOM series into a HU volume sorted along the z axis."""
    import pydicom

    files = sorted(Path(folder).rglob("*"))
    datasets = []
    for f in files:
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(f, stop_before_pixels=True)
        except Exception as exc:  # noqa: BLE001  # non-DICOM or truncated file in the folder
            log.debug("skipping non-DICOM file %s: %s", f.name, exc)
            continue
        if "ImagePositionPatient" in ds:
            datasets.append((f, ds))
    if not datasets:
        raise FileNotFoundError(f"no DICOM slices found in {folder}")

    def z_of(item) -> float:
        _, ds = item
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception as exc:  # noqa: BLE001  # slice without a usable position
            log.debug("ImagePositionPatient[2] missing (%s); using z=0", exc)
            return 0.0

    datasets.sort(key=z_of)
    first = datasets[0][1]
    patient_id = str(getattr(first, "PatientID", ""))
    study_id = str(getattr(first, "StudyInstanceUID", ""))
    series_id = str(getattr(first, "SeriesInstanceUID", ""))

    planes = []
    for f, ds in datasets:
        full = pydicom.dcmread(f)
        planes.append(full.pixel_array.astype(np.float32))
    volume = np.stack(planes)

    # DICOM stores the slope/intercept in a nested tag — apply when present.
    slope, intercept = 1.0, 0.0
    try:
        slope = float(first.RescaleSlope)
        intercept = float(first.RescaleIntercept)
    except Exception as exc:  # noqa: BLE001  # no modality LUT: values are already HU
        log.debug("no RescaleSlope/Intercept (%s); assuming stored values are HU", exc)
    volume = volume * slope + intercept

    z_spacing = 1.0
    if len(datasets) > 1:
        zs = sorted(z_of(d) for d in datasets)
        diffs = [b - a for a, b in itertools.pairwise(zs) if b - a > 1e-6]
        if diffs:
            z_spacing = float(np.median(diffs))
    px = (1.0, 1.0)
    try:
        px = (float(first.PixelSpacing[0]), float(first.PixelSpacing[1]))
    except Exception as exc:  # noqa: BLE001  # spacing absent: fall back to 1 mm
        log.debug("no PixelSpacing (%s); assuming 1x1 mm", exc)
    return CTSeries(
        volume=volume.astype(np.float32),
        patient_id=patient_id,
        study_id=study_id,
        series_id=series_id,
        spacing=(z_spacing, px[0], px[1]),
    )


def load_nifti(path: str | Path) -> CTSeries:
    """Load a NIfTI volume (assumed HU, e.g. BraTS-derived)."""
    import nibabel as nib

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj).astype(np.float32)
    if data.ndim == 4:  # drop the singleton channel/time axis
        data = data[..., 0]
    # NIfTI stores (X, Y, Z); the rest of the pipeline is canonical (Z, Y, X).
    data = np.transpose(data, (2, 1, 0))
    x_sp, y_sp, z_sp = (float(z) for z in img.header.get_zooms()[:3])
    return CTSeries(volume=data, spacing=(z_sp, y_sp, x_sp))  # type: ignore[arg-type]


def resample_to(
    series: CTSeries,
    shape: tuple[int, int, int],
    order: int = 1,
) -> CTSeries:
    """Trilinear resize the volume to (slices, height, width); spacing is recomputed."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(series.volume)  # numpy (Z, Y, X) -> sitk (X, Y, Z)
    img.SetSpacing(
        (float(series.spacing[2]), float(series.spacing[1]), float(series.spacing[0]))
    )  # sitk spacing is (x, y, z)
    resized = sitk.ResampleImageFilter()
    resized.SetSize([int(shape[2]), int(shape[1]), int(shape[0])])  # sitk size is (x, y, z)
    resized.SetOutputPixelType(sitk.sitkFloat32)
    resized.SetInterpolator(sitk.sitkLinear if order >= 1 else sitk.sitkNearestNeighbor)
    out_img = resized.Execute(img)
    old_sp = np.asarray(series.spacing, dtype=np.float64)               # (z, y, x) mm
    old_sz = np.asarray(series.volume.shape, dtype=np.float64)          # (z, y, x) px
    new_sz = np.asarray(shape, dtype=np.float64)                        # (z, y, x) px
    new_sp = old_sp * np.divide(old_sz, new_sz, out=np.ones_like(old_sz), where=new_sz > 0)
    return CTSeries(
        volume=sitk.GetArrayFromImage(out_img).astype(np.float32),
        patient_id=series.patient_id,
        study_id=series.study_id,
        series_id=series.series_id,
        spacing=(float(new_sp[0]), float(new_sp[1]), float(new_sp[2])),
    )


def slice_with_neighbours(volume: np.ndarray, z: int, context: int = 1) -> np.ndarray:
    """2.5D input: the slice at ``z`` plus up to ``context`` neighbours per side."""
    z = int(np.clip(z, 0, volume.shape[0] - 1))
    planes = [volume[z]]
    for dz in range(1, context + 1):
        for idx in (z - dz, z + dz):
            if 0 <= idx < volume.shape[0]:
                planes.append(volume[idx])
    while len(planes) < 1 + 2 * context:  # pad at the volume edges
        planes.append(planes[-1])
    return np.stack(planes, axis=-1)  # (H, W, 2*context+1)
