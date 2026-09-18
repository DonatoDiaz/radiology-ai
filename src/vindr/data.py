"""Dataset / DataLoader for VinDr-CXR with DICOM or PNG images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    import pydicom

    HAVE_DICOM = True
except ImportError:  # pragma: no cover
    HAVE_DICOM = False


def read_dicom(path: Path) -> np.ndarray:
    """Pixel array from a DICOM file, rescaled using modality tags."""
    if not HAVE_DICOM:
        raise ImportError("pydicom is required to read DICOM files")
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    arr = arr * slope + intercept
    # Clip to robust range around window center.
    lo, hi = np.percentile(arr, (1, 99))
    arr = np.clip(arr, lo, hi)
    arr = (arr - lo) / (hi - lo + 1e-8)
    return (arr * 255).astype(np.uint8)


def read_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".dcm":
        return read_dicom(path)
    img = Image.open(path).convert("L")
    return np.asarray(img)


class VinDrDataset(Dataset):
    """Chest X-ray multi-label dataset.

    df: DataFrame from labels.make_splits (image_id, split, label columns).
    images_dir: directory containing the images (dicom or png).
    transforms: callable receiving (H, W) grayscale uint8 array -> tensor.
    labels: list of label column names (order defines output channels).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str | Path,
        labels: list[str],
        transforms=None,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.labels = labels
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.df)

    def _image_path(self, image_id: str) -> Path:
        for ext in (".dicom", ".dcm", ".png", ".jpg", ".jpeg"):
            p = self.images_dir / f"{image_id}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"image {image_id} not found under {self.images_dir}")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        arr = read_image(self._image_path(row["image_id"]))
        if self.transforms is not None:
            arr = self.transforms(image=arr)["image"]
        else:
            arr = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
            arr = arr.unsqueeze(0)
        target = torch.zeros(len(self.labels), dtype=torch.float32)
        for i, col in enumerate(self.labels):
            target[i] = float(row[col])
        return arr, target