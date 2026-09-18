"""Label and dataset helpers for VinDr-CXR."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

# 22 findings + 6 non-finding labels, order matches train.csv columns.
FINDINGS = [
    "Aortic enlargement",
    "Atelectasis",
    "Calcification of the Aorta",
    "Cardiomegaly",
    "Clavicle fracture",
    "Consolidation",
    "Emphysema",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Rib fracture",
    "Other fracture",
    "Mediastinal shift",
    "Tracheal deviation",
    "Ill-defined opacity",
    "Parenchymal bands",
    "Pulmonary cyst",
    "Scarring",
    "Subcutaneous emphysema",
    "Hilar lymphadenopathy",
    "Lung cavity",
]

NO_FINDING = "No finding"

ALL_LABELS = [NO_FINDING] + FINDINGS

# Lung-focused subset: parenchyma / pleura / airway pathologies.
# Excludes bones, vasculature, aorta & cardiac labels to keep the
# first release strictly "lung radiology" (легкие).
LUNG_FINDINGS = [
    "Atelectasis",
    "Consolidation",
    "Emphysema",
    "Infiltration",
    "Lung Opacity",
    "Nodule/Mass",
    "Other lesion",
    "Pleural effusion",
    "Pleural thickening",
    "Pneumothorax",
    "Pulmonary fibrosis",
    "Ill-defined opacity",
    "Parenchymal bands",
    "Pulmonary cyst",
    "Scarring",
    "Subcutaneous emphysema",
    "Lung cavity",
]

LUNG_LABELS = [NO_FINDING] + LUNG_FINDINGS


def load_train_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read train.csv and return a DataFrame with a binary label matrix."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"image_id": "image_id"}).copy()
    for col in ALL_LABELS:
        df[col] = df[col].astype(int)
    # Unknown / -1 values for crack lesions are not part of the public 28 classes.
    df = df[df[ALL_LABELS].sum(axis=1) >= 1]
    return df.reset_index(drop=True)


def restrict_to_lung(df: pd.DataFrame) -> pd.DataFrame:
    """Drop objects with no lung finding and keep only lung label columns."""
    df = df[df[LUNG_LABELS].sum(axis=1) >= 1].reset_index(drop=True)
    keep = [c for c in df.columns if c in LUNG_LABELS] + ["image_id"]
    return df[[*keep]].copy()


def load_meta_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read train_meta.csv, returns modality filters.k"""
    meta = pd.read_csv(csv_path)
    if "modality" not in meta.columns:
        meta["modality"] = "X-ray"
    return meta


def make_splits(df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Add a deterministic split derived from the image hash so splits are stable."""
    split = []
    for img in df["image_id"]:
        h = int(hashlib.sha256(f"{img}{seed}".encode()).hexdigest(), 16)
        split.append("val" if h % 100 < val_fraction * 100 else "train")
    df = df.copy()
    df["split"] = split
    return df