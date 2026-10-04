"""Phase 6 — musculoskeletal X-ray: study-level fracture screening (MURA).

MURA ships as a tree rather than a flat image list — a patient has studies, a
study has series, a series has projections — and two things follow that shape
everything else in this module:

* **The label is not in the CSV.** The manifests carry patient, series, body
  part and laterality, but whether a study is abnormal lives in the folder
  name (`study1_positive`). Parsed by :func:`abnormal_from_path`, and a folder
  that says neither positive nor negative is an error rather than a normal:
  an unlabelled study counted as normal drags the AUROC down silently.
* **A patient contributes several studies.** MURA's 9,912 studies come from
  roughly half as many patients, so a split made per study puts the same arm in
  train and validation. That inflates the score without any new information,
  and the reported number stops meaning what it claims. The split here is
  therefore per *patient*, and :func:`patients_in_both_splits` exists so a
  caller can check the property instead of trusting it.

The task here is study-level abnormal/abnormal, not detection. MURA publishes no
fracture bounding boxes, so a box-based metric cannot be computed from the base
release — see ``ROADMAP`` for the localization variant that does have them.
Per-view probabilities are pooled to a study by :func:`pool_view_probabilities`;
the max is used because a study is abnormal if *any* projection is.

Nothing here is validated: no MURA study has been trained on yet.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset

from vindr.data import read_image
from vindr.model import build_model

NOT_CLAIM = "не диагноз: критерий для проверки врачом"

#: the folder suffixes MURA uses to carry the label
ABNORMAL_SUFFIXES = {"positive": 1, "negative": 0}

#: manifest columns a MURA CSV has to have for this module to work
REQUIRED_COLUMNS = ("mura_id", "study_id", "patient_id", "body_part", "laterality")

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".dcm")


def abnormal_from_path(path: str | Path) -> int | None:
    """The abnormality label for a MURA path, or None if it does not say.

    MURA's release carries the label in the directory name only, so the whole
    label pipeline starts here. The study folder is the last one that matches,
    which keeps a patient folder named e.g. `patient00001` from deciding
    anything.
    """
    parts = [p.lower() for p in Path(path).parts]
    for part in reversed(parts):
        for suffix, value in ABNORMAL_SUFFIXES.items():
            if part.endswith(f"_{suffix}") or part == suffix:
                return value
    return None


def parse_mura_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read one MURA manifest and add the label it does not contain.

    The path in `mura_id` is used for three things at once: the label, the
    patient and the study folder, so a study is identified by its full
    `study_uid` rather than by `study_id` alone — `study_id` is only unique
    within a body part, and MURA reuses `S1` across patients.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: manifest is missing {missing}")
    df = df.copy()
    df["abnormal"] = df["mura_id"].map(abnormal_from_path)
    if df["abnormal"].isna().any():
        n = int(df["abnormal"].isna().sum())
        bad = df.loc[df["abnormal"].isna(), "mura_id"].head(3).tolist()
        raise ValueError(
            f"{csv_path}: {n} row(s) carry no positive/negative in the path, e.g. {bad}. "
            "Refusing to guess — an unlabelled study counted as normal would lower the AUROC."
        )
    df["abnormal"] = df["abnormal"].astype(int)
    df["study_uid"] = df["mura_id"].str.replace(r"^MURA-v[\d.]+/", "", regex=True)
    df["split_source"] = df["mura_id"].str.extract(r"^MURA-v[\d.]+/(train|test)/", expand=False)
    df["study_folder"] = df["study_uid"].str.rsplit("/", n=1).str[0]
    df["image_name"] = df["mura_id"].str.rsplit("/", n=1).str[-1]
    return df


def image_path(root: str | Path, row: pd.Series) -> Path:
    """Full path of one manifest row's image under `root`."""
    return Path(root) / row["study_uid"] / row["image_name"]


def collect_studies(
    df: pd.DataFrame,
    root: str | Path,
    require_images: bool = True,
) -> pd.DataFrame:
    """Collapse image rows into one row per study.

    A study is the unit of prediction: MURA scores abnormal/abnormal per study,
    and the label belongs to the study, not to an individual projection.
    `n_views` comes from the manifest and is re-counted from disk when images
    are required, since the two disagree whenever a release is incomplete — and
    a training run that silently loses projections looks like a clean run.
    """
    studies = (
        df.groupby(["study_folder", "patient_id", "body_part", "laterality"], as_index=False)
        .agg(abnormal=("abnormal", "first"), n_views=("image_name", "size"))
    )
    if require_images:
        root = Path(root)
        counted = []
        for folder in studies["study_folder"]:
            d = root / folder
            counted.append(len([p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]) if d.is_dir() else 0)
        studies["n_images"] = counted
    else:
        studies["n_images"] = studies["n_views"]
    return studies.reset_index(drop=True)


def make_patient_splits(
    studies: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign train/val per patient, deterministically.

    Hashing the patient id rather than shuffling rows is what keeps this whole
    and patient-disjoint at the same time: the same id always lands on the same
    side, so a study of that patient cannot land on the other one.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    out = studies.copy()
    split = []
    for patient in out["patient_id"]:
        h = int(hashlib.sha256(f"{patient}{seed}".encode()).hexdigest(), 16)
        split.append("val" if h % 10000 < val_fraction * 10000 else "train")
    out["split"] = split
    return out


def patients_in_both_splits(studies: pd.DataFrame) -> list[str]:
    """Patients appearing in both train and val — should always be empty."""
    if "split" not in studies.columns:
        raise ValueError("call make_patient_splits first: there is no split column")
    by_split = studies.groupby("patient_id")["split"].nunique()
    return sorted(by_split[by_split > 1].index.tolist())


def split_summary(studies: pd.DataFrame) -> dict:
    """Counts a reviewer wants to see before trusting an AUROC."""
    out: dict = {"n_studies": len(studies), "n_patients": int(studies["patient_id"].nunique())}
    for split in ("train", "val"):
        part = studies[studies["split"] == split]
        out[f"{split}_studies"] = len(part)
        out[f"{split}_patients"] = int(part["patient_id"].nunique())
        out[f"{split}_abnormal"] = int(part["abnormal"].sum())
    out["patients_in_both_splits"] = patients_in_both_splits(studies)
    out["positive_rate"] = round(float(studies["abnormal"].mean()), 3)
    return out


class MuraStudyDataset(Dataset):
    """One item per study, holding up to `max_views` projections.

    Views are stacked into (V, C, H, W) with a companion mask, so a variable
    number of projections per study stays one batch rectangular. Studies vary
    between two and four views, so the tensor is padded to `max_views` and the
    mask says which slots are real — pooling a padded slot as a low
    probability would drag every study down, which is why the mask travels with
    the tensor.

    Studies are taken in manifest order, which is view order within the study;
    no shuffling happens here because the caller's sampler does that.
    """

    def __init__(
        self,
        studies: pd.DataFrame,
        root: str | Path,
        transforms=None,
        max_views: int = 4,
    ):
        if max_views < 1:
            raise ValueError(f"max_views must be >= 1, got {max_views}")
        self.studies = studies.reset_index(drop=True)
        self.root = Path(root)
        self.transforms = transforms
        self.max_views = max_views

    def __len__(self) -> int:
        return len(self.studies)

    def view_paths(self, study_folder: str) -> list[Path]:
        d = self.root / study_folder
        if not d.is_dir():
            raise FileNotFoundError(f"study folder {d} not found")
        paths = sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise FileNotFoundError(f"no images in {d}")
        return paths

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.studies.iloc[idx]
        paths = self.view_paths(row["study_folder"])[: self.max_views]
        images = []
        for p in paths:
            arr = read_image(p)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if self.transforms is not None:
                arr = self.transforms(image=arr)["image"]
            else:
                arr = torch.from_numpy(np.ascontiguousarray(arr)).float() / 255.0
                arr = arr.permute(2, 0, 1)
            images.append(arr)
        views = torch.zeros((self.max_views, *images[0].shape))
        mask = torch.zeros(self.max_views)
        for i, arr in enumerate(images):
            views[i] = arr
            mask[i] = 1.0
        label = torch.tensor(float(row["abnormal"]))
        return views, label, mask


def collate_studies(batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Collate MuraStudyDataset items, padding views to the widest batch.

    Padding across the batch as well as within an item: the last study of a
    batch may carry fewer views than the rest, and `torch.stack` on ragged V
    would fail. Real slots are identified by the mask.
    """
    views, labels, masks = zip(*batch, strict=True)
    width = max(v.shape[0] for v in views)
    shape = views[0].shape[1:]
    padded = torch.zeros((len(views), width, *shape))
    mask = torch.zeros((len(views), width))
    for i, (v, m) in enumerate(zip(views, masks, strict=True)):
        padded[i, : v.shape[0]] = v
        mask[i, : m.shape[0]] = m
    return padded, torch.stack(labels), mask


def pool_view_probabilities(
    probabilities: torch.Tensor,
    mask: torch.Tensor,
    how: str = "max",
) -> torch.Tensor:
    """Reduce per-view probabilities to one per study.

    A study is abnormal if any projection is, so the default is the max. `mean`
    is there because a model trained on padded batches sometimes needs it, but
    it is the weaker rule: a three-view study with one fracture reads as 0.33.
    Padded slots are masked out first — pooling them in would lower every
    study, since padding is exactly zero.
    """
    if how not in ("max", "mean"):
        raise ValueError(f"how must be 'max' or 'mean', got {how!r}")
    p = probabilities * mask
    if how == "max":
        return p.max(dim=1).values
    denom = mask.sum(dim=1).clamp(min=1.0)
    return p.sum(dim=1) / denom


def study_probabilities(
    probabilities: np.ndarray,
    study_uids: list[str],
    study_folders: list[str],
    how: str = "max",
) -> dict[str, float]:
    """Pool image-level scores to one score per study folder.

    Used when a model scores images one at a time, where the pooling order
    matches the dataset: a study is abnormal if its most abnormal projection is.
    """
    if not (len(probabilities) == len(study_uids) == len(study_folders)):
        raise ValueError("probabilities, study_uids and study_folders must be the same length")
    out: dict[str, float] = {}
    for p, _uid, folder in zip(probabilities, study_uids, study_folders, strict=True):
        val = float(p)
        out[folder] = max(out.get(folder, 0.0), val) if how == "max" else out.get(folder, 0.0) + val
    if how == "mean":
        counts: dict[str, int] = {}
        for _p, _uid, folder in zip(probabilities, study_uids, study_folders, strict=True):
            counts[folder] = counts.get(folder, 0) + 1
        out = {k: v / counts[k] for k, v in out.items()}
    return out


class MuraStudyClassifier(nn.Module):
    """A Phase 1 view encoder, applied per projection and pooled to one study.

    MURA is scored per study, so the label belongs to the set of views rather
    than to any single one. Running one backbone over all views and taking the
    max keeps the pretrained Phase 1 weights intact and matches the scoring
    rule — a study is abnormal if any projection is.

    The mask is not optional. A padded view is a black image, and after
    `Normalize(mean=0.5, std=0.5)` black is not a neutral input: it lands at -1
    and the backbone happily returns a real logit for it. Without masking, a
    two-view study would be scored partly on a blank rectangle.
    """

    def __init__(
        self,
        backbone: str = "tf_efficientnet_b0",
        pretrained: bool = True,
        drop_rate: float = 0.2,
        pooling: str = "max",
    ):
        super().__init__()
        if pooling not in ("max", "mean"):
            raise ValueError(f"pooling must be 'max' or 'mean', got {pooling!r}")
        self.view_model = build_model(backbone, pretrained=pretrained, num_classes=1,
                                      drop_rate=drop_rate)
        self.pooling = pooling

    def view_logits(self, views: torch.Tensor) -> torch.Tensor:
        """Per-projection logits, shape (B, V)."""
        b, v = views.shape[:2]
        flat = views.reshape(b * v, *views.shape[2:])
        return self.view_model(flat).reshape(b, v)

    def forward(self, views: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Study logits, shape (B,) — one score per study."""
        logits = self.view_logits(views)
        return pool_view_logits(logits, mask, self.pooling)

    def per_view_probabilities(self, views: torch.Tensor) -> torch.Tensor:
        """(B, V) per-projection probabilities, for explaining which view fired."""
        return self.view_logits(views).sigmoid()


def pool_view_logits(
    logits: torch.Tensor,
    mask: torch.Tensor | None,
    how: str = "max",
) -> torch.Tensor:
    """Reduce per-view logits to one study logit, ignoring padded slots.

    The two rules need different masking. `max` pushes padded slots far down
    with a large negative offset rather than `-inf`, so a wrong mask gives a
    finite score instead of a NaN that would poison the backward pass. `mean`
    instead zeroes them and divides by the count of real views — the offset
    would survive into the sum and make the average meaningless. An
    all-masked study therefore returns 0 under `mean` and a very negative logit
    under `max`; `view_paths` guarantees at least one real view per study, so
    either case means a caller built the tensors wrong.

    `mean` here averages logits, which is the geometric mean of the
    probabilities; `pool_view_probabilities` averages probabilities instead.
    The two agree for `max` (sigmoid is monotonic) but not for `mean`, so a
    mean-pooled study score read off the model will not match one read off
    per-view predictions.
    """
    if how not in ("max", "mean"):
        raise ValueError(f"how must be 'max' or 'mean', got {how!r}")
    if mask is None:
        return logits.max(dim=1).values if how == "max" else logits.mean(dim=1)
    mask = mask.to(logits.dtype)
    if how == "max":
        return (logits * mask + (mask - 1.0) * 1e4).max(dim=1).values
    return (logits * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def build_mura_model(
    backbone: str = "tf_efficientnet_b0",
    pretrained: bool = True,
    drop_rate: float = 0.2,
    pooling: str = "max",
) -> MuraStudyClassifier:
    """Study-level MURA classifier; mirrors `vindr.model.build_model`."""
    return MuraStudyClassifier(backbone, pretrained=pretrained, drop_rate=drop_rate,
                               pooling=pooling)