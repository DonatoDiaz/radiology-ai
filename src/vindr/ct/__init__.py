"""Phase 3 — head CT (brain) pipeline.

Modules:
    volume.py  — DICOM / 16-bit PNG / NIfTI loading, HU rescaling, windows, 2.5D slices
    model.py   — 2.5D slice CNN and attention-pooled study model, RSNA hemorrhage labels
    dataset.py — slice- and study-level torch datasets
    measure.py — HU-based measurements: density, volume, midline shift, Evans index
    train.py   — training loop (final pass, see ROADMAP §6)
"""

from vindr.ct.dataset import HeadCTSliceDataset, HeadCTStudyDataset
from vindr.ct.measure import (
    bbox_mask,
    classify_density,
    evans_index,
    fracture_suspicion,
    lesion_density,
    lesion_volume_ml,
    midline_shift_mm,
    summarize_study,
)
from vindr.ct.model import (
    HEMORRHAGE_TYPES,
    NUM_HEMORRHAGE,
    NUM_OUTPUTS,
    SIGNS,
    HeadCTSliceNet,
    HeadCTStudyNet,
    any_label,
    build_head_ct_model,
    loss_fn,
    slice_to_study_scores,
)
from vindr.ct.volume import (
    HU_ACUTE_BLOOD,
    HU_BLOOD,
    HU_EDEMA,
    WINDOWS,
    CTSeries,
    apply_window,
    load_dicom_series,
    load_nifti,
    load_png_series,
    normalize_uint16_png,
    read_rescale_table,
    resample_to,
    slice_with_neighbours,
)

__all__ = [  # noqa: RUF022  # grouped by module, not alphabetical, for readability
    # volume
    "CTSeries", "WINDOWS", "HU_BLOOD", "HU_ACUTE_BLOOD", "HU_EDEMA",
    "apply_window", "load_dicom_series", "load_nifti", "load_png_series",
    "normalize_uint16_png", "read_rescale_table", "resample_to", "slice_with_neighbours",
    # model
    "HEMORRHAGE_TYPES", "NUM_HEMORRHAGE", "NUM_OUTPUTS", "SIGNS",
    "HeadCTSliceNet", "HeadCTStudyNet", "any_label", "build_head_ct_model",
    "loss_fn", "slice_to_study_scores",
    # dataset
    "HeadCTSliceDataset", "HeadCTStudyDataset",
    # measure
    "bbox_mask", "classify_density", "lesion_density", "lesion_volume_ml", "midline_shift_mm",
    "evans_index", "fracture_suspicion", "summarize_study",
]
