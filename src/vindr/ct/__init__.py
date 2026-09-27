"""Phase 3 — head CT (brain) pipeline.

Modules:
    volume.py  — DICOM / 16-bit PNG / NIfTI loading, HU rescaling, windows, 2.5D slices
    model.py   — 2.5D slice CNN and attention-pooled study model, RSNA hemorrhage labels
    dataset.py — slice-, study- and mask-level torch datasets
    measure.py — HU-based measurements: density, volume, midline shift, Evans index
    segmentation.py — 2.5D U-Net for hemorrhage localization (Dice + BCE)
    fracture.py — skull-vault fracture suspicion: a discontinuity in the bone ring
    organs.py   — Phase 4 organ anchors and measurements: aorta, adrenal, kidney,
                  mediastinal nodes and masses
    lesions.py  — 3D lesion extraction and morphology semiology from a mask
    pseudo.py  — weakly-supervised masks: CAM -> GrabCut -> HU band (RSNA has no pixel labels)
    train.py   — training loop (final pass, see ROADMAP §6)
    train_seg.py — segmentation training loop (localization Dice)
"""

from vindr.ct.dataset import (
    HeadCTSegDataset,
    HeadCTSliceDataset,
    HeadCTStudyDataset,
    collate_seg,
    load_masked_studies,
)
from vindr.ct.fracture import (
    analyze_slice,
    associate_with_hematoma,
    skull_fractures,
)
from vindr.ct.lesions import (
    find_components,
    find_lesions,
    lesion_semiotics,
    summarize_lesions,
)
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
from vindr.ct.organs import (
    adrenal_findings,
    aorta_findings,
    aorta_mask,
    kidney_findings,
    lymph_node_findings,
    mediastinal_mass_findings,
    organ_report,
)
from vindr.ct.pseudo import (
    cam_sharpness,
    cam_to_seed,
    mask_to_volume,
    pseudo_mask_study,
    restrict_to_blood_density,
    write_pseudo_dataset,
)
from vindr.ct.segmentation import (
    HeadCTSegNet,
    build_seg_model,
    dice_loss,
    dice_score,
    iou_score,
    seg_loss,
)
from vindr.ct.shapes import (
    equivalent_diameter_mm,
    extent_mm,
    feret_diameter_mm,
    hydraulic_thickness_mm,
    perimeter_mm,
    short_axis_mm,
    volume_ml,
)
from vindr.ct.sinuses import (
    cavity_measurements,
    find_cavities,
    fluid_level,
    mucosal_thickness,
    sinus_findings,
    sinus_report,
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
    "HeadCTSliceDataset", "HeadCTStudyDataset", "HeadCTSegDataset",
    "collate_seg", "load_masked_studies",
    # segmentation
    "HeadCTSegNet", "build_seg_model", "dice_loss", "dice_score", "iou_score", "seg_loss",
    # lesions
    "find_components", "find_lesions", "lesion_semiotics", "summarize_lesions",
    # fracture (skull vault)
    "analyze_slice", "associate_with_hematoma", "skull_fractures",
    # Phase 4 organs
    "adrenal_findings", "aorta_findings", "aorta_mask", "kidney_findings",
    "lymph_node_findings", "mediastinal_mass_findings", "organ_report",
    # Phase 5 sinuses
    "cavity_measurements", "find_cavities", "fluid_level", "mucosal_thickness",
    "sinus_findings", "sinus_report",
    # shared shape measurements
    "equivalent_diameter_mm", "extent_mm", "feret_diameter_mm",
    "hydraulic_thickness_mm", "perimeter_mm", "short_axis_mm", "volume_ml",
    # pseudo (weakly-supervised masks from slice-level labels)
    "cam_sharpness", "cam_to_seed", "mask_to_volume", "pseudo_mask_study",
    "restrict_to_blood_density", "write_pseudo_dataset",
    # measure
    "bbox_mask", "classify_density", "lesion_density", "lesion_volume_ml", "midline_shift_mm",
    "evans_index", "fracture_suspicion", "summarize_study",
]
