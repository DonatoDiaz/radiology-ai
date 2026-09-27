"""Inference for head CT studies (Phase 3): predictions + HU measurements.

    python -m vindr.ct.predict --input study_dir_or_nifti --ckpt best.pt
    python -m vindr.ct.predict --input study --ckpt seg_best.pt --seg-ckpt seg_best.pt --measure
    python -m vindr.ct.predict --input study --ckpt best.pt --measure   # adds fracture screening
    python -m vindr.ct.predict --input study --abdominal               # adds the Phase 4 organ block
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vindr.ct.dataset import HeadCTSliceDataset, HeadCTStudyDataset, collate_slices
from vindr.ct.fracture import associate_with_hematoma, skull_fractures
from vindr.ct.lesions import summarize_lesions
from vindr.ct.measure import summarize_study
from vindr.ct.model import HEMORRHAGE_TYPES, NUM_OUTPUTS, build_head_ct_model, slice_to_study_scores
from vindr.ct.organs import organ_report
from vindr.ct.pseudo import _resize_mask_to
from vindr.ct.segmentation import build_seg_model
from vindr.ct.volume import (
    CTSeries,
    apply_window,
    load_dicom_series,
    load_nifti,
    load_png_series,
    resample_to,
    slice_with_neighbours,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("vindr.ct.predict")


def load_series(path: str | Path) -> CTSeries:
    """Load one study from a NIfTI/``.npy`` file or a folder of DICOM/PNG slices."""
    path = Path(path)
    if path.is_file():
        if path.suffix == ".npy":  # cached HU volume, (Z, Y, X)
            return CTSeries(volume=np.load(path).astype(np.float32), study_id=path.parent.name)
        if path.suffix in (".nii", ".gz"):
            return load_nifti(path)
        raise ValueError(f"unsupported file: {path}")
    if list(path.glob("*.nii*")):
        return load_nifti(next(path.glob("*.nii*")))
    npy = list(path.glob("*.npy"))
    if npy:
        return CTSeries(volume=np.load(npy[0]).astype(np.float32), study_id=path.name)
    pngs = sorted(path.glob("*.png"))
    if pngs:
        return load_png_series([(p, 1.0, 0.0) for p in pngs], study_id=path.name)
    return load_dicom_series(path)


def load_model(ckpt: str | Path, device: torch.device):
    """Rebuild a model from a checkpoint written by ``vindr.ct.train``."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    # the fallback must include the `any` column, otherwise a checkpoint saved
    # without labels rebuilds a 5-output model and the state dict will not load
    default_labels = [*HEMORRHAGE_TYPES, "any"]
    model = build_head_ct_model(
        kind=blob.get("model", "study"),
        in_channels=int(blob.get("in_channels", 3)),
        num_outputs=int(len(blob.get("labels", default_labels)) or NUM_OUTPUTS),
        width=int(blob.get("width", 32)),
    )
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, blob


def load_seg_model(ckpt: str | Path, device: torch.device):
    """Rebuild the segmentation U-Net from a checkpoint written by ``vindr.ct.train_seg``."""
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = build_seg_model(
        in_channels=int(blob.get("in_channels", 3)),
        num_classes=int(blob.get("num_classes", 1)),
        width=int(blob.get("width", 32)),
        depth=int(blob.get("depth", 4)),
    )
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, blob


@torch.no_grad()
def predict_mask(
    seg_model,
    series: CTSeries,
    seg_blob: dict,
    device: torch.device,
    threshold: float | None = None,
) -> np.ndarray:
    """Run the segmentation net over a whole study; returns a (Z, Y, X[, C]) mask.

    ``series`` must already sit on the network grid (``resample_to`` with the
    checkpoint's ``target_shape``), so the returned mask shares its geometry.
    """
    thr = float(seg_blob.get("threshold", 0.5) if threshold is None else threshold)
    context = int(seg_blob.get("context", 1))
    n_classes = int(seg_blob.get("num_classes", 1))

    masks = np.zeros((series.n_slices, *series.shape[-2:], n_classes), dtype=np.float32)
    for start in range(0, series.n_slices, 8):
        chunk, zs = [], []
        for z in range(start, min(start + 8, series.n_slices)):
            planes = np.ascontiguousarray(slice_with_neighbours(series.volume, z, context=context))
            windowed = np.stack(
                [apply_window(planes[..., i], "brain") for i in range(planes.shape[-1])]
            )
            chunk.append(torch.from_numpy(windowed.astype(np.float32) / 255.0))
            zs.append(z)
        x = torch.stack(chunk).to(device)
        probs = seg_model(x).sigmoid().cpu().numpy()
        for k, z in enumerate(zs):
            masks[z] = (probs[k] > thr).astype(np.float32).transpose(1, 2, 0)
    return masks[..., 0] if n_classes == 1 else masks


@torch.no_grad()
def predict_series(
    model,
    series: CTSeries,
    blob: dict,
    device: torch.device,
    kind: str | None = None,
    threshold: float = 0.5,
) -> dict:
    """Classify one study; slice mode reports the top slices per label."""
    kind = kind or blob.get("model", "study")
    shape = tuple(blob.get("target_shape", (32, 224, 224)))
    label_names = list(blob.get("labels", [*HEMORRHAGE_TYPES, "any"]))

    if kind == "study":
        ds = HeadCTStudyDataset([series], None, int(shape[0]) * 2, target_shape=shape)
        x, _, _ = ds[0]
        logits = model(x.unsqueeze(0).to(device))
        probs = logits.sigmoid().cpu().numpy()[0]
        per_slice = None
    else:
        ds = HeadCTSliceDataset([series], None, target_shape=shape)
        dl = DataLoader(ds, batch_size=8, collate_fn=collate_slices)
        all_p = []
        for x, _, _ in dl:
            all_p.append(model(x.to(device)).sigmoid().cpu().numpy())
        per_slice = np.concatenate(all_p) if all_p else np.zeros((0, NUM_OUTPUTS))
        probs = slice_to_study_scores(per_slice)

    out = {
        "study_id": series.study_id or "",
        "n_slices": series.n_slices,
        "shape_zyx": list(series.volume.shape),
        "spacing_zyx_mm": [round(float(s), 4) for s in series.spacing],
        "kind": kind,
        "labels": label_names,
        "probabilities": {n: round(float(p), 4) for n, p in zip(label_names, probs)},
        "positive": {n: bool(p >= threshold) for n, p in zip(label_names, probs)},
    }
    if per_slice is not None and per_slice.size:
        order = np.argsort(-per_slice[:, -1])[:5]
        out["top_slices"] = [
            {
                "z": int(i),
                "any": round(float(per_slice[i, -1]), 4),
                "types": {
                    n: round(float(per_slice[i, j]), 4)
                    for j, n in enumerate(label_names[:-1])
                    if per_slice[i, j] >= threshold
                },
            }
            for i in order
            if per_slice[i, -1] >= threshold
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="study folder or NIfTI file")
    ap.add_argument("--ckpt", required=None, help="classification checkpoint from vindr-ct-train")
    ap.add_argument("--seg-ckpt", default=None, help="segmentation checkpoint from vindr-ct-seg-train")
    ap.add_argument("--kind", choices=("study", "slice"), default=None, help="override checkpoint kind")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seg-threshold", type=float, default=None, help="override the checkpoint mask threshold")
    ap.add_argument("--save-mask", type=Path, default=None, help="write the predicted mask as .npy")
    ap.add_argument("--measure", action="store_true", help="add HU measurements (density, volume, shift)")
    ap.add_argument(
        "--abdominal",
        action="store_true",
        help="add the Phase 4 organ block: aortic caliber and intimal flap, needs contrast",
    )
    ap.add_argument("--out", type=Path, default=None, help="write JSON here (default: stdout)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    if not args.ckpt and not args.seg_ckpt:
        ap.error("pass --ckpt (subtypes), --seg-ckpt (localization), or both")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    series = load_series(args.input)
    result: dict = {
        "study_id": series.study_id or "",
        "n_slices": series.n_slices,
        "shape_zyx": list(series.volume.shape),
        "spacing_zyx_mm": [round(float(s), 4) for s in series.spacing],
    }

    if args.ckpt:
        model, blob = load_model(args.ckpt, device)
        result.update(predict_series(model, series, blob, device, args.kind, args.threshold))

    if args.seg_ckpt:
        seg_model, seg_blob = load_seg_model(args.seg_ckpt, device)
        seg_shape = tuple(seg_blob.get("target_shape", (32, 224, 224)))
        grid = resample_to(series, seg_shape)  # the mask lives on the net's grid
        mask = predict_mask(seg_model, grid, seg_blob, device, args.seg_threshold)
        result["segmentation"] = {
            "grid_zyx": list(mask.shape),
            "grid_spacing_mm": [round(float(s), 4) for s in grid.spacing],
            **summarize_lesions(mask, grid.spacing),
        }
        log.info(
            "segmentation: %d lesion(s), %.2f mL total",
            result["segmentation"]["n_lesions"], result["segmentation"]["total_volume_ml"],
        )
        if args.save_mask:
            np.save(args.save_mask, mask.astype(np.uint8))
            log.info("wrote mask %s", args.save_mask)

    if args.abdominal:
        # Phase 4: the aorta is located from HU alone, the rest needs a mask
        result["organs"] = organ_report(series.volume, series.spacing)
        if result["organs"].get("aorta", {}).get("flags"):
            log.info("organs: %s", "; ".join(result["organs"]["aorta"]["flags"]))

    if args.measure or args.seg_ckpt:
        # midline shift / Evans on the native grid, volumes on the mask's grid
        meas = summarize_study(series)
        meas["skull_fractures"] = skull_fractures(series)
        if meas["skull_fractures"]["flag"]:
            log.info(
                "skull fracture suspicion: %d candidate line(s), %d debris fragment(s)",
                meas["skull_fractures"]["n_candidate_lines"],
                meas["skull_fractures"]["n_debris"],
            )
        if args.seg_ckpt and mask.any():
            grid_meas = summarize_study(grid, mask)
            for key in ("lesion_density", "lesion_volume"):
                if key in grid_meas:
                    meas[key] = grid_meas[key]
            pairing = associate_with_hematoma(series, _resize_mask_to(mask, series.shape))
            if pairing["adjacent"]:
                meas["hematoma_bone_contact"] = pairing
        result["measurements"] = meas
        result["disclaimer"] = (
            "Heuristic HU measurements and morphology hints for research use only — "
            "not a medical device, not clinically validated."
        )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text)
        log.info("wrote %s", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
