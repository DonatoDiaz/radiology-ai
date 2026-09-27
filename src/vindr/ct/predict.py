"""Inference for head CT studies (Phase 3): predictions + HU measurements.

    python -m vindr.ct.predict --input study_dir_or_nifti --ckpt best.pt
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
from vindr.ct.measure import summarize_study
from vindr.ct.model import HEMORRHAGE_TYPES, NUM_OUTPUTS, build_head_ct_model, slice_to_study_scores
from vindr.ct.volume import CTSeries, load_dicom_series, load_nifti, load_png_series

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
    model = build_head_ct_model(
        kind=blob.get("model", "study"),
        in_channels=int(blob.get("in_channels", 3)),
        num_outputs=int(len(blob.get("labels", HEMORRHAGE_TYPES)) or NUM_OUTPUTS),
        width=int(blob.get("width", 32)),
    )
    model.load_state_dict(blob["model_state"])
    model.to(device).eval()
    return model, blob


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
    ap.add_argument("--ckpt", required=True, help="checkpoint from vindr-ct-train")
    ap.add_argument("--kind", choices=("study", "slice"), default=None, help="override checkpoint kind")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--measure", action="store_true", help="add HU measurements (density, volume, shift)")
    ap.add_argument("--out", type=Path, default=None, help="write JSON here (default: stdout)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    series = load_series(args.input)
    model, blob = load_model(args.ckpt, device)
    result = predict_series(model, series, blob, device, args.kind, args.threshold)

    if args.measure:
        result["measurements"] = summarize_study(series)
        result["disclaimer"] = (
            "Heuristic HU measurements for research use only — not a medical device, "
            "not clinically validated."
        )

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text)
        log.info("wrote %s", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
