"""Weakly-supervised masks from a slice classifier (Phase 3 localization).

RSNA labels whole slices, not pixels, so there is nothing to segment on. But
a trained slice classifier already localizes *somewhat*: its class activation
map (CAM) highlights the features that drove the decision, which for acute
blood is the bleed itself. That turns slice labels into coarse masks for free:

    slice labels -> CAM -> seed -> GrabCut on the HU image -> 3D mask

The masks are pseudo-labels: good enough to pretrain or fine-tune the U-Net
(:mod:`vindr.ct.segmentation`) and to bootstrap self-training, *not* good
enough to report as a result. Every mask here therefore carries the evidence
that produced it (probability, CAM peak, refined area) so training can filter
by confidence instead of trusting the geometry blindly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from vindr.ct.volume import CTSeries, apply_window, slice_with_neighbours

log = logging.getLogger(__name__)

# CAM is re-normalized per slice; this share of pixels forms the seed
DEFAULT_SEED_QUANTILE = 0.90
# share of the slice that is marked as *definite* foreground inside the seed
DEFAULT_CORE_FRACTION = 0.02
# a CAM this flat (top 2% over the mean) means the net is guessing, not that
# the bleed is small; CAMs are normalized, so only their concentration counts
DEFAULT_MIN_SHARPNESS = 1.5
# acute clotted blood is hyperdense against brain; chronic collections are not,
# so the band is exposed as a parameter rather than hard-coded
DEFAULT_BLOOD_HU = (50.0, 110.0)
# GrabCut is expensive; a few iterations are enough to snap to the hyperdense clot
DEFAULT_GRABCUT_ITERS = 3


def _cam_from_feats(feats: torch.Tensor, class_index: int, weights: torch.Tensor | None) -> np.ndarray:
    """CAM = sum_k w_k * f_k(x, y) over the encoder's channels, normalized to [0, 1].

    ``weights`` are the ``any`` column of the classifier's final linear layer.
    ``class_index`` is accepted so a caller can ask for a subtype instead, but
    the caller supplies the matching column; it is not used here.
    """
    del class_index  # the column choice happens in the weight vector
    f = feats.squeeze(0)
    cam = (f * f).sum(dim=0) if weights is None else (f * weights.reshape(-1, 1, 1)).sum(dim=0)
    cam = cam.detach().float()
    cam = cam - cam.min()
    peak = float(cam.max())
    return (cam / peak).cpu().numpy() if peak > 0 else cam.cpu().numpy()


def _encode_hires(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Encoder features with the *last* pooling removed (2x resolution).

    The classifier pools after four downsamplings, so a plain CAM is a 4x4 grid
    upsampled to the slice size — far too coarse to bound a bleed. CAM only
    assumes the feature map is summed with the classifier's channel weights, so
    the same weights apply before the final pool, and the map is 8x8. Coarser
    still on a 32-slice grid, but the localization is usable.
    """
    h = x
    blocks = list(net.encoder)
    for block in blocks[:-1]:
        h = block(h)
    last = blocks[-1]
    body = getattr(last, "body", None)
    if body is None or not any(isinstance(m, nn.MaxPool2d) for m in body):
        return last(h)  # nothing to skip; the plain encoder is the best we have
    convs = [m for m in body if not isinstance(m, nn.MaxPool2d)]
    for layer in convs:
        h = layer(h)
    return h


@torch.no_grad()
def cam_for_slices(
    model: nn.Module,
    series: CTSeries,
    context: int = 1,
    target_shape: tuple[int, int, int] | None = (32, 224, 224),
    window: str = "brain",
    class_index: int = -1,
    hi_res: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """CAMs and class probabilities for every slice of a study.

    Returns ``(cams, probs)`` on the (possibly resampled) image grid, where
    ``cams`` is ``(Z, Y, X)`` in [0, 1] and ``probs`` is ``(Z,)``.
    """
    from vindr.ct.dataset import _to_tensor
    from vindr.ct.volume import resample_to

    st = resample_to(series, target_shape) if target_shape else series
    net = model.slice_model if hasattr(model, "slice_model") else model
    net.eval()
    linear = net.head[-1]
    # Linear stores (out_features, in_features): row `class_index` holds the
    # per-channel weights that made the encoder light up for that class
    weights = linear.weight.detach()[class_index]  # (C,)

    cams = np.zeros((st.n_slices, *st.shape[-2:]), np.float32)
    probs = np.zeros(st.n_slices, np.float32)
    for z in range(st.n_slices):
        planes = np.ascontiguousarray(slice_with_neighbours(st.volume, z, context=context))
        x = _to_tensor(planes, window).unsqueeze(0)
        feats = _encode_hires(net, x) if hi_res else net.encoder(x)
        cam = _cam_from_feats(feats, class_index, weights)
        cams[z] = cv2.resize(cam, (st.shape[2], st.shape[1]), interpolation=cv2.INTER_CUBIC)
        logits = net(x)
        probs[z] = float(logits.sigmoid()[0, class_index])
    return np.clip(cams, 0.0, 1.0), probs


def cam_sharpness(cam: np.ndarray, top_fraction: float = 0.02) -> float:
    """How peaked a CAM is: mean of its top pixels over its overall mean.

    CAMs are min-max normalized, so their maximum is 1.0 by construction and
    cannot say anything about confidence. What distinguishes a real focus from
    a net guessing is *concentration*: a flat CAM has sharpness 1.0, a focused
    one is several times the mean.
    """
    flat = np.asarray(cam, np.float64).reshape(-1)
    if flat.size == 0:
        return 1.0
    k = max(1, round(flat.size * top_fraction))
    top = np.partition(flat, -k)[-k:]
    return float(top.mean() / (flat.mean() + 1e-8))


def cam_to_seed(cam: np.ndarray, quantile: float = DEFAULT_SEED_QUANTILE,
                min_sharpness: float = DEFAULT_MIN_SHARPNESS,
                core_fraction: float = DEFAULT_CORE_FRACTION) -> np.ndarray | None:
    """Turn a CAM into GrabCut seeds; ``None`` when the slice is too ambiguous.

    Three guards keep the pseudo-labels honest. The CAM must be *peaked*
    (:func:`cam_sharpness`), otherwise the classifier is guessing and there is
    nothing to refine. The seed is a fixed pixel count rather than a value
    threshold — a threshold degenerates to "the whole slice" on a flat CAM and
    would hand GrabCut a meaningless region. The top ``core_fraction`` of the
    seed becomes *definite* foreground so GrabCut grows from a core instead of
    swallowing the whole brain.
    """
    if cam_sharpness(cam) < min_sharpness:
        return None
    flat = np.asarray(cam, np.float32).reshape(-1)
    seed = np.zeros(flat.size, np.uint8)
    n_seed = round((1.0 - quantile) * flat.size)
    n_seed = max(1, min(n_seed, flat.size))
    seed[np.argpartition(flat, -n_seed)[-n_seed:]] = 1
    n_core = max(1, round(core_fraction * flat.size))
    n_core = min(n_core, n_seed)
    seed[np.argpartition(flat, -n_core)[-n_core:]] = 2
    return seed.reshape(np.asarray(cam).shape)


def restrict_to_blood_density(
    mask: np.ndarray,
    hu_slice: np.ndarray,
    blood_hu: tuple[float, float] = DEFAULT_BLOOD_HU,
    close_kernel: int = 5,
) -> np.ndarray:
    """Keep only voxels whose HU falls in the acute-blood band, inside ``mask``.

    A CAM says *where to look*, not *what is there*; the density check is what
    turns a region into a clot. Acute blood is hyperdense (roughly 50-80 HU)
    against 20-40 HU of brain, so the band separates them — and it is the step
    that rescues the localization when the CAM itself is sloppy. The band is a
    parameter because chronic subdural hygromas sit down at ~20-30 HU, where a
    fixed acute band would delete them.
    """
    low, high = blood_hu
    out = (mask & (hu_slice >= low) & (hu_slice <= high)).astype(np.uint8)
    if close_kernel > 1:  # close pinholes so a speckle does not become a lesion
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out.astype(bool)


def refine_with_grabcut(
    hu_slice: np.ndarray,
    seed: np.ndarray,
    window: str = "brain",
    iters: int = DEFAULT_GRABCUT_ITERS,
) -> np.ndarray:
    """Grow a CAM seed into a full region using the HU image as evidence.

    ``hu_slice`` is sharpened by the same window the classifier saw, so the
    colour model matches what the network actually responded to. Seeds marked 2
    by :func:`cam_to_seed` become ``GC_FGD``: without that hard core GrabCut
    treats the whole slice as one class and returns the entire brain, because
    acute blood and grey matter sit close together on a brain window.
    """
    img = apply_window(hu_slice, window)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = np.full(hu_slice.shape, cv2.GC_PR_BGD, np.uint8)
    mask[seed > 0] = cv2.GC_PR_FGD
    mask[seed >= 2] = cv2.GC_FGD
    if not (mask == cv2.GC_FGD).any() or not (mask == cv2.GC_BGD | (mask == cv2.GC_PR_BGD)).any():
        return seed > 0
    try:
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        cv2.grabCut(img, mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    except cv2.error:  # degenerate seed -> keep the seed as-is
        return seed > 0
    refined = (mask == cv2.GC_PR_FGD) | (mask == cv2.GC_FGD)
    # GrabCut sometimes returns something wildly larger than the evidence; a
    # pseudo-label that grew 5x is noise, not a lesion
    if refined.sum() > 5 * max(1, int((seed > 0).sum())):
        return seed > 0
    return refined


def mask_to_volume(
    slices: list[np.ndarray | None],
    min_slices: int = 2,
) -> np.ndarray | None:
    """Stack per-slice masks into a ``(Z, Y, X)`` volume, keeping only studied slices.

    Hemorrhage spans several slices; a component that appears on one slice only
    is usually CAM noise, and keeping it teaches the U-Net to hallucinate.
    ``None`` entries (low confidence) become empty planes, so the result stays
    aligned with the series it came from.
    """
    filled = [(z, m) for z, m in enumerate(slices) if m is not None]
    if not filled or sum(1 for _z, m in filled if m.any()) < min_slices:
        return None
    y, x = filled[0][1].shape[-2:]
    out = np.zeros((len(slices), y, x), bool)
    for z, m in filled:
        out[z] = m
    return out if out.any() else None


def _resize_plane(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample of a single label plane."""
    return cv2.resize(plane, shape, interpolation=cv2.INTER_NEAREST)


def _resize_mask_to(mask: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbour resample of a mask onto another ``(Z, Y, X)`` grid.

    Nearest-neighbour is the only interpolation that will not invent partial
    values in a label mask.
    """
    z, y, x = shape
    planes = [
        _resize_plane(mask[min(zi, mask.shape[0] - 1)], (x, y)) for zi in range(z)
    ]
    return np.stack(planes).astype(bool)


def _native_to_grid_z(n_grid: int, n_native: int) -> np.ndarray:
    """For each native slice, the grid slice whose sampling position is nearest.

    Mapping the other way (grid -> native, nearest) silently *skips* native
    slices when the grid is coarser: a 5 mm grid over 12 native slices visits
    only z in {0, 2, 4, 7, 9, 11}, so a bleed confined to z 5-8 gets examined on
    a single slice and is then dropped by ``min_slices``. Going native -> grid
    keeps every native slice on the record.
    """
    positions = np.linspace(0, n_native - 1, n_grid)
    native = np.arange(n_native)
    return np.abs(positions[:, None] - native[None, :]).argmin(axis=0).astype(int)


def pseudo_mask_study(
    model: nn.Module,
    series: CTSeries,
    context: int = 1,
    target_shape: tuple[int, int, int] | None = (32, 224, 224),
    window: str = "brain",
    class_index: int = -1,
    prob_threshold: float = 0.5,
    seed_quantile: float = DEFAULT_SEED_QUANTILE,
    min_sharpness: float = DEFAULT_MIN_SHARPNESS,
    core_fraction: float = DEFAULT_CORE_FRACTION,
    grabcut_iters: int = DEFAULT_GRABCUT_ITERS,
    blood_hu: tuple[float, float] = DEFAULT_BLOOD_HU,
    use_grabcut: bool = True,
    min_slices: int = 2,
) -> dict | None:
    """Full weakly-supervised pipeline for one study.

    Per slice: confident classifier -> CAM -> seed -> GrabCut -> density band.
    Returns a report with the ``(Z, Y, X)`` boolean mask on the *native* grid
    plus the evidence behind it, or ``None`` when no slice clears the bar.

    Two grids are involved on purpose. The CAM has to be computed on the
    network's grid, but GrabCut and the HU band have to run on the native
    voxels: resampling averages a 70 HU bleed against 30 HU brain and against
    air, and the density band would then reject the very haemorrhage it is
    meant to confirm. So the CAM seed is upsampled back to native and the mask
    is stored next to the original image it must align with.
    """
    from vindr.ct.volume import resample_to

    grid = resample_to(series, target_shape) if target_shape else series
    cams, probs = cam_for_slices(
        model, grid, context=context, target_shape=None,
        window=window, class_index=class_index,
    )
    hu = series.volume
    xy = (series.shape[-1], series.shape[-2])
    zmap = _native_to_grid_z(cams.shape[0], series.n_slices)
    slices: list[np.ndarray | None] = []
    for nz in range(series.n_slices):
        gz = int(zmap[nz])
        if probs[gz] < prob_threshold:
            slices.append(None)
            continue
        seed = cam_to_seed(cams[gz], seed_quantile, min_sharpness, core_fraction)
        if seed is None:
            slices.append(None)
            continue
        # the seed comes home before any HU reasoning happens
        native_seed = _resize_plane(seed, xy)
        plane_hu = hu[nz]
        region = native_seed > 0
        if use_grabcut:
            region = refine_with_grabcut(plane_hu, native_seed, window, grabcut_iters)
        slices.append(restrict_to_blood_density(region, plane_hu, blood_hu))

    mask = mask_to_volume(slices, min_slices=min_slices)
    if mask is None:
        return None
    if tuple(mask.shape) != tuple(series.shape):
        # only reachable when mask_to_volume ran on a non-native grid
        mask = _resize_mask_to(mask, series.shape)
    kept = [z for z, m in enumerate(slices) if m is not None and m.any()]
    return {
        "study_id": series.study_id or "",
        "mask": mask,
        "grid_zyx": list(cams.shape),
        "image_zyx": list(series.shape),
        "spacing_zyx": [round(float(s), 4) for s in series.spacing],
        "n_masked_slices": len(kept),
        "z_range": [min(kept), max(kept)],
        "max_prob": round(float(probs.max()), 4),
        # kept holds native indices, probs is indexed by grid slice
        "mean_prob_masked": round(float(np.mean([probs[int(zmap[z])] for z in kept])), 4),
        "n_dropped_slices": int(series.n_slices - len(kept)),
    }


def _link_images(src: Path, dst: Path) -> int:
    """Symlink a study's image files next to its mask (no 144 GB of copying).

    Anything whose name mentions a mask is skipped, but ``image.npy`` is not:
    a .npy cache is a legitimate image format here, only ``mask.npy`` is not.
    """
    if not src.is_dir():
        log.warning("no image folder at %s", src)
        return 0
    n = 0
    for f in sorted(src.iterdir()):
        if not f.is_file() or f.suffix == ".json" or "mask" in f.name.lower():
            continue
        target = dst / f.name
        if target.exists():
            continue
        try:
            target.symlink_to(f.resolve())
            n += 1
        except OSError as exc:  # pragma: no cover  # e.g. Windows without privileges
            log.warning("cannot link %s -> %s: %s", f, target, exc)
    return n


def write_pseudo_dataset(
    reports: list[dict],
    out_root: str | Path,
    image_root: str | Path | None = None,
) -> list[Path]:
    """Persist reports as ``<out_root>/<study_id>/mask.npy`` + a JSON evidence log.

    Writing the mask *next to* the images keeps the dataset readable by
    :func:`vindr.ct.dataset.load_masked_studies` with no copying, so a real
    RSNA tree can be annotated in place. When ``out_root`` differs from
    ``image_root`` the image files are symlinked in, because the loader needs
    both halves under one study folder. ``pseudo_labels.json`` records the
    evidence for every study (probability, z-range, dropped slices) so
    training can filter by confidence rather than trust the geometry blindly.
    """
    import json

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    meta: dict[str, dict] = {}
    for rep in reports:
        if rep is None:
            continue
        sid = rep["study_id"] or f"study_{len(written):06d}"
        d = out_root / sid
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "mask.npy", np.asarray(rep["mask"]).astype(np.uint8))
        if image_root is not None:
            _link_images(Path(image_root) / sid, d)
        written.append(d)
        meta[sid] = {k: v for k, v in rep.items() if k != "mask"}
    (out_root / "pseudo_labels.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    log.info("wrote %d pseudo-masked studies to %s", len(written), out_root)
    return written
