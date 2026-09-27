"""Hemorrhage labels and 2.5D head-CT models (Phase 3).

Labels follow the RSNA intracranial-hemorrhage taxonomy (the de-facto standard):
five hemorrhage compartments plus an "any" roll-up, evaluated per slice and
rolled up to the study level.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# RSNA IICH label order (matches the competition's label columns)
HEMORRHAGE_TYPES: tuple[str, ...] = (
    "epidural",           # 0 — эпидуральная
    "intraparenchymal",   # 1 — внутримозговая
    "intraventricular",   # 2 — внутрижелудочковая
    "subarachnoid",       # 3 — субарахноидальная
    "subdural",           # 4 — субдуральная
)
NUM_HEMORRHAGE = len(HEMORRHAGE_TYPES)
NUM_OUTPUTS = NUM_HEMORRHAGE + 1  # + "any" (multi-label friendly)

# Presumptive signs (по конспекту) that a model should also surface
SIGNS: tuple[str, ...] = (
    "midline_shift",      # смещение средостения/серединных структур
    "hydrocephalus",      # гидроцефалия (расширение желудочков)
    "edema",              # отёк вокруг очага
    "mass_effect",        # масс-эффект
    "skull_fracture",     # перелом кости (bone window)
    "hyperdense_vessel",  # гиперденсный сосуд (тромбоз/САК)
)


def any_label(targets: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Set the 'any hemorrhage' column from the five per-type columns.

    If the input already carries an ``any`` column it is recomputed in place;
    otherwise the column is appended.
    """
    arr = targets[..., :NUM_HEMORRHAGE]
    flag = arr.sum(axis=-1) > 0
    tail = targets[..., NUM_HEMORRHAGE + 1 :]  # extra columns beyond the 'any' roll-up
    if isinstance(targets, torch.Tensor):
        out = flag.to(dtype=targets.dtype, device=targets.device).unsqueeze(-1)
        return torch.cat([arr, out, tail], dim=-1)
    out = flag.astype(np.float32)[..., None]
    return np.concatenate([arr, out, np.asarray(tail)], axis=-1)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class HeadCTSliceNet(nn.Module):
    """2.5D slice classifier: (B, 2*context+1, H, W) -> per-slice hemorrhage logits.

    Kept deliberately small (EfficientNet-B0 scale) to train on 4 GB GPUs;
    swap the encoder for a timm backbone or a 2.5D ResNet when training on T4.
    """

    def __init__(self, in_channels: int = 3, num_outputs: int = NUM_OUTPUTS, width: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, width, pool=True),
            ConvBlock(width, width * 2, pool=True),
            ConvBlock(width * 2, width * 4, pool=True),
            ConvBlock(width * 4, width * 8, pool=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(width * 8, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class HeadCTStudyNet(nn.Module):
    """Study-level model: encodes slices (2.5D) and pools them with attention.

    Hemorrhage is a volume-level diagnosis — a slice classifier alone misses
    small bleeds. Attention pooling over slices gives a study-level answer and
    keeps per-slice logits available via ``return_slices``.
    """

    def __init__(self, in_channels: int = 3, num_outputs: int = NUM_OUTPUTS, width: int = 32):
        super().__init__()
        self.slice_model = HeadCTSliceNet(in_channels, num_outputs, width)
        feat_dim = width * 8
        self.attention = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, num_outputs),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.slice_model.encoder(x).mean(dim=(-2, -1))
        return h

    def forward(self, x: torch.Tensor, return_slices: bool = False):
        """x: (B, S, C, H, W) — B studies, S slices (2.5D channels in C)."""
        b, s = x.shape[:2]
        flat = x.reshape(b * s, *x.shape[2:])
        feat_maps = self.slice_model.encoder(flat)          # (B*S, C, h, w)
        feats = feat_maps.mean(dim=(-2, -1))                 # (B*S, C)
        # softmax over the slices *within* each study, not across the batch
        w = torch.softmax(self.attention(feats).reshape(b, s, -1), dim=1)  # (B, S, 1)
        pooled = (feats.reshape(b, s, -1) * w).sum(dim=1)
        maxed = feats.reshape(b, s, -1).max(dim=1).values
        logits = self.head(torch.cat([pooled, maxed], dim=1))
        if return_slices:
            slice_logits = self.slice_model.head(feat_maps).reshape(b, s, -1)
            return logits, slice_logits, w.reshape(b, s)
        return logits


def loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-type BCE plus a heavier weight on the 'any hemorrhage' column."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.ones_like(targets)
    weights[:, -1] = 2.0
    return (bce * weights).mean()


def slice_to_study_scores(slice_scores: np.ndarray) -> np.ndarray:
    """Max-pool slice probabilities into study probabilities (max over slices)."""
    return np.asarray(slice_scores).max(axis=0)


def build_head_ct_model(kind: str = "study", **kwargs) -> nn.Module:
    if kind == "slice":
        return HeadCTSliceNet(**kwargs)
    if kind == "study":
        return HeadCTStudyNet(**kwargs)
    raise ValueError(f"unknown model kind: {kind}")
