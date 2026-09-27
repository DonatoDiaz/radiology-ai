"""Hemorrhage segmentation for head CT (Phase 3, localization task).

A compact 2.5D U-Net: axial slices are fed together with their neighbours
(2*context+1 channels) and produce a per-slice hemorrhage mask. Subtype
labelling is left to the classifier in :mod:`vindr.ct.model`; the mask gives
localization, which the study-level classifier cannot.

ROADMAP target: localization Dice >= 0.7.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DoubleConv(nn.Module):
    """Two conv-BN-ReLU units, no pooling — used in both encoder and decoder."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class HeadCTSegNet(nn.Module):
    """2.5D U-Net.

    Parameters
    ----------
    in_channels:
        ``2*context+1`` for 2.5D input (3 for one neighbour per side).
    num_classes:
        1 for a binary hemorrhage mask, more for multi-label subtype masks.
    width:
        Base channel count; ``width=32`` keeps a GTX 1650 (4 GB) usable.
    depth:
        Number of down/up-sampling pairs. Input size is padded to a multiple
        of ``2**(depth+1)`` and cropped back in :meth:`forward`.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, width: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth
        self.num_classes = num_classes
        chans = [width * (2**i) for i in range(depth + 1)]

        # encoder i outputs at resolution 1/2**i (encoders[0] keeps full size)
        self.encoders = nn.ModuleList()
        prev = in_channels
        for c in chans:
            self.encoders.append(DoubleConv(prev, c))
            prev = c
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(prev, prev * 2)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = prev * 2
        for c in reversed(chans):
            self.ups.append(nn.ConvTranspose2d(prev, c, 2, stride=2))
            self.decoders.append(DoubleConv(c * 2, c))
            prev = c

        self.head = nn.Conv2d(chans[0], num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        factor = 2 ** (self.depth + 1)
        h, w = x.shape[-2:]
        pad_h = (-h) % factor
        pad_w = (-w) % factor
        if pad_h or pad_w:  # keep the skip-connection shapes aligned
            x = F.pad(x, (0, pad_w, 0, pad_h))

        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips), strict=True):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = dec(torch.cat([skip, x], dim=1))

        logits = self.head(x)
        if pad_h or pad_w:
            logits = logits[..., :h, :w]
        return logits


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice loss, averaged over the batch and channels."""
    probs = logits.sigmoid()
    dims = tuple(range(2, probs.dim()))
    inter = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def seg_loss(logits: torch.Tensor, targets: torch.Tensor, bce_weight: float = 1.0,
             dice_weight: float = 1.0) -> torch.Tensor:
    """BCE (per-pixel) + soft Dice, the standard pair for sparse masks."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce_weight * bce + dice_weight * dice_loss(logits, targets)


def dice_score(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5,
               eps: float = 1.0) -> float:
    """Hard Dice on thresholded predictions — the ROADMAP localization metric."""
    pred = (probs > threshold).float()
    dims = tuple(range(2, probs.dim()))
    inter = (pred * targets).sum(dim=dims)
    denom = pred.sum(dim=dims) + targets.sum(dim=dims)
    return float(((2.0 * inter + eps) / (denom + eps)).mean())


def iou_score(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5,
              eps: float = 1.0) -> float:
    pred = (probs > threshold).float()
    dims = tuple(range(2, probs.dim()))
    inter = (pred * targets).sum(dim=dims)
    union = ((pred + targets) > 0).float().sum(dim=dims)
    return float(((inter + eps) / (union + eps)).mean())


def build_seg_model(**kwargs) -> HeadCTSegNet:
    return HeadCTSegNet(**kwargs)
