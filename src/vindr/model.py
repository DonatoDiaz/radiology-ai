"""Model builder using timm backbones with a multi-label head."""

from __future__ import annotations

import timm
import torch
from torch import nn


def build_model(
    backbone: str = "tf_efficientnet_b0",
    pretrained: bool = True,
    num_classes: int = 28,
    drop_rate: float = 0.2,
) -> nn.Module:
    """Create multi-label classifier from a timm backbone."""
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_mb(model: nn.Module) -> float:
    return count_parameters(model) / 1e6


if __name__ == "__main__":
    m = build_model()
    print(f"{type(m).__module__}.{type(m).__name__}  params: {count_parameters_mb(m):.2f}M")
    x = torch.zeros(1, 3, 512, 512)
    with torch.no_grad():
        y = m(x)
    print(f"output shape: {tuple(y.shape)}")