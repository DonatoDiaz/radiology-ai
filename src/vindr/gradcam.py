"""Grad-CAM attention maps for the last convolutional block."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Reusable Grad-CAM for a timm/EfficientNet classifier.

    Usage:
        cam = GradCAM(model)
        heatmap = cam(x, target_class=5)          # (H, W) float 0..1
        blended = cam.overlay(x, target_class=5)  # image with heatmap overlay
    """

    def __init__(self, model: nn.Module, target_layer: str | None = None):
        self.model = model.eval()
        self.layer = self._find_target_layer(target_layer)
        self._gradients: torch.Tensor | None = None
        self._activations: torch.Tensor | None = None
        self._handles = []

        def _fw_hook(m, inp, out):
            self._activations = out.detach()

        def _bw_hook(m, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._handles.append(self.layer.register_forward_hook(_fw_hook))
        self._handles.append(self.layer.register_full_backward_hook(_bw_hook))

    def _find_target_layer(self, name: str | None) -> nn.Module:
        if name is not None:
            mod = self.model
            for part in name.split("."):
                mod = getattr(mod, part)
            return mod
        # Heuristic: deepest Conv2d before any classifier.
        convs = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
        if not convs:
            raise ValueError("no Conv2d found in model")
        return convs[-1]

    def __call__(self, x: torch.Tensor, target_class: int | None = None) -> np.ndarray:
        """Compute a (H, W) heatmap in [0, 1]."""
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if target_class is None:
            target_class = int(logits.argmax(dim=1))
        score = logits[0, target_class]
        self.model.zero_grad()
        score.backward(retain_graph=False)

        acts = self._activations[0]  # (C, h, w)
        grads = self._gradients[0]   # (C, h, w)
        weights = torch.mean(grads, dim=(1, 2), keepdim=True)  # (C, 1, 1)
        heat = F.relu(torch.sum(weights * acts, dim=0))         # (h, w)
        heat = F.interpolate(heat[None, None], size=x.shape[2:], mode="bilinear")
        heat = heat[0, 0]
        hmin, hmax = heat.min(), heat.max()
        if hmax > hmin:
            heat = (heat - hmin) / (hmax - hmin)
        x.detach_()
        return heat.cpu().numpy()

    def overlay(self, x: torch.Tensor, target_class: int | None = None, alpha: float = 0.45) -> np.ndarray:
        """Return RGB numpy overlay: grayscale image + colored heatmap."""
        import matplotlib.cm as cm

        heat = self(x, target_class=target_class)
        img = x[0].detach().cpu().permute(1, 2, 0).numpy()
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        img = np.repeat(img[..., :1], 3, axis=-1)

        heat = self(x, target_class=target_class)
        colored = (cm.jet(heat)[..., :3] * 255).astype(np.uint8)
        blended = (alpha * colored + (1 - alpha) * img).astype(np.uint8)
        return blended

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()