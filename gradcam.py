"""Grad-CAM for the CIFAKE CNN.

Design notes
------------
* Target layer: the output of the last conv block's ReLU (``conv_layers[12]``,
  128 channels at 8x8). Using the post-activation features is the standard
  Grad-CAM choice; the previous code used the pre-BatchNorm conv output.
* No hooks: the forward pass is split manually and ``retain_grad()`` is used on
  the target activation. Nothing is registered on the (cached, shared) model, so
  nothing can accumulate across uploads and no deprecated hook API is used.
* Class-aware: the score being explained is the *pre-sigmoid logit* of the
  predicted class (+logit for REAL, -logit for FAKE). Using the logit avoids the
  vanishing gradient of a saturated sigmoid, and explains FAKE predictions
  correctly (the old code always explained the REAL score).

Grad-CAM shows which regions most influenced the model's score. It does NOT
prove that an image is real or AI-generated.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from model import CIFAKE_CNN

TARGET_LAYER_INDEX = 12          # ReLU after the third conv block
_EPS = 1e-8


@dataclass
class CamResult:
    cam: np.ndarray              # (8, 8) float32 in [0, 1]
    target_class: str            # "REAL" or "FAKE" (class being explained)
    blank: bool                  # True when no positive evidence was found


def compute_gradcam(model: CIFAKE_CNN, tensor: torch.Tensor,
                    target_class: str | None = None) -> CamResult:
    """Compute a Grad-CAM map for a (1, 3, 32, 32) tensor.

    ``target_class`` defaults to the model's predicted class.
    """
    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            x = tensor.detach().clone()
            conv = model.conv_layers
            acts = conv[: TARGET_LAYER_INDEX + 1](x)
            acts.retain_grad()
            feats = conv[TARGET_LAYER_INDEX + 1:](acts)
            logit = model.fc_layers[:-1](feats).squeeze()   # before the Sigmoid

            if target_class is None:
                target_class = "REAL" if logit.item() >= 0 else "FAKE"
            score = logit if target_class == "REAL" else -logit

            model.zero_grad(set_to_none=True)
            score.backward()

            weights = acts.grad.mean(dim=(2, 3), keepdim=True)        # (1, C, 1, 1)
            cam = torch.relu((weights * acts).sum(dim=1))[0]          # (8, 8)
            cam = cam.detach().cpu().numpy().astype(np.float32)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)

    peak = float(cam.max())
    if not np.isfinite(peak) or peak < _EPS:
        return CamResult(np.zeros_like(cam), target_class, blank=True)
    return CamResult(cam / peak, target_class, blank=False)


def render_heatmap(cam: np.ndarray, size: int = 256) -> np.ndarray:
    """Coloured (JET) RGB uint8 heatmap upsampled smoothly to ``size`` pixels."""
    resized = cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC)
    resized = np.clip(resized, 0.0, 1.0)
    coloured = cv2.applyColorMap(np.uint8(255 * resized), cv2.COLORMAP_JET)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


def render_overlay(image32: Image.Image, cam: np.ndarray, size: int = 256,
                   alpha: float = 0.45) -> np.ndarray:
    """Blend the heatmap over the 32x32 model input, shown at ``size`` pixels.

    A fully blank map returns the plain image so no misleading colour is added.
    """
    base = np.asarray(image32.convert("RGB").resize((size, size), Image.BICUBIC),
                      dtype=np.float32)
    if not np.any(cam > 0):
        return base.astype(np.uint8)
    heat = render_heatmap(cam, size).astype(np.float32)
    # Only tint where the map is active, so cold regions keep original colours.
    mask = np.clip(cv2.resize(cam, (size, size), interpolation=cv2.INTER_CUBIC),
                   0, 1)[..., None]
    blended = base * (1 - alpha * mask) + heat * (alpha * mask)
    return np.clip(blended, 0, 255).astype(np.uint8)
