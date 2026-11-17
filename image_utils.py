"""Image validation and preprocessing for the CIFAKE model.

Preprocessing is deliberately identical to training
(Resize((32, 32)) -> ToTensor -> Normalize(0.5, 0.5)).
"""
from __future__ import annotations

import io
from typing import BinaryIO

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torchvision import transforms

MODEL_SIZE = (32, 32)
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MAX_FILE_BYTES = 10 * 1024 * 1024      # 10 MB upload limit
MAX_PIXELS = 50_000_000                # reject decompression bombs (~50 MP)

# Same transform as notebooks/02_baseline_model.ipynb
TRANSFORM = transforms.Compose([
    transforms.Resize(MODEL_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class ImageValidationError(ValueError):
    """Raised with a user-friendly message when an upload cannot be used."""


def load_image(source: bytes | BinaryIO) -> Image.Image:
    """Validate and decode an uploaded image into an RGB PIL image."""
    data = source if isinstance(source, (bytes, bytearray)) else source.read()
    if not data:
        raise ImageValidationError("The uploaded file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise ImageValidationError(
            f"File is too large ({len(data) / 1e6:.1f} MB). "
            f"Maximum is {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        width, height = probe.size          # header only; no full decode yet
        if fmt not in ALLOWED_FORMATS:
            raise ImageValidationError(
                f"Unsupported image format ({fmt or 'unknown'}). "
                "Use JPG, JPEG, PNG or WEBP.")
        if width * height > MAX_PIXELS:
            raise ImageValidationError(
                f"Image is too large ({width}x{height} pixels).")
        probe.verify()                      # detects truncated/corrupt data
        image = Image.open(io.BytesIO(data))  # verify() invalidates `probe`
        image.load()
    except ImageValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError,
            Image.DecompressionBombError) as exc:
        raise ImageValidationError(
            "The file is not a valid or readable image (it may be corrupted).") from exc

    image = ImageOps.exif_transpose(image)   # respect camera orientation
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")         # grayscale, palette, CMYK -> RGB
    return image


def preprocess(image: Image.Image) -> torch.Tensor:
    """RGB PIL image -> normalised tensor of shape (1, 3, 32, 32)."""
    return TRANSFORM(image).unsqueeze(0)


def to_display_32(image: Image.Image) -> Image.Image:
    """The 32x32 image the model actually sees (before normalisation)."""
    return image.resize(MODEL_SIZE, Image.BILINEAR)
