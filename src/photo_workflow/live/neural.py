"""
Neural cleanup of the final sRGB image.

darktable's profiled denoise removes most sensor noise but leaves colored speckle on skin,
hair and smooth surfaces. SCUNet (a real-world denoising network) removes it, but on its own
it smooths real detail. So the network supplies the chroma and half of the luma, the original
supplies the rest of the luma, and a light unsharp mask restores crispness.

The model weights are downloaded once to `~/.cache/photo-workflow/models`. Torch runs on the
Apple GPU (MPS) when available.
"""

from __future__ import annotations

import urllib.request
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from spandrel import ImageModelDescriptor, ModelLoader

MODEL_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth"
MODEL_PATH = Path.home() / ".cache" / "photo-workflow" / "models" / "scunet_color_real_psnr.pth"
TILE = 768
TILE_PAD = 64
LUMA_BLEND = 0.5
SHARPEN_AMOUNT = 0.7
SHARPEN_SIGMA = 1.3
MAX_8 = 255


def ensure_model(path: Path = MODEL_PATH, url: str = MODEL_URL) -> Path:
    """
    Return the model weights file, downloading it on first use.

    Returns:
        Path to the weights file.
    """
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as handle:
        while chunk := response.read(1 << 20):
            handle.write(chunk)
    partial.replace(path)
    return path


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@lru_cache(maxsize=1)
def _model() -> ImageModelDescriptor:
    return ModelLoader().load_from_file(ensure_model()).to(_device()).eval()


def _run_tiles(
    rgb: NDArray[np.uint8], report: Callable[[float], None] = lambda _fraction: None
) -> NDArray[np.uint8]:
    """
    Run the network over overlapping tiles, keeping only each tile's interior.

    Returns:
        The denoised RGB image.
    """
    model = _model()
    device = _device()
    height, width = rgb.shape[:2]
    output = np.empty_like(rgb)
    total = -(-height // TILE) * -(-width // TILE)
    done = 0
    for top in range(0, height, TILE):
        for left in range(0, width, TILE):
            bottom, right = min(top + TILE, height), min(left + TILE, width)
            y0, x0 = max(top - TILE_PAD, 0), max(left - TILE_PAD, 0)
            y1, x1 = min(bottom + TILE_PAD, height), min(right + TILE_PAD, width)
            tile = torch.from_numpy(rgb[y0:y1, x0:x1]).permute(2, 0, 1)[None].to(device)
            with torch.no_grad():
                cleaned = model(tile.float() / MAX_8)
            cleaned = (cleaned[0].permute(1, 2, 0).clamp(0, 1) * MAX_8).round().byte().cpu()
            output[top:bottom, left:right] = cleaned.numpy()[
                top - y0 : bottom - y0, left - x0 : right - x0
            ]
            done += 1
            report(done / total)
    return output


def blend_detail(original: NDArray[np.uint8], cleaned: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Combine the original and network output: network chroma, mixed luma, light sharpening.

    Returns:
        The blended RGB image.
    """
    source = cv2.cvtColor(original, cv2.COLOR_RGB2LAB)
    network = cv2.cvtColor(cleaned, cv2.COLOR_RGB2LAB)
    luma = LUMA_BLEND * network[..., 0].astype(np.float32) + (1 - LUMA_BLEND) * source[
        ..., 0
    ].astype(np.float32)
    luma += SHARPEN_AMOUNT * (luma - cv2.GaussianBlur(luma, (0, 0), SHARPEN_SIGMA))
    network[..., 0] = np.clip(np.round(luma), 0, MAX_8).astype(np.uint8)
    return cv2.cvtColor(network, cv2.COLOR_LAB2RGB)


def clean(
    rgb: NDArray[np.uint8], report: Callable[[float], None] = lambda _fraction: None
) -> NDArray[np.uint8]:
    """
    Remove colored speckle from an sRGB image while keeping its detail.

    Returns:
        A new RGB array of the same shape.
    """
    return blend_detail(rgb, _run_tiles(rgb, report))
