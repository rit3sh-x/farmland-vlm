"""Model inputs: any view of a tile as a 384 px RGB image, or as an NDVI map in the fixed colour ramp.

The ramp and range are shared with the UI (red = stressed or bare, green = healthy), so the NDVI
variant's second image is exactly what the user sees.
"""
from pathlib import Path

import numpy as np
from PIL import Image

from farmvlm.scheme import VIEW_SLICES, VIEWS

SIZE = 384
NDVI_RANGE = (-0.2, 0.8)  # week 1: set from the measured NDVI distribution of the tiles
RAMP = np.array([[165, 0, 38], [244, 109, 67], [254, 224, 139], [166, 217, 106], [26, 152, 80]], np.float32)


def _crop(a: np.ndarray, view: str) -> np.ndarray:
    rs, cs = VIEW_SLICES[view]
    return a[rs, cs]


def rgb_view(root: Path, split: str, stem: str, view: str = "full") -> Image.Image:
    a = np.asarray(Image.open(root / split / "images" / "rgb" / f"{stem}.jpg").convert("RGB"))
    return Image.fromarray(_crop(a, view)).resize((SIZE, SIZE), Image.BICUBIC)


def ndvi(rgb: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Relative NDVI from 8-bit JPEG bands (uncalibrated cameras, so only comparable within the dataset)."""
    r, n = rgb[..., 0].astype(np.float32), nir.astype(np.float32)
    return (n - r) / (n + r + 1e-6)


def colourise(v: np.ndarray, lo_hi=NDVI_RANGE) -> np.ndarray:
    lo, hi = lo_hi
    x = np.clip((v - lo) / (hi - lo), 0, 1) * (len(RAMP) - 1)
    i = np.minimum(x.astype(int), len(RAMP) - 2)
    t = (x - i)[..., None]
    return (RAMP[i] * (1 - t) + RAMP[i + 1] * t).astype(np.uint8)


def ndvi_view(root: Path, split: str, stem: str, view: str = "full") -> Image.Image:
    base = root / split / "images"
    rgb = np.asarray(Image.open(base / "rgb" / f"{stem}.jpg").convert("RGB"))
    nir = np.asarray(Image.open(base / "nir" / f"{stem}.jpg").convert("L"))
    return Image.fromarray(colourise(ndvi(_crop(rgb, view), _crop(nir, view)))).resize((SIZE, SIZE), Image.BILINEAR)


def view_name(view_index: int) -> str:
    return VIEWS[view_index]
