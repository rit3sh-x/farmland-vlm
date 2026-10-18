"""Robustness variants (issues/06-evaluation-protocol.md): photometric, resolution loss, geometric with remapped answers."""
import io
from dataclasses import replace

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from farmvlm.items import Finding, Item
from farmvlm.scheme import CELLS, THROUGHOUT

PHOTOMETRIC = {
    "noise": (5, 15, 30),
    "blur": (1, 2, 4),
    "jpeg": (50, 20, 10),
    "light": (0.10, 0.25, 0.40),
}
RESOLUTION = (2, 3, 6)
GEOMETRIC = ("hflip", "rot90")


def variants():
    """Every (kind, strength) pair the suite runs, after the clean and paraphrase passes."""
    out = [(k, s) for k, ss in PHOTOMETRIC.items() for s in ss]
    return out + [("resolution", f) for f in RESOLUTION] + [(g, None) for g in GEOMETRIC]


def apply(img: Image.Image, kind: str, s, seed: int = 0) -> Image.Image:
    if kind == "noise":
        a = np.asarray(img).astype(np.float32) + np.random.default_rng(seed).normal(0, s, (*img.size[::-1], 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(s))
    if kind == "jpeg":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=s)
        return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if kind == "light":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(1 + s)).enhance(1 - s)
    if kind == "resolution":
        w, h = img.size
        return img.resize((w // s, h // s), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if kind == "hflip":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if kind == "rot90":
        return img.transpose(Image.Transpose.ROTATE_90)
    raise ValueError(kind)


def _cell(word: str, kind: str) -> str:
    if word == THROUGHOUT:
        return word
    r, c = divmod(CELLS.index(word), 3)
    r, c = (r, 2 - c) if kind == "hflip" else (2 - c, r)
    return CELLS[3 * r + c]


def remap_loc(loc: str, kind: str) -> str:
    return " or ".join(_cell(w, kind) for w in loc.split(" or "))


def remap(item: Item, kind: str) -> Item:
    """The same item as the masks would read after a flip or counter-clockwise 90 degree rotation."""
    if kind not in GEOMETRIC:
        return item
    fs = tuple(Finding(f.cls, f.area, f.extent, remap_loc(f.loc, kind), f.tiny) for f in item.findings)
    answer = remap_loc(item.answer, kind) if item.family == "location" else item.answer
    return replace(item, answer=answer, findings=fs)
