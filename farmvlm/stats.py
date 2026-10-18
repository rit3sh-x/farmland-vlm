"""Per-view mask statistics: the one table every QA pair, caption and fold is derived from.

One row per usable view (full tile or quadrant): in-field pixel count, each class's share of the
in-field area, and each class's spread over the view's 3x3 grid. Built once from the official
Agriculture-Vision folders, then kept private (it is derived from the dataset).

    python -m farmvlm.stats --root /content/av/Agriculture-Vision-2021 \
        --manifest /content/av/data_manifest.csv --out /content/stats.npz
"""
import argparse
import collections
import csv
import os
import random
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

from farmvlm.scheme import CLASSES, MIN_VALID_SHARE, N_CLASSES, VIEW_SLICES, VIEWS

VERSION = 1


def grid_shares(m: np.ndarray) -> np.ndarray:
    """Share of the mask's pixels in each cell of a 3x3 grid, row-major (top left first)."""
    h, w = m.shape
    rows = [0, round(h / 3), round(2 * h / 3)]
    cols = [0, round(w / 3), round(2 * w / 3)]
    counts = np.add.reduceat(np.add.reduceat(m.astype(np.int64), rows, axis=0), cols, axis=1).ravel()
    total = counts.sum()
    return counts / total if total else np.zeros(9)


def view_rows(labels: np.ndarray, valid: np.ndarray):
    """Yield (view index, in-field pixels, area[9], cells[9, 9]) for each usable view.

    labels is bool[N_CLASSES, 512, 512]; valid is bool[512, 512] (field boundary AND valid-pixel mask).
    """
    for vi, view in enumerate(VIEWS):
        rs, cs = VIEW_SLICES[view]
        v = valid[rs, cs]
        n = int(v.sum())
        if n < MIN_VALID_SHARE * v.size:
            continue
        area = np.zeros(N_CLASSES, np.float32)
        cells = np.zeros((N_CLASSES, 9), np.float32)
        for k in range(N_CLASSES):
            m = labels[k, rs, cs] & v
            px = int(m.sum())
            if px:
                area[k] = px / n
                cells[k] = grid_shares(m)
        yield vi, n, area, cells


@dataclass
class StatsTable:
    """Columnar stats. Tile-level arrays are indexed by tile; view-level arrays by row."""
    tile: np.ndarray        # str[T]  filename stem "<flight>_<x1>-<y1>-<x2>-<y2>"
    split: np.ndarray       # str[T]  official split: train / val
    field: np.ndarray       # str[T]  manifest field (the grouping key for folds)
    year: np.ndarray        # int16[T]
    row_tile: np.ndarray    # int32[R] index into the tile arrays
    row_view: np.ndarray    # uint8[R] index into scheme.VIEWS
    valid_px: np.ndarray    # int32[R]
    area: np.ndarray        # float32[R, 9]
    cells: np.ndarray       # float16[R, 9, 9]

    def save(self, path):
        np.savez_compressed(path, version=VERSION, **self.__dict__)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        if int(z["version"]) != VERSION:
            raise ValueError(f"{path}: stats version {int(z['version'])}, code expects {VERSION}")
        return cls(**{k: z[k] for k in cls.__dataclass_fields__})

    def __len__(self):
        return len(self.row_tile)


def read_manifest(path) -> dict[str, tuple[str, int]]:
    """flight code -> (field, year). The official CSV stores numbers as floats ("2018.0")."""
    with open(path, newline="") as f:
        return {r["flightcode"]: (r["field"].removesuffix(".0"), int(float(r["year"]))) for r in csv.DictReader(f)}


def _empty_sizes(root: Path, split: str, sample: int = 40):
    """Byte size of an all-zero label PNG per class, verified by decoding a sample; None if unsure."""
    sizes = {}
    for c in CLASSES:
        d = root / split / "labels" / c.key
        files = list(os.scandir(d))
        common, _ = collections.Counter(e.stat().st_size for e in files).most_common(1)[0]
        same = [e.path for e in files if e.stat().st_size == common]
        probe = random.Random(0).sample(same, min(sample, len(same)))
        sizes[c.key] = common if all(not np.asarray(Image.open(p)).any() for p in probe) else None
    return sizes


def _load(path):
    return np.asarray(Image.open(path)) > 0


_CTX = {}


def _tile_job(args):
    split, stem = args
    root, empty = _CTX["root"], _CTX["empty"][split]
    base = root / split
    valid = _load(base / "boundaries" / f"{stem}.png") & _load(base / "masks" / f"{stem}.png")
    labels = np.zeros((N_CLASSES, 512, 512), bool)
    for k, c in enumerate(CLASSES):
        p = base / "labels" / c.key / f"{stem}.png"
        if empty[c.key] is None or p.stat().st_size != empty[c.key]:
            labels[k] = _load(p)
    return list(view_rows(labels, valid))


def build(root: Path, manifest: dict, splits=("train", "val"), workers=None) -> StatsTable:
    jobs = [(s, p.stem) for s in splits for p in sorted((root / s / "images" / "rgb").glob("*.jpg"))]
    _CTX.update(root=root, empty={s: _empty_sizes(root, s) for s in splits})
    tiles, tsplit, tfield, tyear = [], [], [], []
    rows = {k: [] for k in ("row_tile", "row_view", "valid_px", "area", "cells")}
    with Pool(workers or os.cpu_count()) as pool:
        for ti, ((split, stem), views) in enumerate(zip(jobs, pool.imap(_tile_job, jobs, chunksize=64))):
            field, year = manifest[stem.split("_")[0]]
            tiles.append(stem); tsplit.append(split); tfield.append(field); tyear.append(year)
            for vi, n, area, cells in views:
                rows["row_tile"].append(ti); rows["row_view"].append(vi); rows["valid_px"].append(n)
                rows["area"].append(area); rows["cells"].append(cells)
    return StatsTable(
        tile=np.array(tiles), split=np.array(tsplit), field=np.array(tfield), year=np.array(tyear, np.int16),
        row_tile=np.array(rows["row_tile"], np.int32), row_view=np.array(rows["row_view"], np.uint8),
        valid_px=np.array(rows["valid_px"], np.int32), area=np.array(rows["area"], np.float32),
        cells=np.array(rows["cells"], np.float16),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    a = ap.parse_args()
    t = build(a.root, read_manifest(a.manifest), tuple(a.splits))
    t.save(a.out)
    print(f"{len(t.tile)} tiles, {len(t)} views -> {a.out}")


if __name__ == "__main__":
    main()
