"""Week-1 numbers from the stats table: extent tertiles, flag rates, class counts, clean views, NDVI range.

    python scripts/explore_stats.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from farmvlm.folds import tile_presence
from farmvlm.images import ndvi
from farmvlm.items import findings_of
from farmvlm.scheme import CLASSES, VIEWS
from farmvlm.stats import StatsTable


def nice(x: float) -> float:
    """Round a cut point to two significant figures, then to the nearest 0.5 step of its decade."""
    d = 10 ** np.floor(np.log10(x))
    return round(float(round(x / d * 2) / 2 * d), 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path)
    ap.add_argument("--ndvi-tiles", type=int, default=300)
    a = ap.parse_args()
    t = StatsTable.load(a.stats)
    full = t.row_view == VIEWS.index("full")
    present = t.area[full][t.area[full] > 0]
    q = np.quantile(present, [1 / 3, 2 / 3])
    cuts = (nice(q[0]), nice(q[1]))
    flags = {"tiny": 0, "extent_borderline": 0, "location_borderline": 0, "findings": 0}
    for r in np.flatnonzero(full):
        for f in findings_of(t.area[r], t.cells[r], cuts):
            flags["findings"] += 1
            flags["tiny"] += f.tiny
            flags["extent_borderline"] += " to " in f.extent
            flags["location_borderline"] += " or " in f.loc
    pres = tile_presence(t)
    clean = (~(t.area > 0).any(1)) & ~full
    out = {
        "tiles": int(len(t.tile)), "views": int(len(t)), "full_views": int(full.sum()),
        "fields": int(len(np.unique(t.field))),
        "tiles_per_class": {c.key: int(pres[:, i].sum()) for i, c in enumerate(CLASSES)},
        "classes_per_tile": {int(k): int(v) for k, v in zip(*np.unique(pres.sum(1), return_counts=True))},
        "area_tertiles_raw": [float(x) for x in q], "extent_cuts": cuts,
        "area_quantiles": {p: float(np.quantile(present, p)) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "flag_rates": {k: v / flags["findings"] for k, v in flags.items() if k != "findings"},
        "clean_quadrant_views": int(clean.sum()), "quadrant_views": int((~full).sum()),
    }
    if a.root:
        rng = np.random.default_rng(0)
        vals = []
        for ti in rng.choice(len(t.tile), a.ndvi_tiles, replace=False):
            base = a.root / t.split[ti] / "images"
            rgb = np.asarray(Image.open(base / "rgb" / f"{t.tile[ti]}.jpg").convert("RGB"))
            nir = np.asarray(Image.open(base / "nir" / f"{t.tile[ti]}.jpg").convert("L"))
            v = ndvi(rgb, nir)
            vals.append(v[(rgb.sum(-1) > 0)].ravel()[::16])
        v = np.concatenate(vals)
        out["ndvi_percentiles"] = {p: float(np.percentile(v, p)) for p in (1, 5, 50, 95, 99)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
