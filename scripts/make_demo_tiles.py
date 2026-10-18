"""Pick the demo gallery: tiles from fold 0's test sample (never trained on by the fold-0 served model),
covering every class, with their ground-truth findings. Writes into a folder that must stay private.

    python scripts/make_demo_tiles.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 --out /content/demo_tiles
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from farmvlm import folds as F
from farmvlm.items import findings_of
from farmvlm.scheme import CLASSES, VIEWS
from farmvlm.stats import StatsTable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    t = StatsTable.load(a.stats)
    fold = F.make_folds(t, a.seed)
    sample = F.pick_test_sample(t, fold, 0, a.seed)
    full_row = {int(t.row_tile[r]): r for r in np.flatnonzero(t.row_view == VIEWS.index("full"))}
    rng = np.random.default_rng(a.seed)
    chosen = []
    for c in range(len(CLASSES)):
        pool = [ti for ti in sample if ti in full_row and t.area[full_row[ti], c] > 0.02 and ti not in chosen]
        chosen += [int(x) for x in rng.choice(pool, min(2, len(pool)), replace=False)]
    rest = [ti for ti in sample if ti in full_row and ti not in chosen]
    chosen += [int(x) for x in rng.choice(rest, max(0, a.n - len(chosen)), replace=False)]
    a.out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for ti in chosen[: a.n]:
        stem, split, r = str(t.tile[ti]), str(t.split[ti]), full_row[ti]
        shutil.copy(a.root / split / "images" / "rgb" / f"{stem}.jpg", a.out / f"{stem}.jpg")
        shutil.copy(a.root / split / "images" / "nir" / f"{stem}.jpg", a.out / f"{stem}_nir.jpg")
        truth = [{"cls": f.cls, "extent": f.extent, "loc": f.loc, "possibly": f.tiny, "hedge": f.hedged, "conf": None}
                 for f in findings_of(t.area[r], t.cells[r])]
        names = ", ".join(CLASSES[f["cls"]].name for f in truth) or "clean"
        tiles.append({"id": stem, "label": f"{names} · {int(t.year[ti])}", "year": int(t.year[ti]), "has_nir": True, "truth": truth})
    (a.out / "tiles.json").write_text(json.dumps(tiles, indent=1))
    print(len(tiles), "demo tiles")


if __name__ == "__main__":
    main()
