"""Baseline vs BLIP-LoRA: where each looks when asked about a present anomaly (fold 0's test sample).

    python scripts/heatmaps.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 \
        --adapter /content/runs/blip-vqa-f0/best --out /content/runs/heatmaps
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm import folds as F  # noqa: E402
from farmvlm.heatmap import BlipAttention, mask_grid, pointing  # noqa: E402
from farmvlm.images import rgb_view  # noqa: E402
from farmvlm.items import Item, render_question  # noqa: E402
from farmvlm.scheme import CLASSES, VIEW_SLICES, VIEWS  # noqa: E402
from farmvlm.stats import StatsTable  # noqa: E402


def class_mask(root: Path, split: str, stem: str, key: str, view: str) -> np.ndarray:
    base = root / split
    rs, cs = VIEW_SLICES[view]
    m = np.asarray(Image.open(base / "labels" / key / f"{stem}.png")) > 0
    valid = (np.asarray(Image.open(base / "boundaries" / f"{stem}.png")) > 0) & (np.asarray(Image.open(base / "masks" / f"{stem}.png")) > 0)
    return (m & valid)[rs, cs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t = StatsTable.load(a.stats)
    fold = F.make_folds(t, a.seed)
    tiles = F.pick_test_sample(t, fold, 0, a.seed)
    full = np.flatnonzero(np.isin(t.row_tile, tiles) & (t.row_view == VIEWS.index("full")))
    pairs = [(int(r), int(k)) for r in full for k in np.flatnonzero(t.area[r] > 0.02)]
    rng = np.random.default_rng(a.seed)
    pairs = [pairs[i] for i in rng.permutation(len(pairs))[: a.n]]

    from peft import PeftModel
    from transformers import BlipForQuestionAnswering
    from farmvlm.blip import VQA_ID
    readers = {"baseline": BlipAttention()}
    tuned = BlipForQuestionAnswering.from_pretrained(VQA_ID, attn_implementation="eager")
    readers["blip-lora"] = BlipAttention(PeftModel.from_pretrained(tuned, a.adapter).merge_and_unload())

    results = {name: [] for name in readers}
    keep = {}
    for j, (r, k) in enumerate(pairs):
        ti, view = t.row_tile[r], VIEWS[t.row_view[r]]
        img = rgb_view(a.root, t.split[ti], t.tile[ti], view)
        q = render_question(Item(r, "presence", k, "yes", False), 0)
        cover = mask_grid(class_mask(a.root, t.split[ti], t.tile[ti], CLASSES[k].key, view), 24)
        for name, reader in readers.items():
            h = reader.heat(img, q)
            results[name].append({"row": r, "cls": CLASSES[k].key, **pointing(h, cover)})
            if j < 24:
                keep[f"{name}/{j}"] = h
        if j < 24:
            keep[f"mask/{j}"] = cover
            keep[f"row/{j}"] = np.array([r, k])
    np.savez_compressed(a.out / "examples.npz", **keep)
    summary = {name: {"hit_rate": float(np.mean([x["hit"] for x in rs])), "energy": float(np.mean([x["energy"] for x in rs])),
                      "chance": float(np.mean([x["chance"] for x in rs])), "n": len(rs)} for name, rs in results.items()}
    (a.out / "pointing.json").write_text(json.dumps({"summary": summary, "per_pair": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
