"""Score BLIP (zero-shot, or with LoRA adapters) on every fold's test sample, or run the robustness suite.

    python scripts/eval_blip.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 --out /content/runs/blip-zeroshot
    python scripts/eval_blip.py ... --vqa-adapter '/content/runs/vqa-f{fold}/best' --caption-adapter '/content/runs/cap-f{fold}/best'
    python scripts/eval_blip.py ... --suite robustness --folds 0
    python scripts/eval_blip.py ... --suite leakage --vqa-adapter /content/runs/vqa-official/best
"""
import argparse
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from farmvlm import folds as F
from farmvlm.evalset import eval_items
from farmvlm.evaluate import predict, summarise, write
from farmvlm.images import rgb_view
from farmvlm.perturb import apply, remap, variants
from farmvlm.scheme import VIEWS
from farmvlm.stats import StatsTable

ROBUST_TILES = 500


def load_reader(vqa_adapter, cap_adapter):
    import torch
    from peft import PeftModel
    from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering
    from farmvlm.blip import CAPTION_ID, VQA_ID, BlipReader
    vqa = BlipForQuestionAnswering.from_pretrained(VQA_ID, dtype=torch.float16)
    cap = BlipForConditionalGeneration.from_pretrained(CAPTION_ID, dtype=torch.float16)
    if vqa_adapter:
        vqa = PeftModel.from_pretrained(vqa, vqa_adapter).merge_and_unload()
    if cap_adapter:
        cap = PeftModel.from_pretrained(cap, cap_adapter).merge_and_unload()
    return BlipReader(vqa=vqa.half().cuda().eval(), captioner=cap.half().cuda().eval())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vqa-adapter", help="path, may contain {fold}")
    ap.add_argument("--caption-adapter", help="path, may contain {fold}")
    ap.add_argument("--suite", choices=["clean", "robustness", "leakage"], default="clean")
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (a.out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "commit": sha}))
    t = StatsTable.load(a.stats)
    fold = F.make_folds(t, a.seed)
    field_of = lambda r: str(t.field[t.row_tile[r]])

    @lru_cache(maxsize=8192)
    def base(row):
        ti = t.row_tile[row]
        return rgb_view(a.root, t.split[ti], t.tile[ti], VIEWS[t.row_view[row]])

    def run(reader, items, name, image=base, heldout=False, ranked=False):
        t0 = time.time()
        rows = predict(reader, items, image, a.batch, heldout=heldout, ranked=ranked)
        m = summarise(rows, ranked=ranked)
        write(rows, m, a.out / name, field_of)
        print(f"{name}: {len(items)} items {time.time() - t0:.0f}s acc {m['vqa_generated']['accuracy']:.3f}", flush=True)

    if a.suite == "leakage":
        reader = load_reader(a.vqa_adapter, a.caption_adapter)
        train_fields = set(t.field[t.split == "train"])
        val = np.flatnonzero(t.split == "val")
        seen = np.array([t.field[i] in train_fields for i in val])
        rng = np.random.default_rng(a.seed)
        n = min(2000, seen.sum(), (~seen).sum())
        for name, pool in (("seen-field", val[seen]), ("unseen-field", val[~seen])):
            run(reader, eval_items(t, rng.choice(pool, n, replace=False), np.array([], int)), name)
        return
    for f in a.folds:
        fmt = lambda p: p.format(fold=f) if p else None
        reader = load_reader(fmt(a.vqa_adapter), fmt(a.caption_adapter))
        tiles = F.pick_test_sample(t, fold, f, a.seed)
        if a.suite == "clean":
            run(reader, eval_items(t, tiles, F.clean_views(t, np.flatnonzero(fold == f), a.seed)), f"f{f}", ranked=True)
            continue
        items = eval_items(t, np.random.default_rng(a.seed).permutation(tiles)[:ROBUST_TILES], np.array([], int))
        run(reader, items, f"f{f}-clean")
        run(reader, items, f"f{f}-paraphrase", heldout=True)
        for kind, s in variants():
            run(reader, [remap(i, kind) for i in items], f"f{f}-{kind}-{s}",
                image=lambda row, kind=kind, s=s: apply(base(row), kind, s, seed=row))


if __name__ == "__main__":
    main()
