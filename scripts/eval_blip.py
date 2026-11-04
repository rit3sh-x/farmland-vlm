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

from farmvlm import folds as F, hub
from farmvlm.evalset import eval_items
from farmvlm.evaluate import VIEWS_D4, predict, predict_tta, summarise, write
from farmvlm.images import SIZE, rgb_view
from farmvlm.perturb import apply, remap, variants
from farmvlm.scheme import VIEWS
from farmvlm.stats import StatsTable

ROBUST_TILES = 500


def recipe(vqa_adapter, cap_adapter) -> dict:
    """How the adapters were trained, from their runs' config.json: each task's base model, and the input size and bands
    both tasks' images share. Runs from before these options read as BLIP base, 384 px, RGB."""
    r, inputs = {"vqa_base": "base", "cap_base": "base", "size": SIZE, "bands": "rgb"}, set()
    for key, adapter in (("vqa_base", vqa_adapter), ("cap_base", cap_adapter)):
        cfg = Path(adapter).parent / "config.json" if adapter else None
        if cfg and cfg.exists():
            c = json.loads(cfg.read_text())
            r[key] = c.get("base", "base")
            inputs.add((c.get("size", SIZE), c.get("bands", "rgb")))
    assert len(inputs) <= 1, f"the two adapters were trained on different inputs: {inputs}"
    if inputs:
        r["size"], r["bands"] = inputs.pop()
    return r


def load_reader(vqa_adapter, cap_adapter, beams=1, r=None):
    import torch
    from peft import PeftModel
    from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering
    from farmvlm.blip import BASES, BlipReader
    r = r or recipe(vqa_adapter, cap_adapter)
    vqa = BlipForQuestionAnswering.from_pretrained(BASES[r["vqa_base"]][0], dtype=torch.float16)
    cap = BlipForConditionalGeneration.from_pretrained(BASES[r["cap_base"]][1], dtype=torch.float16)
    if vqa_adapter:
        vqa = PeftModel.from_pretrained(vqa, vqa_adapter).merge_and_unload()
    if cap_adapter:
        cap = PeftModel.from_pretrained(cap, cap_adapter).merge_and_unload()
    return BlipReader(vqa=vqa.half().cuda().eval(), captioner=cap.half().cuda().eval(), beams=beams, size=r["size"],
                      base=r["vqa_base"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vqa-adapter", help="path, may contain {fold}")
    ap.add_argument("--caption-adapter", help="path, may contain {fold}")
    ap.add_argument("--suite", choices=["clean", "robustness", "leakage", "validation"], default="clean",
                    help="validation: the fold's inner-validation items (train_blip's split and draw), where readouts are chosen")
    ap.add_argument("--val-tiles", type=int, default=1000)
    ap.add_argument("--tta", type=int, default=1, help="views per tile for the ranked readout (1, 4 or 8; see VIEWS_D4)")
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--beams", type=int, default=1, help="beam search width for generated answers and captions (1: greedy)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    first = lambda p: p.format(fold=a.folds[0]) if p else None
    rec = recipe(first(a.vqa_adapter), first(a.caption_adapter))
    (a.out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "commit": sha, "recipe": rec}))
    t = StatsTable.load(a.stats)
    fold = F.make_folds(t, a.seed)
    field_of = lambda r: str(t.field[t.row_tile[r]])

    @lru_cache(maxsize=8192)
    def base(row):
        ti = t.row_tile[row]
        return rgb_view(a.root, t.split[ti], t.tile[ti], VIEWS[t.row_view[row]], rec["size"], rec["bands"])

    def run(reader, items, name, image=base, heldout=False, ranked=False):
        # Each result is pushed as it lands and skipped on a rerun, so a recycled Colab VM loses one result, not the suite.
        if (a.out / f"{name}.json").exists():
            print(f"{name}: already scored", flush=True)
            return
        t0 = time.time()
        rows = (predict_tta(reader, items, image, VIEWS_D4[:a.tta], a.batch) if ranked and a.tta > 1
                else predict(reader, items, image, a.batch, heldout=heldout, ranked=ranked))
        m = summarise(rows, ranked=ranked)
        write(rows, m, a.out / name, field_of)
        hub.push(a.out, a.out.name)
        print(f"{name}: {len(items)} items {time.time() - t0:.0f}s acc {m['vqa_generated']['accuracy']:.3f}", flush=True)

    if a.suite == "leakage":
        reader = load_reader(a.vqa_adapter, a.caption_adapter, a.beams, rec)
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
        reader = load_reader(fmt(a.vqa_adapter), fmt(a.caption_adapter), a.beams, rec)
        if a.suite == "validation":
            from farmvlm.train_blip import split_tiles
            _, pool = split_tiles(t, "folds", f, a.seed)
            val = np.random.default_rng(a.seed).choice(pool, min(a.val_tiles, len(pool)), replace=False)
            run(reader, eval_items(t, val, np.array([], int)), f"val-f{f}", ranked=True)
            continue
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
