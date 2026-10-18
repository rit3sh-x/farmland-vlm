"""Score BLIP (zero-shot, or with LoRA adapters) on every fold's test sample, or run the robustness suite.

    python scripts/eval_blip.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 --out /content/runs/blip-zeroshot
    python scripts/eval_blip.py ... --vqa-adapter '/content/runs/vqa-f{fold}/best' --caption-adapter '/content/runs/cap-f{fold}/best'
    python scripts/eval_blip.py ... --suite robustness --folds 0
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
from farmvlm.images import rgb_view
from farmvlm.items import caption_refs, render_question
from farmvlm.metrics import caption_facts, caption_overlap, vqa_scores
from farmvlm.perturb import apply, remap, variants
from farmvlm.scheme import CLASSES, EXTENTS, POSSIBLY, VIEWS
from farmvlm.stats import StatsTable

ROBUST_TILES = 500


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def candidates(items):
    by = {"presence": ["yes", "no", POSSIBLY],
          "extent": list(EXTENTS) + [f"{a} to {b}" for a, b in zip(EXTENTS, EXTENTS[1:])]}
    for fam in ("location", "identify"):
        by[fam] = sorted({i.answer for i in items if i.family == fam})
    return by


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
    return BlipReader(vqa=vqa.half(), captioner=cap.half())


def score(reader, items, image, out: Path, ranked: bool, heldout: bool, batch: int, field_of):
    vqa = [i for i in items if i.family != "caption"]
    caps = [i for i in items if i.family == "caption"]
    cands = candidates(vqa)
    rows = []
    for b in chunks(vqa, batch):
        imgs = [image(i.row) for i in b]
        qs = [render_question(i, k, heldout=heldout) for k, i in enumerate(b)] if heldout else [render_question(i, 0) for i in b]
        gen = reader.answer(imgs, qs)
        rk = reader.rank(imgs, qs, [cands[i.family] for i in b]) if ranked else [(None, None)] * len(b)
        for i, q, (g, gc), (r, rc) in zip(b, qs, gen, rk):
            rows.append({"row": i.row, "field": field_of(i.row), "family": i.family,
                         "cls": CLASSES[i.cls].key if i.cls >= 0 else None, "question": q, "truth": i.answer,
                         "ambiguous": i.ambiguous, "pred": g, "conf": gc, "ranked": r, "ranked_conf": rc})
    captions = []
    for b in chunks(caps, batch):
        captions += reader.caption([image(i.row) for i in b])
    with open(out.with_suffix(".jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
        for i, (c, cc) in zip(caps, captions):
            fh.write(json.dumps({"row": i.row, "field": field_of(i.row), "family": "caption", "refs": caption_refs(i),
                                 "pred": c, "conf": cc, "ambiguous": i.ambiguous}) + "\n")
    m = {"items": len(items), "vqa_generated": vqa_scores(vqa, [r["pred"] for r in rows]),
         "caption_facts": caption_facts(caps, [c for c, _ in captions]),
         "caption_overlap": caption_overlap([caption_refs(i) for i in caps], [c for c, _ in captions])}
    if ranked:
        m["vqa_ranked"] = vqa_scores(vqa, [r["ranked"] for r in rows])
    out.with_suffix(".json").write_text(json.dumps(m, indent=1))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vqa-adapter", help="path, may contain {fold}")
    ap.add_argument("--caption-adapter", help="path, may contain {fold}")
    ap.add_argument("--suite", choices=["clean", "robustness"], default="clean")
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

    for f in a.folds:
        fmt = lambda p: p.format(fold=f) if p else None
        reader = load_reader(fmt(a.vqa_adapter), fmt(a.caption_adapter))
        tiles = F.pick_test_sample(t, fold, f, a.seed)
        t0 = time.time()
        if a.suite == "clean":
            items = eval_items(t, tiles, F.clean_views(t, np.flatnonzero(fold == f), a.seed))
            m = score(reader, items, base, a.out / f"f{f}", True, False, a.batch, field_of)
            print(f"fold {f}: {len(items)} items, {time.time() - t0:.0f}s, acc {m['vqa_generated']['accuracy']:.3f}"
                  f" ranked {m['vqa_ranked']['accuracy']:.3f} cider {m['caption_overlap']['cider']:.3f}", flush=True)
            continue
        sub = np.random.default_rng(a.seed).permutation(tiles)[:ROBUST_TILES]
        items = eval_items(t, sub, np.array([], int))
        score(reader, items, base, a.out / f"f{f}-clean", False, False, a.batch, field_of)
        score(reader, items, base, a.out / f"f{f}-paraphrase", False, True, a.batch, field_of)
        for kind, s in variants():
            img = lambda row, kind=kind, s=s: apply(base(row), kind, s, seed=row)
            m = score(reader, [remap(i, kind) for i in items], img, a.out / f"f{f}-{kind}-{s}", False, False, a.batch, field_of)
            print(f"fold {f} {kind} {s}: acc {m['vqa_generated']['accuracy']:.3f}", flush=True)


if __name__ == "__main__":
    main()
