"""Zero-shot BLIP on every fold's test sample: the Baseline rows of the graded ablation.

    python scripts/zero_shot.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 \
        --seed 0 --out /content/runs/blip-zeroshot-s0
"""
import argparse
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from farmvlm import folds as F
from farmvlm.blip import BlipReader
from farmvlm.evalset import eval_items
from farmvlm.images import rgb_view
from farmvlm.items import caption_refs, render_question
from farmvlm.metrics import caption_facts, caption_overlap, vqa_scores
from farmvlm.scheme import CLASSES, EXTENTS, POSSIBLY, VIEWS
from farmvlm.stats import StatsTable


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def candidates(items):
    by = {"presence": ["yes", "no", POSSIBLY],
          "extent": list(EXTENTS) + [f"{a} to {b}" for a, b in zip(EXTENTS, EXTENTS[1:])]}
    for fam in ("location", "identify"):
        by[fam] = sorted({i.answer for i in items if i.family == fam})
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=48)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (a.out / "config.json").write_text(json.dumps({"model": "blip-zeroshot", "seed": a.seed, "commit": sha, "folds": a.folds}))
    t = StatsTable.load(a.stats)
    fold = F.make_folds(t, a.seed)
    reader = BlipReader()

    @lru_cache(maxsize=4096)
    def image(row):
        ti = t.row_tile[row]
        return rgb_view(a.root, t.split[ti], t.tile[ti], VIEWS[t.row_view[row]])

    for f in a.folds:
        t0 = time.time()
        tiles = F.pick_test_sample(t, fold, f, a.seed)
        items = eval_items(t, tiles, F.clean_views(t, np.flatnonzero(fold == f), a.seed))
        vqa = [i for i in items if i.family != "caption"]
        caps = [i for i in items if i.family == "caption"]
        cands = candidates(vqa)
        rows = []
        for b in chunks(vqa, a.batch):
            imgs, qs = [image(i.row) for i in b], [render_question(i, 0) for i in b]
            gen = reader.answer(imgs, qs)
            ranked = reader.rank(imgs, qs, [cands[i.family] for i in b])
            for i, q, (g, gc), (r, rc) in zip(b, qs, gen, ranked):
                rows.append({"row": i.row, "field": str(t.field[t.row_tile[i.row]]), "family": i.family,
                             "cls": CLASSES[i.cls].key if i.cls >= 0 else None, "question": q, "truth": i.answer,
                             "ambiguous": i.ambiguous, "pred": g, "conf": gc, "ranked": r, "ranked_conf": rc})
        captions = []
        for b in chunks(caps, a.batch):
            captions += reader.caption([image(i.row) for i in b])
        with open(a.out / f"preds_f{f}.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            for i, (c, cc) in zip(caps, captions):
                fh.write(json.dumps({"row": i.row, "field": str(t.field[t.row_tile[i.row]]), "family": "caption",
                                     "refs": caption_refs(i), "pred": c, "conf": cc, "ambiguous": i.ambiguous}) + "\n")
        metrics = {
            "fold": f, "tiles": len(tiles), "items": len(items), "seconds": round(time.time() - t0),
            "vqa_generated": vqa_scores(vqa, [r["pred"] for r in rows]),
            "vqa_ranked": vqa_scores(vqa, [r["ranked"] for r in rows]),
            "caption_facts": caption_facts(caps, [c for c, _ in captions]),
            "caption_overlap": caption_overlap([caption_refs(i) for i in caps], [c for c, _ in captions]),
        }
        (a.out / f"metrics_f{f}.json").write_text(json.dumps(metrics, indent=1))
        print(json.dumps({k: metrics[k] for k in ("fold", "items", "seconds")}),
              "gen acc", round(metrics["vqa_generated"]["accuracy"], 3),
              "ranked acc", round(metrics["vqa_ranked"]["accuracy"], 3),
              "cider", round(metrics["caption_overlap"]["cider"], 3), flush=True)


if __name__ == "__main__":
    main()
