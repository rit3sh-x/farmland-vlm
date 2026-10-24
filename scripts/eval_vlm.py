"""Score a served-model candidate (zero-shot or with an adapter) on fold 0's test sample, or run the robustness suite.

    python scripts/eval_vlm.py --model lfm-450m --adapter /content/runs/final-rgb/best --stats ... --root ... --out ...
    python scripts/eval_vlm.py ... --ndvi --adapter /content/runs/final-ndvi/best --suite robustness
"""
import argparse
import json
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm import folds as F, vlm  # noqa: E402
from farmvlm.evalset import eval_items  # noqa: E402
from farmvlm.evaluate import predict, summarise, write  # noqa: E402
from farmvlm.perturb import apply, remap  # noqa: E402
from farmvlm.stats import StatsTable  # noqa: E402
from farmvlm.train_vlm import view_image  # noqa: E402

# Qwen generates ~0.4 s per item on a T4: the first 200 of BLIP's 500 robustness tiles (same seed, so the same
# items) and the middle strength of each perturbation, instead of BLIP's full 18-pass suite.
ROBUST_TILES = 200
ROBUST_VARIANTS = (("noise", 15), ("blur", 2), ("jpeg", 20), ("light", 0.25), ("resolution", 3), ("hflip", None), ("rot90", None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(vlm.MODELS), required=True)
    ap.add_argument("--adapter", type=Path)
    ap.add_argument("--ndvi", action="store_true")
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--suite", choices=["clean", "robustness"], default="clean")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "-C", str(Path(__file__).parent), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    (a.out / "config.json").write_text(json.dumps({**{k: str(v) for k, v in vars(a).items()}, "commit": sha}))
    t = StatsTable.load(a.stats)
    fold = F.make_folds(t, a.seed)
    model, proc = vlm.load(a.model)
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    reader = vlm.VlmReader(model.eval(), proc)
    field_of = lambda r: str(t.field[t.row_tile[r]])
    base = lru_cache(maxsize=4096)(lambda row: view_image(t, a.root, row, a.ndvi))
    tiles = F.pick_test_sample(t, fold, a.fold, a.seed)

    def run(items, name, image=base, heldout=False):
        t0 = time.time()
        rows = predict(reader, items, image, a.batch, heldout=heldout)
        m = summarise(rows)
        write(rows, m, a.out / name, field_of)
        print(f"{name}: {len(items)} items {time.time() - t0:.0f}s acc {m['vqa_generated']['accuracy']:.3f}", flush=True)

    if a.suite == "clean":
        run(eval_items(t, tiles, F.clean_views(t, np.flatnonzero(fold == a.fold), a.seed)), f"f{a.fold}")
        return
    items = eval_items(t, np.random.default_rng(a.seed).permutation(tiles)[:ROBUST_TILES], np.array([], int))
    run(items, f"f{a.fold}-clean")
    run(items, f"f{a.fold}-paraphrase", heldout=True)
    for kind, s in ROBUST_VARIANTS:
        def image(row, kind=kind, s=s):
            img = base(row)
            return tuple(apply(i, kind, s, seed=row) for i in img) if isinstance(img, tuple) else apply(img, kind, s, seed=row)
        run([remap(i, kind) for i in items], f"f{a.fold}-{kind}-{s}", image=image)


if __name__ == "__main__":
    main()
