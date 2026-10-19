"""A few-minute check of every GPU path on a handful of pairs, before committing hours to the pipeline.

    python scripts/smoke_gpu.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="/content/smoke")
    a = ap.parse_args()
    env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
    common = ["--stats", a.stats, "--root", a.root]

    def run(name, *args):
        t0 = time.time()
        subprocess.run([sys.executable, *args, *common, "--out", f"{a.out}/{name}"], check=True, env=env, cwd=CODE)
        done = json.loads(Path(f"{a.out}/{name}/done.json").read_text())
        print(f"SMOKE {name}: ok in {time.time() - t0:.0f}s, val {done['best_val']:.3f}", flush=True)

    tiny = ["--eval-every", "48", "--val-tiles", "3"]
    run("blip-vqa", "-m", "farmvlm.train_blip", "--task", "vqa", "--pairs", "96", *tiny)
    run("blip-caption", "-m", "farmvlm.train_blip", "--task", "caption", "--pairs", "96", *tiny)
    run("blip-dora", "-m", "farmvlm.train_blip", "--task", "vqa", "--pairs", "48", "--dora", *tiny)
    vlm_tiny = ["--pairs", "32", "--eval-every", "32", "--val-tiles", "2"]
    run("vlm-rgb", "-m", "farmvlm.train_vlm", "--model", "lfm-450m", *vlm_tiny)
    run("vlm-ndvi", "-m", "farmvlm.train_vlm", "--model", "lfm-450m", "--ndvi", *vlm_tiny)
    run("vlm-vision", "-m", "farmvlm.train_vlm", "--model", "lfm-450m", "--vision", *vlm_tiny)

    sys.path.insert(0, str(CODE))
    import numpy as np
    from farmvlm import folds as F
    from farmvlm.blip import BlipReader
    from farmvlm.evalset import eval_items
    from farmvlm.evaluate import predict, summarise
    from farmvlm.images import rgb_view
    from farmvlm.scheme import VIEWS
    from farmvlm.stats import StatsTable
    t = StatsTable.load(a.stats)
    fold = F.make_folds(t, 0)
    tiles = F.pick_test_sample(t, fold, 0, 0)[:4]
    items = eval_items(t, tiles, np.array([], int))
    image = lambda r: rgb_view(Path(a.root), t.split[t.row_tile[r]], t.tile[t.row_tile[r]], VIEWS[t.row_view[r]])
    rows = predict(BlipReader(), items, image, batch=16, ranked=True)
    for r in rows[:12]:
        i = r["item"]
        print(f"SMOKE blip-zero-shot {i.family:9s} truth={i.answer!r:28s} pred={r['pred']!r} ranked={r.get('ranked')!r} conf={r['conf']:.2f}")
    m = summarise(rows, ranked=True)
    print("SMOKE blip-zero-shot summary", json.dumps({k: m[k]["accuracy"] for k in ("vqa_generated", "vqa_ranked")}))


if __name__ == "__main__":
    main()
