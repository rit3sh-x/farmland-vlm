"""The served-model bake-off and vision test (issues/07 and 13): train each candidate on the same 10k pairs,
take the fastest within 2 points of the best, then test adapting the winner's vision tower.

    python scripts/bakeoff.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 --out /content/runs/bakeoff
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm.vlm import SPEED_ORDER  # noqa: E402

MARGIN = 0.02
VISION_WIN = 0.02


def train(model, out, a, *extra):
    if not (out / "done.json").exists():
        subprocess.run([sys.executable, "-m", "farmvlm.train_vlm", "--model", model, "--stats", a.stats, "--root", a.root,
                        "--pairs", str(a.pairs), "--eval-every", str(a.pairs // 2), "--out", out, *extra], check=True)
    return json.loads((out / "done.json").read_text())["best_val"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=10_000)
    a = ap.parse_args()
    scores = {m: train(m, a.out / m, a) for m in SPEED_ORDER}
    best = max(scores.values())
    winner = next(m for m in SPEED_ORDER if scores[m] >= best - MARGIN)
    vision = train(winner, a.out / f"{winner}-vision", a, "--vision")
    decision = {"scores": scores, "winner": winner, "vision_score": vision,
                "merged_route": vision >= scores[winner] + VISION_WIN}
    (a.out / "bakeoff.json").write_text(json.dumps(decision, indent=1))
    print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
