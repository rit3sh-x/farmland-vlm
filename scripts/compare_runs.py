"""Paired comparison of two models on the same items: per question type accuracy, the change in the four-question mean
with a field-grouped 95% interval, and how many items each gets right that the other misses.

    python scripts/compare_runs.py study-vqa/ref study-vqa/size512      # two runs, each at its best validation check
    python scripts/compare_runs.py A.jsonl B.jsonl                     # any two prediction files (e.g. held-out folds)
    python scripts/compare_runs.py study-vqa/ref --checks              # every check of one run against its last

Run names resolve under --runs (the Hub mirror from scripts/mirror.py). Captions compare fact F1 (micro, over the
anomaly mentions parsed from the reference and the prediction) instead of question accuracy.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm.metrics import partial_credit, vqa_credit  # noqa: E402
from farmvlm.text import parse_caption  # noqa: E402

FAMILIES = ("presence", "identify", "extent", "location")


def best_check(run: Path) -> Path:
    """The prediction file of the run's best validation check, as the loop chose it (second half of training only)."""
    pairs = json.loads((run / "config.json").read_text())["pairs"]
    rows = [json.loads(x) for x in open(run / "metrics.jsonl")]
    k = max((i for i, r in enumerate(rows) if r["pairs"] >= pairs // 2), key=lambda i: rows[i]["val"])
    return run / "val" / f"{k:02d}.jsonl"


def load(path: Path) -> dict:
    """item key -> record, keyed so the same question about the same view matches across models."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["row"], r["family"], r.get("cls"), r.get("question"))] = r
    return out


def scores(r: dict) -> tuple[float, float]:
    """(exact, partial) credit of one VQA record."""
    s = vqa_credit(SimpleNamespace(family=r["family"], answer=r["truth"], ambiguous=r["ambiguous"]), r["pred"])
    return s, partial_credit(r["family"], r["truth"], r["pred"], s)


def facts(r: dict) -> tuple[int, int, int]:
    """(true positive, false positive, false negative) anomaly mentions of one caption record."""
    want, said = {m.cls for m in parse_caption(r["refs"][0])}, {m.cls for m in parse_caption(r["pred"])}
    return len(want & said), len(said - want), len(want - said)


def family_mean_ci(diff: np.ndarray, fam: np.ndarray, field: np.ndarray, n: int = 2000, seed: int = 0):
    """The change in the mean over question types, with a 95% interval resampling whole fields."""
    fams = [f for f in FAMILIES if (fam == f).any()]
    point = float(np.mean([diff[fam == f].mean() for f in fams]))
    uniq, inv = np.unique(field, return_inverse=True)
    sums = {f: np.bincount(inv[fam == f], diff[fam == f], len(uniq)) for f in fams}
    counts = {f: np.bincount(inv[fam == f], minlength=len(uniq)) for f in fams}
    pick = np.random.default_rng(seed).integers(0, len(uniq), size=(n, len(uniq)))
    boot = np.mean([sums[f][pick].sum(1) / np.maximum(counts[f][pick].sum(1), 1) for f in fams], axis=0)
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def compare(a_path: Path, b_path: Path) -> dict:
    a, b = load(a_path), load(b_path)
    keys = sorted(a.keys() & b.keys(), key=str)
    assert keys, f"no shared items between {a_path} and {b_path}"
    fam = np.array([a[k]["family"] for k in keys])
    field = np.array([a[k]["field"] for k in keys])
    out = {"items": len(keys), "only_a": len(a) - len(keys), "only_b": len(b) - len(keys)}
    vqa = fam != "caption"
    if vqa.any():
        sa, sb = np.array([scores(a[k]) for k in keys if a[k]["family"] != "caption"]), np.array(
            [scores(b[k]) for k in keys if b[k]["family"] != "caption"])
        f, fl = fam[vqa], field[vqa]
        for i, kind in enumerate(("exact", "partial")):
            out[kind] = {x: (float(sa[f == x, i].mean()), float(sb[f == x, i].mean())) for x in FAMILIES if (f == x).any()}
            out[f"{kind}_mean_change"] = family_mean_ci(sb[:, i] - sa[:, i], f, fl)
        out["b_fixes"], out["b_breaks"] = int(((sa[:, 0] < 1) & (sb[:, 0] == 1)).sum()), int(((sa[:, 0] == 1) & (sb[:, 0] < 1)).sum())
    if (~vqa).any():
        fa = np.array([facts(a[k]) for k in keys if a[k]["family"] == "caption"])
        fb = np.array([facts(b[k]) for k in keys if b[k]["family"] == "caption"])
        f1 = lambda m: 2 * m[0] / max(2 * m[0] + m[1] + m[2], 1)
        out["caption_fact_f1"] = (f1(fa.sum(0)), f1(fb.sum(0)))
    return out


def show(name_a: str, name_b: str, c: dict):
    print(f"{name_a}  vs  {name_b}: {c['items']} shared items")
    if "exact" in c:
        for x, (sa, sb) in c["exact"].items():
            print(f"  {x:9s} {sa:.3f} -> {sb:.3f}  ({sb - sa:+.3f})   partial {c['partial'][x][0]:.3f} -> {c['partial'][x][1]:.3f}")
        for kind in ("exact", "partial"):
            d, lo, hi = c[f"{kind}_mean_change"]
            print(f"  four-question mean change ({kind}): {d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
        print(f"  items B fixes {c['b_fixes']}, breaks {c['b_breaks']}")
    if "caption_fact_f1" in c:
        print(f"  caption fact F1 {c['caption_fact_f1'][0]:.3f} -> {c['caption_fact_f1'][1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b", nargs="?")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--checks", action="store_true", help="every validation check of run A against its last one")
    a = ap.parse_args()
    resolve = lambda x: Path(x) if x.endswith(".jsonl") else best_check(a.runs / x)
    if a.checks:
        files = sorted((a.runs / a.a / "val").glob("*.jsonl"))
        for f in files[:-1]:
            show(f.name, files[-1].name, compare(f, files[-1]))
        return
    show(a.a, a.b, compare(resolve(a.a), resolve(a.b)))


if __name__ == "__main__":
    main()
