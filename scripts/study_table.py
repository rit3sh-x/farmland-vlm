"""The hyperparameter study as a table, one row per trial: what changed from the reference, validation (the four-question
mean, exact and partial credit), the paired change against the reference with a field-grouped 95% interval, and what it
cost (trainable parameters, GPU minutes, peak GPU memory, throughput).

    python scripts/mirror.py && python scripts/study_table.py [--task vqa] [--out report/tables/study-vqa.md]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import FAMILIES, best_check, compare  # noqa: E402
from farmvlm.report import gpu_minutes  # noqa: E402

KNOBS = ("base", "size", "bands", "rank", "alpha", "lr", "targets", "dropout", "train_norms", "pairs", "train_seed")


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in open(path)] if path.exists() else []


def config(run: Path) -> dict:
    c = json.loads((run / "config.json").read_text())
    return {**c, "train_seed": c.get("train_seed", c["seed"])}  # runs before --train-seed seeded training with --seed


def row(run: Path, ref: Path, task: str) -> dict:
    cfg, ref_cfg = config(run), config(ref)
    same_length = ref.parent / f"pairs{cfg['pairs'] // 1000}k"
    if cfg["pairs"] != ref_cfg["pairs"] and run != same_length and same_length.exists():  # a knob at another length
        ref, ref_cfg = same_length, config(same_length)
    done = json.loads((run / "done.json").read_text()) if (run / "done.json").exists() else {}
    check = best_check(run)
    m = json.loads(check.with_suffix(".json").read_text())
    change = ", ".join(f"{k}={cfg.get(k)}" for k in KNOBS if cfg.get(k) != ref_cfg.get(k)) or "reference"
    r = {"trial": run.name, "pairs": cfg["pairs"], "change": change + (f" (vs {ref.name})" if ref.name != "ref" else ""),
         "done": "done.json" in {p.name for p in run.iterdir()}, "trainable_m": cfg["trainable"] / 1e6,
         "gpu_min": gpu_minutes(run), "peak_gb": done.get("peak_gpu_gb"),
         "pairs_per_s": statistics.median([x["train/pairs_per_s"] for x in rows(run / "train.jsonl")] or [float("nan")])}
    if task == "vqa":
        g = m["vqa_generated"]
        r["val"] = sum(g["by_family"][f] for f in FAMILIES) / len(FAMILIES)
        r["partial"] = sum(g["partial_by_family"][f] for f in FAMILIES) / len(FAMILIES)
        r["delta"] = compare(best_check(ref), check)["exact_mean_change"] if run != ref else (0.0, 0.0, 0.0)
    else:
        r["val"] = m["caption_facts"]["f1"]
        r["delta"] = (r["val"] - json.loads(best_check(ref).with_suffix(".json").read_text())["caption_facts"]["f1"], None, None)
    return r


def seed_spread(rows: list[dict]) -> tuple[float, float, int] | None:
    """(mean, sd, n) of validation over the finished reference and its training-seed replicates (ref, ref-seed*). The
    field bootstrap covers item sampling only; this is the run-to-run noise every single-run trial also carries."""
    vals = [r["val"] for r in rows if r["done"] and (r["trial"] == "ref" or r["trial"].startswith("ref-seed"))]
    return (statistics.mean(vals), statistics.stdev(vals), len(vals)) if len(vals) >= 3 else None


def table(runs: Path, task: str) -> str:
    root, ref = runs / f"study-{task}", runs / f"study-{task}" / "ref"
    found = [d for d in sorted(root.iterdir()) if (d / "val").exists()] if root.exists() else []
    rs = [row(d, ref, task) for d in sorted(found, key=lambda d: (d.name != "ref", d.name))]
    spread, ref_pairs = seed_spread(rs), json.loads((ref / "config.json").read_text())["pairs"]
    lines = ["| trial | change | val | partial | change vs ref (95% CI) | vs seed mean (sd units) | trainable M | GPU min "
             "| peak GB | pairs/s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        vs_seeds = "--"
        if spread and r["pairs"] == ref_pairs:  # one run against the mean of n seeds: sd of the gap is sd * sqrt(1 + 1/n)
            m, sd, n = spread
            vs_seeds = f"{(r['val'] - m) * 100:+.1f} ({(r['val'] - m) / (sd * (1 + 1 / n) ** 0.5):+.1f})"
        delta, lo, hi = r["delta"]
        ci = f"{delta * 100:+.1f} [{lo * 100:+.1f}, {hi * 100:+.1f}]" if lo is not None else f"{delta * 100:+.1f}"
        part = f"{r['partial'] * 100:.1f}" if "partial" in r else "--"
        lines.append(f"| {r['trial']}{'' if r['done'] else ' (running)'} | {r['change']} | {r['val'] * 100:.1f} | {part} | {ci} | {vs_seeds} "
                     f"| {r['trainable_m']:.1f} | {r['gpu_min']:.0f} | {r['peak_gb'] or '--'} | {r['pairs_per_s']:.0f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--task", choices=["vqa", "caption"], default="vqa")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    t = table(a.runs, a.task)
    print(t)
    if a.out:
        a.out.write_text(t + "\n")


if __name__ == "__main__":
    main()
