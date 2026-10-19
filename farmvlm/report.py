"""Every results table of issues/06-evaluation-protocol.md, rebuilt from the run records in runs/.

    python -m farmvlm.report --runs runs --out report/tables
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from farmvlm.metrics import ece, grouped_bootstrap, is_hedge, risk_coverage, vqa_credit
from farmvlm.items import Item
from farmvlm.scheme import CLASSES, CLASS_INDEX

FAMILIES = ("presence", "identify", "extent", "location")


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def item_of(r: dict) -> Item:
    return Item(r["row"], r["family"], CLASS_INDEX[r["cls"]] if r.get("cls") else -1, r["truth"], r["ambiguous"])


def credits(rs, key="pred"):
    return np.array([vqa_credit(item_of(r), r[key]) for r in rs if r["family"] != "caption" and r.get(key) is not None])


def mean_std(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (None, None)


def fmt(m, s=None, pct=True):
    if m is None:
        return "--"
    k = 100 if pct else 1
    return f"{m * k:.1f}" + (f" $\\pm$ {s * k:.1f}" if s is not None else "")


def table_ablation(runs: Path) -> list[dict]:
    """Table 1: Baseline vs BLIP-LoRA over 5 folds."""
    out = []
    for name, label in (("baseline", "Zero-shot BLIP"), ("blip-lora", "BLIP-LoRA")):
        ms = [json.loads(p.read_text()) for p in sorted((runs / name).glob("f[0-4].json"))]
        if not ms:
            continue
        row = {"model": label, "folds": len(ms)}
        for fam in FAMILIES:
            row[fam] = mean_std([m["vqa_generated"]["by_family"].get(fam) for m in ms])
            row[fam + "_ranked"] = mean_std([m.get("vqa_ranked", {}).get("by_family", {}).get(fam) for m in ms])
        row["vqa"] = mean_std([m["vqa_generated"]["accuracy"] for m in ms])
        for k in ("bleu1", "bleu4", "rougeL", "cider"):
            row[k] = mean_std([m.get("caption_overlap", {}).get(k) for m in ms])
        row["fact_f1"] = mean_std([m["caption_facts"]["f1"] for m in ms])
        out.append(row)
    return out


def table_per_class(runs: Path, names=("baseline", "blip-lora", "served-rgb-eval", "served-ndvi-eval")) -> dict:
    """Table 2: presence accuracy and yes-F1 per class (fold 0, plus the clean slice reported separately)."""
    res = {}
    for name in names:
        rs = [r for r in rows(runs / name / "f0.jsonl") if r["family"] == "presence"]
        if not rs:
            continue
        per = {}
        for c in CLASSES:
            cr = [r for r in rs if r["cls"] == c.key]
            yes_t = np.array([r["truth"] != "no" for r in cr])
            yes_p = np.array([r["pred"].strip().lower() not in ("no", "") for r in cr])
            tp, fp, fn = (yes_t & yes_p).sum(), (~yes_t & yes_p).sum(), (yes_t & ~yes_p).sum()
            per[c.key] = {"acc": float(credits(cr).mean()) if cr else None, "f1": float(2 * tp / max(1, 2 * tp + fp + fn)),
                          "n_present": int(yes_t.sum())}
        res[name] = per
    return res


def aligned(runs: Path, a: str, b: str, fold: int = 0):
    """Per-item credit of two models on the same fold-0 items, with the field of each item."""
    ka = {(r["row"], r["family"], r.get("cls")): r for r in rows(runs / a / f"f{fold}.jsonl") if r["family"] != "caption"}
    kb = {(r["row"], r["family"], r.get("cls")): r for r in rows(runs / b / f"f{fold}.jsonl") if r["family"] != "caption"}
    keys = sorted(ka.keys() & kb.keys(), key=str)
    ca = np.array([vqa_credit(item_of(ka[k]), ka[k]["pred"]) for k in keys])
    cb = np.array([vqa_credit(item_of(kb[k]), kb[k]["pred"]) for k in keys])
    return ca, cb, np.array([ka[k]["field"] for k in keys])


def table_fold0(runs: Path) -> list[dict]:
    """Table 3: fold-0 comparison with field-grouped 95% CIs and paired differences against BLIP-LoRA."""
    out = []
    for name, label in (("blip-lora", "BLIP-LoRA"), ("served-zeroshot-eval", "Served model, zero-shot"),
                        ("served-rgb-eval", "Served model, RGB"), ("served-ndvi-eval", "Served model, RGB + NDVI")):
        if not (runs / name / "f0.jsonl").exists():
            continue
        if name == "blip-lora":
            ca, _, g = aligned(runs, name, name)
            out.append({"model": label, "acc": grouped_bootstrap(ca, g), "diff": None})
            continue
        ca, cb, g = aligned(runs, name, "blip-lora")
        out.append({"model": label, "acc": grouped_bootstrap(ca, g), "diff": grouped_bootstrap(ca, g, other=cb)})
    if (runs / "served-ndvi-eval" / "f0.jsonl").exists() and (runs / "served-rgb-eval" / "f0.jsonl").exists():
        ca, cb, g = aligned(runs, "served-ndvi-eval", "served-rgb-eval")
        out.append({"model": "NDVI minus RGB", "acc": None, "diff": grouped_bootstrap(ca, g, other=cb)})
    return out


def table_hedging(runs: Path, names=("baseline", "blip-lora", "served-rgb-eval", "served-ndvi-eval", "dora")) -> list[dict]:
    """Table 4 and the calibration figures: hedge rates, partial credit, ECE, AURC (fold 0)."""
    out = []
    for name in names:
        rs = [r for r in rows(runs / name / "f0.jsonl") if r["family"] != "caption"]
        if not rs:
            continue
        amb = np.array([r["ambiguous"] for r in rs])
        hedged = np.array([is_hedge(r["pred"]) for r in rs])
        cr = credits(rs)
        conf = np.array([r["conf"] for r in rs], float)
        correct = (cr == 1.0).astype(float)
        cov, acc, aurc = risk_coverage(conf, correct)
        out.append({"model": name, "hedge_amb": float(hedged[amb].mean()), "hedge_clear": float(hedged[~amb].mean()),
                    "credit_amb": float(cr[amb].mean()), "acc_clear": float(cr[~amb].mean()), "ece": ece(conf, correct),
                    "aurc": aurc, "acc_at_80": float(acc[int(0.8 * len(acc)) - 1])})
    return out


def table_robustness(runs: Path) -> dict:
    """Table 5: accuracy drop from clean per variant (fold 0, 500-tile subset)."""
    res = {}
    for d in sorted(runs.glob("robust-*")):
        clean = d / "f0-clean.json"
        if not clean.exists():
            continue
        base = json.loads(clean.read_text())["vqa_generated"]["accuracy"]
        res[d.name] = {p.stem.removeprefix("f0-"): json.loads(p.read_text())["vqa_generated"]["accuracy"] - base
                       for p in sorted(d.glob("f0-*.json")) if p.stem != "f0-clean"}
    return res


def table_leakage(runs: Path) -> dict:
    """Table 8: one model trained on the official split, scored on seen-field vs unseen-field val tiles."""
    out = {}
    for part in ("seen-field", "unseen-field"):
        rs = rows(runs / "leakage" / f"{part}.jsonl")
        if rs:
            cr = credits(rs)
            out[part] = grouped_bootstrap(cr, np.array([r["field"] for r in rs if r["family"] != "caption"]))
    return out


def sweep_scores(runs: Path) -> dict:
    return {t: json.loads((runs / f"sweep-{t}" / "choice.json").read_text()) for t in ("vqa", "caption")
            if (runs / f"sweep-{t}" / "choice.json").exists()}


def latex_ablation(t1: list[dict]) -> str:
    head = "Model & Presence & Identify & Extent & Location & BLEU-4 & ROUGE-L & CIDEr & Fact F1 \\\\"
    body = []
    for r in t1:
        fam = [f"{fmt(*r[f])} ({fmt(r[f + '_ranked'][0])})" for f in FAMILIES]
        cap = [fmt(*r["bleu4"]), fmt(*r["rougeL"]), fmt(*r["cider"], pct=False), fmt(*r["fact_f1"])]
        body.append(" & ".join([r["model"], *fam, *cap]) + " \\\\")
    return "\\begin{tabular}{l" + "c" * 8 + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(body) + "\n\\bottomrule\n\\end{tabular}\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("report/tables"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    results = {"ablation": table_ablation(a.runs), "per_class": table_per_class(a.runs), "fold0": table_fold0(a.runs),
               "hedging": table_hedging(a.runs), "robustness": table_robustness(a.runs), "leakage": table_leakage(a.runs),
               "sweep": sweep_scores(a.runs)}
    (a.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (a.out / "ablation.tex").write_text(latex_ablation(results["ablation"]))
    print(json.dumps({k: (len(v) if hasattr(v, "__len__") else v) for k, v in results.items()}))


if __name__ == "__main__":
    main()
