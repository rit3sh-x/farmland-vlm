"""Every results table of issues/06-evaluation-protocol.md, rebuilt from the run records in runs/.

    python -m farmvlm.report --runs runs --out report/tables
"""
import argparse
import ast
import json
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from farmvlm.metrics import (PRIOR_WEIGHTS, ece, field_resample_means, grouped_bootstrap, is_hedge, margin_hedge, prior_pick,
                             prior_ratios, risk_coverage, vqa_credit)
from farmvlm.items import Item
from farmvlm.text import lenient_answer
from farmvlm.scheme import CLASSES, CLASS_INDEX, THROUGHOUT

FAMILIES = ("presence", "identify", "extent", "location")
# Held-out record of each training-length cycle (VQA adapters of that cycle, cycle 2's caption adapters) and its pairs.
CYCLES = (("blip-lora", 2, 120_000), ("c3-blip", 3, 240_000), ("c4-blip", 4, 480_000), ("c5-blip", 5, 960_000),
          ("c6-blip", 6, 1_920_000), ("c7-blip", 7, 3_840_000))
T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78}  # two-sided 95% t quantiles by degrees of freedom


def cycle_validation(runs: Path, cycle: int) -> float | None:
    """The selection score of a cycle: the mean over its five folds of each fold's best second-half validation check."""
    done = [runs / f"c{cycle}-vqa-f{k}" / "done.json" for k in range(5)]
    return float(np.mean([json.loads(p.read_text())["best_val"] for p in done])) if all(p.exists() for p in done) else None


def chosen_cycle(runs: Path) -> int:
    """The rule fixed on 2026-11-05 at 10:00, before any held-out score was read: of the cycles trained on all five
    folds, the one with the highest selection score. Held-out scores never enter."""
    val = {c: cycle_validation(runs, c) for _, c, _ in CYCLES[1:]}
    return max((c for c in val if val[c] is not None), key=val.get)


def final_run(runs: Path) -> str:
    """The final model's held-out record: the chosen cycle's VQA adapters with the 480k-pair caption adapters."""
    return f"c{chosen_cycle(runs)}-capc4-blip"


def fold0_rows(final: str) -> tuple:
    """(run, label, run its paired difference is taken against) for the fold-0 table."""
    return (("baseline", "Zero-shot BLIP", final), ("archive/v1/blip-lora", "BLIP-LoRA v1", final),
            ("blip-lora", "BLIP-LoRA v2", final), (final, "BLIP-LoRA", final),
            ("dora", "BLIP-DoRA v1", "archive/v1/blip-lora"),  # DoRA trained in the first cycle: paired with that LoRA
            ("balanced", "BLIP-LoRA v2, balanced location draws", "blip-lora"),  # an ablation of the v2 recipe
            ("served-zeroshot-eval", "Qwen3.5-2B, zero-shot", final),
            ("archive/v1/served-rgb-eval", "Qwen3.5-2B LoRA, RGB v1", final), ("served-rgb-eval", "Qwen3.5-2B LoRA, RGB", final),
            ("archive/v1/served-ndvi-eval", "Qwen3.5-2B LoRA, RGB + NDVI v1", final),
            ("served-ndvi-eval", "Qwen3.5-2B LoRA, RGB + NDVI", final))


def rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []


def item_of(r: dict) -> Item:
    return Item(r["row"], r["family"], CLASS_INDEX[r["cls"]] if r.get("cls") else -1, r["truth"], r["ambiguous"])


def credits(rs, key="pred"):
    return np.array([vqa_credit(item_of(r), r[key]) for r in rs if r["family"] != "caption" and r.get(key) is not None])



def best_val_rows(runs: Path, run: str) -> list[dict]:
    """The validation rows of the check a run's best adapter was saved at."""
    if not (runs / run / "done.json").exists():
        return []
    best = json.loads((runs / run / "done.json").read_text())["best_val"]
    return rows(runs / run / "val" / f"{[m['val'] for m in rows(runs / run / 'metrics.jsonl')].index(best):02d}.jsonl")


def family_means(rs: list[dict], picks: list[str]) -> np.ndarray:
    by = defaultdict(list)
    for r, p in zip(rs, picks):
        by[r["family"]].append(vqa_credit(item_of(r), p))
    return np.array([np.mean(by[f]) for f in FAMILIES])


def fit_prior(rs: list[dict]) -> tuple[dict, dict]:
    """Label-shift ratios from validation rows and, per family, the weight in PRIOR_WEIGHTS that scores best on them."""
    ratios = prior_ratios(rs)
    score = {w: family_means(rs, [prior_pick(r, ratios, w) for r in rs]) for w in PRIOR_WEIGHTS}
    return ratios, {f: max(PRIOR_WEIGHTS, key=lambda w: score[w][k]) for k, f in enumerate(FAMILIES)}


def prior_corrected(rs: list[dict], ratios: dict, weights: dict) -> list[str]:
    return [prior_pick(r, ratios, weights[r["family"]]) for r in rs]


def prior_validation(runs: Path, cycle: int) -> tuple[list, list] | None:
    """Per fold, the four-question validation mean of the ranked readout and of its label-shift correction, each
    field-grouped half of the fold's validation scored with ratios and weights fitted on the other half."""
    before, after = [], []
    for k in range(5):
        rs = best_val_rows(runs, f"c{cycle}-vqa-f{k}")
        if not rs:
            return None
        halves = [[r for r in rs if zlib.crc32(r["field"].encode()) % 2 == h] for h in (0, 1)]
        b, a = [], []
        for fit, test in ((0, 1), (1, 0)):
            ratios, w = fit_prior(halves[fit])
            b.append(family_means(halves[test], [r["ranked"] for r in halves[test]]).mean())
            a.append(family_means(halves[test], prior_corrected(halves[test], ratios, w)).mean())
        before.append(float(np.mean(b)))
        after.append(float(np.mean(a)))
    return before, after


def undercalled(runs: Path, cycle: int, family: str = "identify", answer: str = "nutrient deficiency") -> tuple[int, int]:
    """(times true, times the ranked readout gave it) for one answer over a cycle's five validation sets."""
    rs = [r for k in range(5) for r in best_val_rows(runs, f"c{cycle}-vqa-f{k}") if r["family"] == family]
    return sum(r["truth"] == answer for r in rs), sum(r["ranked"] == answer for r in rs)


def prior_heldout(runs: Path, cycle: int, name: str) -> dict | None:
    """Held-out four-question means per fold for the ranked readout and its label-shift correction, fitted on each
    fold's own validation rows, with the per-family change and the weights chosen."""
    out = {"before": [], "after": [], "after_by_family": [], "weights": []}
    for k in range(5):
        val = best_val_rows(runs, f"c{cycle}-vqa-f{k}")
        rs = [r for r in rows(runs / name / f"f{k}.jsonl") if r["family"] in FAMILIES]
        if not val or not rs:
            return None
        ratios, w = fit_prior(val)
        fb, fa = family_means(rs, [r["ranked"] for r in rs]), family_means(rs, prior_corrected(rs, ratios, w))
        out["before"].append(float(fb.mean()))
        out["after"].append(float(fa.mean()))
        out["after_by_family"].append(fa.tolist())
        out["weights"].append(w)
    return out

def mean_std(xs):
    xs = [x for x in xs if x is not None]
    return (float(np.mean(xs)), float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0) if xs else (None, None)


def fmt(m, s=None, pct=True):
    if m is None:
        return "--"
    k = 100 if pct else 1
    return f"{m * k:.1f}" + (f" $\\pm$ {s * k:.1f}" if s is not None else "")


def fold_means(runs: Path, name: str, readout: str = "vqa_ranked", key: str = "by_family") -> list[float]:
    """The four-question mean of each scored fold of a held-out record."""
    ms = [json.loads(p.read_text()) for p in sorted((runs / name).glob("f[0-4].json"))]
    return [float(np.mean([m[readout][key][f] for f in FAMILIES])) for m in ms]


def paired(after, before) -> tuple:
    """Mean fold-paired change with a 95% t interval over folds, and the number of folds that improved."""
    d = np.array(after) - np.array(before)
    h = T95[len(d) - 1] * d.std(ddof=1) / np.sqrt(len(d))
    return float(d.mean()), float(d.mean() - h), float(d.mean() + h), int((d > 0).sum())


def fact_f1s(runs: Path, name: str) -> list[float]:
    return [json.loads((runs / name / f"f{k}.json").read_text())["caption_facts"]["f1"] for k in range(5)
            if (runs / name / f"f{k}.json").exists()]


def table_cycles(runs: Path) -> list[dict]:
    """Held-out four-question mean per training-length cycle over five folds, with the change from the cycle before as
    a paired t interval over folds; the selection score beside it, and the mean greedy accuracy change over the image
    perturbations on fold 0's robustness tiles."""
    out, prev, rob = [], None, table_robustness(runs)
    for name, cycle, pairs in CYCLES:
        ranked = fold_means(runs, name)
        if len(ranked) < 5:
            continue
        row = {"cycle": cycle, "pairs": pairs, "val": cycle_validation(runs, cycle) if cycle > 2 else None,
               "ranked": mean_std(ranked), "greedy": mean_std(fold_means(runs, name, "vqa_generated")),
               "partial": mean_std(fold_means(runs, name, key="partial_by_family")), "change": None,
               "perturbed": float(np.mean([v for k, v in rob[f"robust-c{cycle}"].items() if k not in ("clean", "paraphrase")]))
               if f"robust-c{cycle}" in rob else None}
        if prev:
            row["change"] = paired(ranked, prev)
        out.append(row)
        prev = ranked
    return out


def table_ablation(runs: Path, final: str) -> list[dict]:
    """Table 1: Baseline vs BLIP-LoRA over 5 folds, with the first and second cycles' rows above the final one."""
    out = []
    for name, label in (("baseline", "Zero-shot BLIP"), ("archive/v1/blip-lora", "BLIP-LoRA v1"), ("blip-lora", "BLIP-LoRA v2"),
                        (final, "BLIP-LoRA")):
        ms = [json.loads(p.read_text()) for p in sorted((runs / name).glob("f[0-4].json"))]
        if not ms:
            continue
        row = {"model": label, "folds": len(ms)}
        for fam in FAMILIES:
            row[fam] = mean_std([m["vqa_generated"]["by_family"].get(fam) for m in ms])
            row[fam + "_ranked"] = mean_std([m.get("vqa_ranked", {}).get("by_family", {}).get(fam) for m in ms])
        row["vqa"] = mean_std([m["vqa_generated"]["accuracy"] for m in ms])
        row["mean"] = mean_std([np.mean([m["vqa_generated"]["by_family"][f] for f in FAMILIES]) for m in ms])
        row["mean_ranked"] = mean_std([np.mean([m["vqa_ranked"]["by_family"][f] for f in FAMILIES]) for m in ms if m.get("vqa_ranked")])
        for k in ("bleu1", "bleu4", "rougeL", "cider"):
            row[k] = mean_std([m.get("caption_overlap", {}).get(k) for m in ms])
        row["fact_f1"] = mean_std([m["caption_facts"]["f1"] for m in ms])
        out.append(row)
    return out


def table_constant(runs: Path) -> dict:
    """The best single answer per question type, picked with hindsight on each fold's own test items: the floor to clear."""
    folds = []
    src = next((runs / n for n in ("blip-lora", "baseline") if (runs / n / "f0.jsonl").exists()), runs / "blip-lora")  # same items either way
    for path in sorted(src.glob("f[0-4].jsonl")):
        rs = [r for r in rows(path) if r["family"] != "caption"]
        fam = {}
        for f in FAMILIES:
            its = [item_of(r) for r in rs if r["family"] == f]
            fam[f] = max(np.mean([vqa_credit(i, a) for i in its]) for a in {i.answer for i in its})
        folds.append(fam)
    return {f: mean_std([d[f] for d in folds]) for f in FAMILIES} | {"mean": mean_std([np.mean(list(d.values())) for d in folds])}


def table_per_class(runs: Path, names: tuple) -> dict:
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


def table_fold0(runs: Path, final: str) -> list[dict]:
    """Table 3: fold-0 comparison with field-grouped 95% CIs and paired differences against BLIP-LoRA."""
    out = []
    for name, label, ref in fold0_rows(final):
        if not (runs / name / "f0.jsonl").exists():
            continue
        ca, cb, g = aligned(runs, name, ref)
        paired = name != ref and len(ca) > 0
        if not len(ca):  # the reference is not scored yet: pooled accuracy over the model's own items, no paired difference
            ca, cb, g = aligned(runs, name, name)
        m = json.loads((runs / name / "f0.json").read_text())
        all_rows = rows(runs / name / "f0.jsonl")
        rs = [x for x in all_rows if x["family"] != "caption"]
        caps = [x for x in all_rows if x["family"] == "caption"]
        lenient = {f: float(np.mean([vqa_credit(item_of(x), lenient_answer(f, x["pred"])) for x in rs if x["family"] == f]))
                   for f in FAMILIES}
        out.append({"model": label, "run": name, "acc": grouped_bootstrap(ca, g), "diff": grouped_bootstrap(ca, g, other=cb) if paired else None,
                    "by_family": m["vqa_generated"]["by_family"], "by_family_lenient": lenient,
                    "caption": {k: m["caption_facts"][k] for k in ("precision", "recall", "f1")},
                    "caption_words": float(np.mean([len(str(x["pred"]).split()) for x in caps])) if caps else None,
                    "reference_words": float(np.mean([len(first_ref(x).split()) for x in caps])) if caps else None})
    if (runs / "served-ndvi-eval" / "f0.jsonl").exists() and (runs / "served-rgb-eval" / "f0.jsonl").exists():
        ca, cb, g = aligned(runs, "served-ndvi-eval", "served-rgb-eval")
        out.append({"model": "NDVI minus RGB", "acc": None, "diff": grouped_bootstrap(ca, g, other=cb)})
    return out


def first_ref(r: dict) -> str:
    """The first reference caption of a scored caption row (records store the list as text)."""
    refs = r.get("refs") or []
    if isinstance(refs, str):
        refs = ast.literal_eval(refs)
    return refs[0] if refs else ""


def table_caption_examples(runs: Path, final_name: str, fold: int = 0) -> list[dict]:
    """Three fold-0 tiles (clean, one finding, several): the reference caption, the first cycle's caption and the final one."""
    v1 = {r["row"]: r for r in rows(runs / "archive" / "v1" / "blip-lora" / f"f{fold}.jsonl") if r["family"] == "caption"}
    final = {r["row"]: r for r in rows(runs / final_name / f"f{fold}.jsonl") if r["family"] == "caption"}
    src = final or v1
    clean = lambda ref: ref.lower().startswith("no anomalies")
    kinds = (("clean", clean), ("one finding", lambda ref: not clean(ref) and "," not in ref), ("several findings", lambda ref: "," in ref))
    out = []
    for kind, ok in kinds:
        row = next((k for k in sorted(src) if ok(first_ref(src[k]))), None)  # the lowest row id of its kind: no cherry-picking
        if row is not None:
            out.append({"kind": kind, "row": row, "reference": first_ref(src[row]).rstrip("."),
                        "v1": v1.get(row, {}).get("pred"), "final": final.get(row, {}).get("pred")})
    return out


def latex_captions(t: list[dict]) -> str:
    """The caption examples: one row per tile, reference beside the first cycle's and the final caption."""
    esc = lambda x: "\\textbf{??}" if x is None else x.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
    body = [" & ".join([r["kind"].capitalize(), esc(r["reference"]), esc(r["v1"]), esc(r["final"])]) for r in t]
    return tabular("l" + "p{0.26\\textwidth}" * 3, "Tile & Reference & First cycle & Final", body)


def table_location_split(runs: Path, names: tuple) -> dict:
    """Location accuracy on fold-0 items whose truth is "throughout" and on the localised ones, per model."""
    out = {}
    for name in names:
        rs = [r for r in rows(runs / name / "f0.jsonl") if r["family"] == "location" and r.get("pred") is not None]
        if rs:
            c, thr = credits(rs), np.array([r["truth"] == THROUGHOUT for r in rs])
            out[name] = {"throughout": float(c[thr].mean()), "localised": float(c[~thr].mean()),
                         "n_throughout": int(thr.sum()), "n_localised": int((~thr).sum())}
    return out


def table_hedging(runs: Path, names: tuple) -> list[dict]:
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
                    "aurc": aurc, "acc_at_80": float(acc[int(0.8 * len(acc)) - 1]), "acc_all": float(acc[-1])})
    return out


def table_fit(runs: Path) -> list[dict]:
    """Fold 0 per cycle: the ranked four-question mean on 2,000 of the adapter's own training tiles (drawn by the held-out
    rules) against its held-out fold-0 score, with clear accuracy and ambiguous credit on the training tiles."""
    out = []
    for name, cycle, pairs in CYCLES:
        tr, ho = runs / f"train-c{cycle}" / "train-f0.json", runs / name / "f0.json"
        if not (tr.exists() and ho.exists()):
            continue
        t, h = (json.loads(p.read_text())["vqa_ranked"] for p in (tr, ho))
        mean = lambda m: float(np.mean([m["by_family"][f] for f in FAMILIES]))
        out.append({"cycle": cycle, "pairs": pairs, "train": mean(t), "held": mean(h), "train_clear": t["clear_accuracy"],
                    "train_ambiguous": t["ambiguous_credit"], "train_by_family": t["by_family"]})
    return out


def table_robustness(runs: Path) -> dict:
    """Table 5: accuracy drop from clean per variant (fold 0, 500-tile subset), and the clean accuracy under "clean"."""
    res = {}
    for d in sorted(runs.glob("robust-*")) + sorted((runs / "archive" / "v1").glob("robust-*")):
        clean = d / "f0-clean.json"
        if not clean.exists():
            continue
        base = json.loads(clean.read_text())["vqa_generated"]["accuracy"]
        res[str(d.relative_to(runs))] = {"clean": base} | {
            p.stem.removeprefix("f0-"): json.loads(p.read_text())["vqa_generated"]["accuracy"] - base
            for p in sorted(d.glob("f0-*.json")) if p.stem != "f0-clean"}
    return res


def table_augment(runs: Path) -> dict:
    """Augmented against plain training at one cycle's fold-0 recipe: best validation of each, and the paired held-out
    change in greedy item credit (augmented minus plain, field bootstrap)."""
    out = {}
    for c in (4, 6):
        done = [runs / f"{r}-vqa-f0" / "done.json" for r in (f"c{c}", f"aug-c{c}")]
        if all(d.exists() for d in done) and (runs / f"aug-c{c}-blip" / "f0.jsonl").exists():
            ca, cb, g = aligned(runs, f"aug-c{c}-blip", f"c{c}-blip")
            out[c] = {"val": [json.loads(d.read_text())["best_val"] for d in done], "held_diff": grouped_bootstrap(ca, g, other=cb)}
    return out


def robustness_shared(runs: Path) -> dict:
    """Accuracy change from clean for the chosen cycle's BLIP and both Qwen variants, on the 200 tiles Qwen was scored on."""
    dirs = {"blip": runs / f"robust-c{chosen_cycle(runs)}", "servedRgb": runs / "robust-served-rgb",
            "servedNdvi": runs / "robust-served-ndvi"}
    if not all((d / "f0-clean.jsonl").exists() for d in dirs.values()):
        return {}
    subset = {r["row"] for r in rows(dirs["servedRgb"] / "f0-clean.jsonl")}
    variants = [p.stem.removeprefix("f0-") for p in sorted(dirs["servedRgb"].glob("f0-*.jsonl")) if p.stem != "f0-clean"]
    acc = lambda d, v: float(credits([r for r in rows(d / f"f0-{v}.jsonl") if r["row"] in subset]).mean())
    return {k: {v: acc(d, v) - acc(d, "clean") for v in variants} for k, d in dirs.items()}


def table_leakage(runs: Path) -> dict:
    """Table 8: one model trained on the official split, scored on seen-field vs unseen-field val tiles."""
    out, draws = {}, {}
    for seed, part in enumerate(("seen-field", "unseen-field")):
        rs = [r for r in rows(runs / "leakage" / f"{part}.jsonl") if r["family"] != "caption"]
        if rs:
            cr, fields = credits(rs), np.array([r["field"] for r in rs])
            out[part] = grouped_bootstrap(cr, fields)
            draws[part] = field_resample_means(cr, fields, seed=seed)
    if len(draws) == 2:  # the halves hold different fields, so their resamples are independent
        gap = draws["seen-field"] - draws["unseen-field"]
        out["gap"] = (out["seen-field"][0] - out["unseen-field"][0], float(np.percentile(gap, 2.5)), float(np.percentile(gap, 97.5)))
    return out


def table_hedge_rule(runs: Path, name: str, deltas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4)) -> list[dict]:
    """Table 4b: hedging from the ranked readout's likelihood margin, swept over the margin (fold 0)."""
    rs = [r for r in rows(runs / name / "f0.jsonl") if r["family"] != "caption" and r.get("ranked2") is not None]
    if not rs:
        return []
    items = [item_of(r) for r in rs]
    amb = np.array([i.ambiguous for i in items])
    out = []
    for d in deltas:
        preds = [margin_hedge(r["family"], r["ranked"], r["ranked2"], r["ranked_conf"], r["ranked2_conf"], d) for r in rs]
        cr = np.array([vqa_credit(i, p) for i, p in zip(items, preds)])
        hedged = np.array([is_hedge(p) for p in preds])
        out.append({"delta": d, "hedge_amb": float(hedged[amb].mean()), "hedge_clear": float(hedged[~amb].mean()),
                    "credit_amb": float(cr[amb].mean()), "acc_clear": float(cr[~amb].mean()), "accuracy": float(cr.mean())})
    return out


def gpu_minutes(run: Path) -> float:
    """Training plus validation minutes of a run from its metrics log; the clock restarts at every resume."""
    total, prev = 0.0, 0.0
    for m in (x["minutes"] for x in rows(run / "metrics.jsonl")):
        total += m if m < prev else m - prev
        prev = m
    return total


COST_GROUPS = (("BLIP VQA LoRA, 5 folds", ["blip-vqa-f[0-4]"]), ("BLIP caption LoRA, 5 folds", ["blip-caption-f[0-4]"]),
               ("BLIP sweeps, 20 trials", ["sweep-vqa/*", "sweep-caption/*"]), ("Leakage model (official split)", ["leak-vqa-official"]),
               ("BLIP VQA DoRA, fold 0", ["blip-vqa-f0-dora"]), ("Served bake-off, 4 runs", ["bakeoff/*"]),
               ("Qwen3.5-2B LoRA, RGB", ["served-rgb", "served-rgb-c*"]), ("Qwen3.5-2B LoRA, RGB + NDVI", ["served-ndvi", "served-ndvi-c*"]),
               ("BLIP study, fold 0", ["study-vqa/*", "study-caption/*"]),
               *((f"BLIP VQA cycle {c}, {p // 1000:,}k pairs", [f"c{c}-vqa-f[0-4]"]) for _, c, p in CYCLES[1:]),
               ("BLIP caption LoRA, 480k pairs", ["c4-caption-f[0-4]"]))


def gpu_name(run: Path) -> str:
    """The GPU a run trained on, as its config or done record names it; runs from before either recorded it trained on
    free Colab and Kaggle T4s."""
    name = " ".join(json.loads((run / f).read_text()).get("gpu") or "" for f in ("config.json", "done.json") if (run / f).exists())
    return "RTX PRO 5000" if "PRO 5000" in name else "T4"


def table_cost(runs: Path) -> list[dict]:
    """Table 7: trainable parameters and GPU-hours per group of training runs, from their configs and logs."""
    out = []
    for label, patterns in COST_GROUPS:
        dirs = [d for p in patterns for d in sorted(runs.glob(p)) if (d / "metrics.jsonl").exists()]
        if not dirs:
            continue
        trainable = {json.loads((d / "config.json").read_text()).get("trainable") for d in dirs if (d / "config.json").exists()}
        gpus = {gpu_name(d) for d in dirs}
        out.append({"group": label, "runs": len(dirs), "trainable": trainable.pop() if len(trainable) == 1 else None,
                    "gpu": " / ".join(sorted(gpus)), "gpu_hours": sum(gpu_minutes(d) for d in dirs) / 60})
    return out


def hours_by_gpu(runs: Path) -> dict:
    """Training and validation GPU-hours of every costed run, per GPU model."""
    dirs = {d for _, patterns in COST_GROUPS for p in patterns for d in runs.glob(p) if (d / "metrics.jsonl").exists()}
    out = defaultdict(float)
    for d in dirs:
        out[gpu_name(d)] += gpu_minutes(d) / 60
    return dict(out)


def latex_cost(t7: list[dict]) -> str:
    body = [" & ".join([r["group"], str(r["runs"]), f"{r['trainable'] / 1e6:.1f}" if r["trainable"] else "--", r["gpu"],
                        f"{r['gpu_hours']:.1f}"]) for r in t7]
    return tabular("lrrlr", "Runs & Count & Trainable params (M) & GPU & GPU-hours", body)


def latex_cycles(t: list[dict]) -> str:
    """The length table: per cycle, the selection score, held-out ranked and greedy four-question means, partial credit
    and the paired change from the cycle before."""
    body = []
    for r in t:
        ch = r["change"]
        change = "--" if ch is None else f"{100 * ch[0]:+.1f} [{100 * ch[1]:+.1f}, {100 * ch[2]:+.1f}], {ch[3]}/5"
        body.append(" & ".join([str(r["cycle"]), f"{r['pairs'] // 1000:,}k", fmt(r["val"]) if r["val"] else "--",
                                fmt(*r["ranked"]), change, fmt(*r["greedy"]), fmt(*r["partial"]),
                                "--" if r["perturbed"] is None else f"${100 * r['perturbed']:+.1f}$"]))
    head = ("Cycle & Pairs & Validation & Held out, ranked & Change [95\\% CI], folds up & Held out, greedy & Partial credit"
            " & Perturbed images")
    return tabular("crccccc" + "c", head, body)


def table_pointing(runs: Path, cycle: int) -> dict:
    """The pointing game: baseline vs the chosen cycle's fold-0 VQA adapter, hit rate and heat-in-mask energy, with chance."""
    p = runs / f"heatmaps-c{cycle}" / "pointing.json"
    return json.loads(p.read_text())["summary"] if p.exists() else {}


FOLD0_MACROS = (("Qwen3.5-2B LoRA, RGB", "qwenRgb"), ("Qwen3.5-2B LoRA, RGB + NDVI", "qwenNdvi"), ("BLIP-LoRA", "blipFold"),
                ("BLIP-LoRA v1", "voneFold"), ("BLIP-LoRA v2", "vtwoFold"), ("BLIP-LoRA v2, balanced location draws", "balanced"),
                ("BLIP-DoRA v1", "dora"))
ROBUST_DROPS = {"paraphrase": "Paraphrase", "jpeg-10": "Jpeg", "blur-4": "Blur", "noise-30": "Noise", "resolution-6": "Res",
                "light-0.4": "Light", "hflip-None": "Flip", "rot90-None": "Rot"}
ROBUST_MID = {"jpeg-20": "Jpeg", "blur-2": "Blur", "noise-15": "Noise", "resolution-3": "Res", "light-0.25": "Light"}

# Every macro the report and the deck may quote: one without a record yet prints ?? instead of breaking the build.
# Five-fold "Mean" macros are the ranked readout, chosen on validation; "Greedy" ones are the generated answers.
MACROS = tuple("""blipMean blipMeanStd blipGreedy blipGreedyStd blipPartial blipFactF blipFactFStd blipPresence blipIdentify
    blipExtent blipLocation blipRanked zeroMean zeroGreedy zeroPresence zeroIdentify voneMean voneMeanStd voneGreedy voneFactF
    vonePresence voneIdentify voneExtent voneLocation vtwoMean vtwoMeanStd vtwoGreedy vtwoFactF
    tfourHours proHours finalCycle finalPairs finalVal prevVal finalChange finalChangeLo finalChangeHi finalChangeFolds totalChange totalChangeLo totalChangeHi
    constMean constLocation constIdentify blipAccEighty blipAurc blipCreditAmb blipAccClear blipEce blipAccAll zeroAurc
    leakGap leakLo leakHi leakSeen leakUnseen ndviDiff ndviLo ndviHi qwenZeroLenient pointZero pointLora pointChance energyZero energyLora
    voneLocThroughout voneLocLocalised vtwoLocThroughout vtwoLocLocalised blipLocThroughout blipLocLocalised balLocThroughout
    balLocLocalised locNThroughout locNLocalised balLocDrop referenceWords blipFoldWords voneFoldWords""".split()) + tuple(
    name + s for _, name in FOLD0_MACROS for s in ("Mean", "Pooled", "Presence", "Location", "FactF", "Precision", "Recall", "Diff")
    if name != "blipFold" or s != "Diff") + tuple(
    m + s for m in ("blip", "vtwo", "vone", "cFour", "augFour") for s in [d + "Drop" for d in ROBUST_DROPS.values()] + ["JpegAcc", "RobustClean", "Perturbed"]) + (
    "augFourVal", "augFourPlainVal", "augFourHeld", "augFourHeldLo", "augFourHeldHi",
    "capChange", "capChangeLo", "capChangeHi", "capChangeFolds", "blipCorrMean", "blipCorrMeanStd", "corrChange", "corrChangeLo",
    "corrChangeHi", "corrChangeFolds", "corrIdentify", "corrValChange", "corrValLo", "corrValHi", "corrValFolds", "corrNdTrue", "corrNdSaid", "fitTrain", "fitHeld", "fitGap", "fitClear", "fitAmbig", "fitExtent", "fitLocation",
    "fitPresence", "fitTwoTrain", "fitTwoHeld", "fitTwoGap",
    "servedRgbParaphraseDrop", "servedNdviParaphraseDrop") + tuple(
    m + "Mid" + s for m in ("blip", "servedRgb", "servedNdvi") for s in ROBUST_MID.values())


def latex_numbers(res: dict) -> str:
    """\\newcommand macros for the headline numbers, so the deck and the report quote the current run records."""
    def pct(x):
        v = f"{100 * x:.1f}"
        return f"${v}$" if v.startswith("-") else v
    abl = {r["model"]: r for r in res["ablation"]}
    f0 = {r["model"]: r for r in res["fold0"]}
    hed = {r["model"]: r for r in res["hedging"]}
    n = {}
    for label, k in (("BLIP-LoRA", "blip"), ("Zero-shot BLIP", "zero"), ("BLIP-LoRA v1", "vone"), ("BLIP-LoRA v2", "vtwo")):
        if label in abl:
            b = abl[label]
            n.update({k + "Mean": pct(b["mean_ranked"][0]), k + "MeanStd": pct(b["mean_ranked"][1]), k + "Greedy": pct(b["mean"][0]),
                      k + "GreedyStd": pct(b["mean"][1]), k + "FactF": pct(b["fact_f1"][0]), k + "FactFStd": pct(b["fact_f1"][1]),
                      **{k + f.capitalize(): pct(b[f + "_ranked"][0]) for f in FAMILIES}})
    if "BLIP-LoRA" in abl:
        n["blipRanked"] = n["blipMean"]
    cyc = {r["cycle"]: r for r in res["cycles"]}
    if res["chosen"] in cyc:
        r, first = cyc[res["chosen"]], res["cycles"][0]
        n.update(finalCycle=str(r["cycle"]), finalPairs=f"{r['pairs'] // 1000:,}k", finalVal=pct(r["val"]), blipPartial=pct(r["partial"][0]))
        if r["change"]:
            n.update(finalChange=pct(r["change"][0]), finalChangeLo=pct(r["change"][1]), finalChangeHi=pct(r["change"][2]),
                     finalChangeFolds=str(r["change"][3]))
            prev = cyc.get(r["cycle"] - 1)
            if prev and prev["val"]:
                n["prevVal"] = pct(prev["val"])
        t = paired(res["cycle_folds"][r["cycle"]], res["cycle_folds"][first["cycle"]])
        n.update(totalChange=pct(t[0]), totalChangeLo=pct(t[1]), totalChangeHi=pct(t[2]))
    if res.get("caption_change"):
        c = res["caption_change"]
        n.update(capChange=pct(c[0]), capChangeLo=pct(c[1]), capChangeHi=pct(c[2]), capChangeFolds=str(c[3]))
    if res.get("prior"):
        p = res["prior"]
        c = paired(p["after"], p["before"])
        n.update(blipCorrMean=pct(np.mean(p["after"])), blipCorrMeanStd=pct(np.std(p["after"], ddof=1)), corrChange=pct(c[0]),
                 corrChangeLo=pct(c[1]), corrChangeHi=pct(c[2]), corrChangeFolds=str(c[3]),
                 corrIdentify=pct(np.mean([f[FAMILIES.index("identify")] for f in p["after_by_family"]])))
    fit = {r["cycle"]: r for r in res.get("fit", [])}
    if res["chosen"] in fit:
        r = fit[res["chosen"]]
        n.update(fitTrain=pct(r["train"]), fitHeld=pct(r["held"]), fitGap=pct(r["train"] - r["held"]), fitClear=pct(r["train_clear"]),
                 fitAmbig=pct(r["train_ambiguous"]), **{"fit" + f.capitalize(): pct(r["train_by_family"][f])
                                                       for f in ("presence", "extent", "location")})
    if 4 in res.get("augment", {}):
        r = res["augment"][4]
        n.update(augFourVal=pct(r["val"][1]), augFourPlainVal=pct(r["val"][0]), augFourHeld=pct(r["held_diff"][0]),
                 augFourHeldLo=pct(r["held_diff"][1]), augFourHeldHi=pct(r["held_diff"][2]))
    if 2 in fit:
        n.update(fitTwoTrain=pct(fit[2]["train"]), fitTwoHeld=pct(fit[2]["held"]), fitTwoGap=pct(fit[2]["train"] - fit[2]["held"]))
    if res.get("undercalled") and res["undercalled"][0]:
        n.update(corrNdTrue=f"{res['undercalled'][0]:,}", corrNdSaid=f"{res['undercalled'][1]:,}")
    if res.get("prior_validation"):
        c = paired(res["prior_validation"][1], res["prior_validation"][0])
        n.update(corrValChange=pct(c[0]), corrValLo=pct(c[1]), corrValHi=pct(c[2]), corrValFolds=str(c[3]))
    n.update(tfourHours=f"{res['hours'].get('T4', 0):.0f}", proHours=f"{res['hours'].get('RTX PRO 5000', 0):.0f}")
    if res.get("constant"):
        c = res["constant"]
        n.update(constMean=pct(c["mean"][0]), constLocation=pct(c["location"][0]), constIdentify=pct(c["identify"][0]))
    if res["final"] in hed:
        h = hed[res["final"]]
        n.update(blipAccEighty=pct(h["acc_at_80"]), blipAurc=f"{h['aurc']:.2f}", blipCreditAmb=pct(h["credit_amb"]),
                 blipAccClear=pct(h["acc_clear"]), blipEce=f"{h['ece']:.2f}", blipAccAll=pct(h["acc_all"]))
    if "baseline" in hed:
        n["zeroAurc"] = f"{hed['baseline']['aurc']:.2f}"
    if "gap" in res.get("leakage", {}):
        g = res["leakage"]["gap"]
        n.update(leakGap=pct(g[0]), leakLo=pct(g[1]), leakHi=pct(g[2]), leakSeen=pct(res["leakage"]["seen-field"][0]),
                 leakUnseen=pct(res["leakage"]["unseen-field"][0]))
    if "NDVI minus RGB" in f0:
        d = f0["NDVI minus RGB"]["diff"]
        n.update(ndviDiff=pct(d[0]), ndviLo=pct(d[1]), ndviHi=pct(d[2]))
    for key, name in FOLD0_MACROS:
        if key in f0:
            r, fam = f0[key], f0[key]["by_family"]
            n[name + "Mean"] = pct(np.mean([fam[f] for f in FAMILIES]))
            n[name + "Pooled"] = pct(r["acc"][0])
            n[name + "Presence"] = pct(fam["presence"])
            n[name + "Location"] = pct(fam["location"])
            n[name + "FactF"] = pct(r["caption"]["f1"])
            n[name + "Precision"] = pct(r["caption"]["precision"])
            n[name + "Recall"] = pct(r["caption"]["recall"])
            if r["diff"]:
                n[name + "Diff"] = pct(r["diff"][0])
            if r.get("caption_words"):
                n[name + "Words"] = f"{r['caption_words']:.0f}"
                n["referenceWords"] = f"{r['reference_words']:.0f}"
    for key, name in ((f"robust-c{res['chosen']}", "blip"), ("robust-c2", "vtwo"), ("archive/v1/robust-blip-lora", "vone"),
                      ("robust-served-rgb", "servedRgb"), ("robust-served-ndvi", "servedNdvi"),
                      ("robust-c4", "cFour"), ("robust-aug-c4", "augFour")):
        r = res["robustness"].get(key, {})
        n.update({name + s + "Drop": pct(-r[v]) for v, s in ROBUST_DROPS.items() if v in r})
        if "jpeg-10" in r:
            n.update({name + "JpegAcc": pct(r["clean"] + r["jpeg-10"]), name + "RobustClean": pct(r["clean"]),
                      name + "Perturbed": pct(-np.mean([v for k, v in r.items() if k not in ("clean", "paraphrase")]))})
    for name, r in res.get("robustness_shared", {}).items():
        n.update({name + "Mid" + s: pct(-r[v]) for v, s in ROBUST_MID.items() if v in r})
    for key, name in (("archive/v1/blip-lora", "voneLoc"), ("blip-lora", "vtwoLoc"), (res["final"], "blipLoc"), ("balanced", "balLoc")):
        s = res.get("location_split", {}).get(key)
        if s:
            n.update({name + "Throughout": pct(s["throughout"]), name + "Localised": pct(s["localised"]),
                      "locNThroughout": f"{s['n_throughout']:,}", "locNLocalised": f"{s['n_localised']:,}"})
    if "BLIP-LoRA v2" in f0 and "BLIP-LoRA v2, balanced location draws" in f0:
        n["balLocDrop"] = pct(f0["BLIP-LoRA v2"]["by_family"]["location"] - f0["BLIP-LoRA v2, balanced location draws"]["by_family"]["location"])
    if "Qwen3.5-2B, zero-shot" in f0:
        n["qwenZeroLenient"] = pct(np.mean([f0["Qwen3.5-2B, zero-shot"]["by_family_lenient"][f] for f in FAMILIES]))
    if res.get("pointing"):
        p = res["pointing"]
        n.update(pointZero=pct(p["baseline"]["hit_rate"]), pointLora=pct(p["blip-lora"]["hit_rate"]),
                 pointChance=pct(p["blip-lora"]["chance"]), energyZero=pct(p["baseline"]["energy"]), energyLora=pct(p["blip-lora"]["energy"]))
    missing = [k for k in MACROS if k not in n]
    if missing:
        print("numbers.tex: no record yet for", " ".join(missing))
    return ("".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in n.items())
            + "".join(f"\\providecommand{{\\{k}}}{{\\textbf{{??}}}}\n" for k in missing))


def sweep_scores(runs: Path) -> dict:
    return {t: json.loads((runs / f"sweep-{t}" / "choice.json").read_text()) for t in ("vqa", "caption")
            if (runs / f"sweep-{t}" / "choice.json").exists()}


def tabular(cols: str, head: str, body: list[str]) -> str:
    return f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{head} \\\\\n\\midrule\n" + "\n".join(b + " \\\\" for b in body) + "\n\\bottomrule\n\\end{tabular}\n"


def ci(t, pct=True) -> str:
    k = 100 if pct else 1
    return "--" if t is None else f"{t[0] * k:.1f} [{t[1] * k:.1f}, {t[2] * k:.1f}]"


def latex_ablation(t1: list[dict], const: dict) -> str:
    """Table 1: Baseline, the constant-answer floor and BLIP-LoRA over 5 folds."""
    body = []
    for r in t1:
        fam = [f"{fmt(*r[f + '_ranked'])} ({fmt(r[f][0])})" for f in FAMILIES] + [f"{fmt(*r['mean_ranked'])} ({fmt(r['mean'][0])})"]
        cider = f"{r['cider'][0]:.2f}"
        body.append(" & ".join([r["model"], *fam, fmt(*r["bleu4"]), fmt(*r["rougeL"]), cider, fmt(*r["fact_f1"])]))
        if r["model"] == "Zero-shot BLIP" and const:
            body.append(" & ".join(["Most common answer", *[fmt(*const[f]) for f in FAMILIES], fmt(*const["mean"]), "--", "--", "--", "--"]))
    head = "Model & Presence & Identify & Extent & Location & Mean & BLEU-4 & ROUGE-L & CIDEr & Fact F1"
    return tabular("l" + "c" * 9, head, body)


def latex_fold0(t3: list[dict]) -> str:
    """Table 3: every model on the same fold-0 items, field-grouped 95% CIs, paired differences against BLIP-LoRA."""
    body = []
    for r in t3:
        if r["acc"] is None:
            body.append(f"\\midrule\n{r['model']} & \\multicolumn{{6}}{{c}}{{}} & {ci(r['diff'])} &")
            continue
        fam = [f"{100 * r['by_family'][f]:.1f}" for f in FAMILIES]
        mean = f"{100 * np.mean([r['by_family'][f] for f in FAMILIES]):.1f}"
        lenient = 100 * np.mean([r["by_family_lenient"][f] for f in FAMILIES])
        if lenient - float(mean) >= 0.5:  # a model that answers in sentences: show what its words amount to
            mean += f" ({lenient:.1f})"
        c = r["caption"]
        body.append(" & ".join([r["model"], *fam, mean, ci(r["acc"]), ci(r["diff"]), f"{100 * c['f1']:.1f} ({100 * c['precision']:.0f}/{100 * c['recall']:.0f})"]))
    head = "Model & Presence & Identify & Extent & Location & Mean & Pooled acc. [95\\% CI] & $\\Delta$ vs BLIP-LoRA & Fact F1 (P/R)"
    return tabular("l" + "c" * 8, head, body)


def latex_hedging(t4: list[dict], final: str) -> str:
    labels = {name: label for name, label, _ in fold0_rows(final)}
    body = [" & ".join([labels.get(r["model"], r["model"]), f"{100 * r['hedge_amb']:.1f}", f"{100 * r['hedge_clear']:.1f}",
                        f"{100 * r['credit_amb']:.1f}", f"{100 * r['acc_clear']:.1f}", f"{r['ece']:.3f}", f"{r['aurc']:.3f}",
                        f"{100 * r['acc_at_80']:.1f}"]) for r in t4]
    head = "Model & Hedge, amb. & Hedge, clear & Credit, amb. & Acc., clear & ECE & AURC & Acc. at 80\\% cov."
    return tabular("l" + "c" * 7, head, body)


def latex_per_class(t2: dict, final: str) -> str:
    names = [n for n in (final, "served-rgb-eval", "served-ndvi-eval") if n in t2]
    labels = {final: "BLIP-LoRA", "served-rgb-eval": "Qwen, RGB", "served-ndvi-eval": "Qwen, RGB + NDVI"}
    if not names:
        return tabular("l", "Class", ["\\textbf{??} no scored record yet"])
    body = []
    for c in CLASSES:
        n = t2[names[0]][c.key]["n_present"]
        mark = "$^\\dagger$" if n < 60 else ""
        body.append(" & ".join([c.name + mark, str(n), *[f"{100 * t2[m][c.key]['f1']:.1f}" for m in names]]))
    head = "Class & Present & " + " & ".join(labels[m] for m in names)
    return tabular("lr" + "c" * len(names), head, body)


def latex_leakage(t8: dict) -> str:
    if "gap" not in t8:
        return "\\emph{Pending: the official-split run is still training.}\n"
    body = [f"Seen field (in official train) & {ci(t8['seen-field'])}", f"Unseen field & {ci(t8['unseen-field'])}",
            f"Seen minus unseen & {ci(t8['gap'])}"]
    return tabular("lc", "Official val tiles & Pooled accuracy [95\\% CI]", body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("report/tables"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    chosen = chosen_cycle(a.runs)
    final = final_run(a.runs)
    served = ("served-rgb-eval", "served-ndvi-eval")
    results = {"chosen": chosen, "final": final, "cycles": table_cycles(a.runs),
               "cycle_folds": {c: fold_means(a.runs, name) for name, c, _ in CYCLES},
               "ablation": table_ablation(a.runs, final), "per_class": table_per_class(a.runs, ("baseline", final, *served)),
               "fold0": table_fold0(a.runs, final), "hedging": table_hedging(a.runs, ("baseline", final, *served, "dora")),
               "hedge_rule": table_hedge_rule(a.runs, final), "robustness": table_robustness(a.runs),
               "robustness_shared": robustness_shared(a.runs),
               "caption_change": paired(fact_f1s(a.runs, final), fact_f1s(a.runs, f"c{chosen}-blip"))
               if len(fact_f1s(a.runs, final)) == len(fact_f1s(a.runs, f"c{chosen}-blip")) == 5 else None,
               "prior": prior_heldout(a.runs, chosen, final), "prior_validation": prior_validation(a.runs, chosen),
               "undercalled": undercalled(a.runs, chosen), "fit": table_fit(a.runs), "augment": table_augment(a.runs),
               "leakage": table_leakage(a.runs), "pointing": table_pointing(a.runs, chosen), "cost": table_cost(a.runs), "hours": hours_by_gpu(a.runs),
               "sweep": sweep_scores(a.runs), "captions": table_caption_examples(a.runs, final),
               "location_split": table_location_split(a.runs, ("archive/v1/blip-lora", "blip-lora", final, "balanced"))}
    (a.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    results["constant"] = table_constant(a.runs)
    (a.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (a.out / "numbers.tex").write_text(latex_numbers(results))
    for name, tex in (("ablation", latex_ablation(results["ablation"], results["constant"])), ("fold0", latex_fold0(results["fold0"])),
                      ("hedging", latex_hedging(results["hedging"], final)), ("per_class", latex_per_class(results["per_class"], final)),
                      ("cycles", latex_cycles(results["cycles"])),
                      ("leakage", latex_leakage(results["leakage"])), ("cost", latex_cost(results["cost"])),
                      ("captions", latex_captions(results["captions"]))):
        (a.out / f"{name}.tex").write_text(tex)
    print(json.dumps({k: (len(v) if hasattr(v, "__len__") else v) for k, v in results.items()}))


if __name__ == "__main__":
    main()
