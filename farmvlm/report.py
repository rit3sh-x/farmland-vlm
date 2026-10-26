"""Every results table of issues/06-evaluation-protocol.md, rebuilt from the run records in runs/.

    python -m farmvlm.report --runs runs --out report/tables
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from farmvlm.metrics import ece, field_resample_means, grouped_bootstrap, is_hedge, margin_hedge, risk_coverage, vqa_credit
from farmvlm.items import Item
from farmvlm.text import lenient_answer
from farmvlm.scheme import CLASSES, CLASS_INDEX

FAMILIES = ("presence", "identify", "extent", "location")
FOLD0_ROWS = (("baseline", "Zero-shot BLIP"), ("blip-lora", "BLIP-LoRA"), ("dora", "BLIP-DoRA"),
              ("served-zeroshot-eval", "Qwen3.5-2B, zero-shot"), ("served-rgb-eval", "Qwen3.5-2B LoRA, RGB"),
              ("served-ndvi-eval", "Qwen3.5-2B LoRA, RGB + NDVI"))


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
    """Table 1: Baseline vs BLIP-LoRA over 5 folds, with the archived v1 BLIP-LoRA row when v2 has replaced it."""
    out = []
    for name, label in (("baseline", "Zero-shot BLIP"), ("archive/v1/blip-lora", "BLIP-LoRA v1"), ("blip-lora", "BLIP-LoRA")):
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
    for path in sorted((runs / "blip-lora").glob("f[0-4].jsonl")):
        rs = [r for r in rows(path) if r["family"] != "caption"]
        fam = {}
        for f in FAMILIES:
            its = [item_of(r) for r in rs if r["family"] == f]
            fam[f] = max(np.mean([vqa_credit(i, a) for i in its]) for a in {i.answer for i in its})
        folds.append(fam)
    return {f: mean_std([d[f] for d in folds]) for f in FAMILIES} | {"mean": mean_std([np.mean(list(d.values())) for d in folds])}


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
    for name, label in FOLD0_ROWS:
        if not (runs / name / "f0.jsonl").exists():
            continue
        ca, cb, g = aligned(runs, name, "blip-lora")
        m = json.loads((runs / name / "f0.json").read_text())
        rs = [x for x in rows(runs / name / "f0.jsonl") if x["family"] != "caption"]
        lenient = {f: float(np.mean([vqa_credit(item_of(x), lenient_answer(f, x["pred"])) for x in rs if x["family"] == f]))
                   for f in FAMILIES}
        out.append({"model": label, "acc": grouped_bootstrap(ca, g),
                    "diff": None if name == "blip-lora" else grouped_bootstrap(ca, g, other=cb),
                    "by_family": m["vqa_generated"]["by_family"], "by_family_lenient": lenient,
                    "caption": {k: m["caption_facts"][k] for k in ("precision", "recall", "f1")}})
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


def table_hedge_rule(runs: Path, name: str = "blip-lora", deltas=(0.0, 0.05, 0.1, 0.2, 0.3, 0.4)) -> list[dict]:
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


def table_pointing(runs: Path) -> dict:
    """The pointing game: baseline vs BLIP-LoRA hit rate and heat-in-mask energy, with chance."""
    p = runs / "heatmaps" / "pointing.json"
    return json.loads(p.read_text())["summary"] if p.exists() else {}


def latex_numbers(res: dict) -> str:
    """\\newcommand macros for the headline numbers, so the deck and the report quote the current run records."""
    def pct(x):
        v = f"{100 * x:.1f}"
        return f"${v}$" if v.startswith("-") else v
    abl = {r["model"]: r for r in res["ablation"]}
    f0 = {r["model"]: r for r in res["fold0"]}
    hed = {r["model"]: r for r in res["hedging"]}
    rob = res["robustness"].get("robust-blip-lora", {})
    n = {}
    if "BLIP-LoRA" in abl:
        b = abl["BLIP-LoRA"]
        n.update(blipMean=pct(b["mean"][0]), blipMeanStd=pct(b["mean"][1]), blipFactF=pct(b["fact_f1"][0]),
                 blipFactFStd=pct(b["fact_f1"][1]), blipPresence=pct(b["presence"][0]), blipIdentify=pct(b["identify"][0]),
                 blipExtent=pct(b["extent"][0]), blipLocation=pct(b["location"][0]), blipRanked=pct(b["mean_ranked"][0]))
    if "Zero-shot BLIP" in abl:
        z = abl["Zero-shot BLIP"]
        n.update(zeroMean=pct(z["mean"][0]), zeroPresence=pct(z["presence"][0]), zeroIdentify=pct(z["identify"][0]))
    if res.get("constant"):
        c = res["constant"]
        n.update(constMean=pct(c["mean"][0]), constLocation=pct(c["location"][0]), constIdentify=pct(c["identify"][0]))
    if "blip-lora" in hed:
        h = hed["blip-lora"]
        n.update(blipAccEighty=pct(h["acc_at_80"]), blipAurc=f"{h['aurc']:.2f}", blipCreditAmb=pct(h["credit_amb"]),
                 blipAccClear=pct(h["acc_clear"]), blipEce=f"{h['ece']:.2f}")
    if "baseline" in hed:
        n["zeroAurc"] = f"{hed['baseline']['aurc']:.2f}"
    if rob:
        n.update(blipParaphraseDrop=pct(-rob["paraphrase"]), blipJpegDrop=pct(-rob.get("jpeg-10", 0.0)),
                 blipBlurDrop=pct(-rob.get("blur-4", 0.0)))
    if "gap" in res.get("leakage", {}):
        g = res["leakage"]["gap"]
        n.update(leakGap=pct(g[0]), leakLo=pct(g[1]), leakHi=pct(g[2]), leakSeen=pct(res["leakage"]["seen-field"][0]),
                 leakUnseen=pct(res["leakage"]["unseen-field"][0]))
    if "NDVI minus RGB" in f0:
        d = f0["NDVI minus RGB"]["diff"]
        n.update(ndviDiff=pct(d[0]), ndviLo=pct(d[1]), ndviHi=pct(d[2]))
    for key, name in (("Qwen3.5-2B LoRA, RGB", "qwenRgb"), ("Qwen3.5-2B LoRA, RGB + NDVI", "qwenNdvi"), ("BLIP-LoRA", "blipFold"),
                      ("BLIP-DoRA", "dora")):
        if key in f0:
            fam = f0[key]["by_family"]
            n[name + "Mean"] = pct(np.mean([fam[f] for f in FAMILIES]))
            n[name + "Presence"] = pct(fam["presence"])
            n[name + "FactF"] = pct(f0[key]["caption"]["f1"])
            n[name + "Precision"] = pct(f0[key]["caption"]["precision"])
    if "BLIP-DoRA" in f0 and f0["BLIP-DoRA"]["diff"]:
        n["doraDiff"] = pct(f0["BLIP-DoRA"]["diff"][0])
    if "Qwen3.5-2B, zero-shot" in f0:
        n["qwenZeroLenient"] = pct(np.mean([f0["Qwen3.5-2B, zero-shot"]["by_family_lenient"][f] for f in FAMILIES]))
    if res.get("pointing"):
        p = res["pointing"]
        n.update(pointZero=pct(p["baseline"]["hit_rate"]), pointLora=pct(p["blip-lora"]["hit_rate"]),
                 pointChance=pct(p["blip-lora"]["chance"]), energyZero=pct(p["baseline"]["energy"]), energyLora=pct(p["blip-lora"]["energy"]))
    return "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in n.items())


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
        fam = [f"{fmt(*r[f])} ({fmt(r[f + '_ranked'][0])})" for f in FAMILIES] + [f"{fmt(*r['mean'])} ({fmt(r['mean_ranked'][0])})"]
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


def latex_hedging(t4: list[dict]) -> str:
    labels = dict(FOLD0_ROWS)
    body = [" & ".join([labels.get(r["model"], r["model"]), f"{100 * r['hedge_amb']:.1f}", f"{100 * r['hedge_clear']:.1f}",
                        f"{100 * r['credit_amb']:.1f}", f"{100 * r['acc_clear']:.1f}", f"{r['ece']:.3f}", f"{r['aurc']:.3f}",
                        f"{100 * r['acc_at_80']:.1f}"]) for r in t4]
    head = "Model & Hedge, amb. & Hedge, clear & Credit, amb. & Acc., clear & ECE & AURC & Acc. at 80\\% cov."
    return tabular("l" + "c" * 7, head, body)


def latex_per_class(t2: dict) -> str:
    names = [n for n in ("blip-lora", "served-rgb-eval", "served-ndvi-eval") if n in t2]
    labels = {"blip-lora": "BLIP-LoRA", "served-rgb-eval": "Qwen, RGB", "served-ndvi-eval": "Qwen, RGB + NDVI"}
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
    results = {"ablation": table_ablation(a.runs), "per_class": table_per_class(a.runs), "fold0": table_fold0(a.runs),
               "hedging": table_hedging(a.runs), "hedge_rule": table_hedge_rule(a.runs), "robustness": table_robustness(a.runs),
               "leakage": table_leakage(a.runs), "pointing": table_pointing(a.runs),
               "sweep": sweep_scores(a.runs)}
    (a.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    results["constant"] = table_constant(a.runs)
    (a.out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    (a.out / "numbers.tex").write_text(latex_numbers(results))
    for name, tex in (("ablation", latex_ablation(results["ablation"], results["constant"])), ("fold0", latex_fold0(results["fold0"])),
                      ("hedging", latex_hedging(results["hedging"])), ("per_class", latex_per_class(results["per_class"])),
                      ("leakage", latex_leakage(results["leakage"]))):
        (a.out / f"{name}.tex").write_text(tex)
    print(json.dumps({k: (len(v) if hasattr(v, "__len__") else v) for k, v in results.items()}))


if __name__ == "__main__":
    main()
