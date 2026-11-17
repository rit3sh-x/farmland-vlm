import argparse
import json
import re
from pathlib import Path

import numpy as np

from farmvlm.metrics import risk_coverage, vqa_credit
from farmvlm.report import FAMILIES, chosen_cycle, credits, final_run, item_of, robustness_shared, rows, table_cycles

STYLE = r"""% Written by scripts/figures.py. Needs \usepackage{pgfplots}.
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{figink}{HTML}{1A1A19}
\definecolor{figmuted}{HTML}{6B6A64}
\definecolor{figgrid}{HTML}{E4E3DE}
\definecolor{mbase}{HTML}{9A9992}
\definecolor{mblip}{HTML}{2A78D6}
\definecolor{mrgb}{HTML}{EB6834}
\definecolor{mndvi}{HTML}{1BAF7A}
\definecolor{mred}{HTML}{E34948}
\definecolor{mgold}{HTML}{EDA100}
\pgfplotsset{chart/.style={
  scale only axis, axis line style={figgrid}, axis x line*=bottom, axis y line*=left, tick style={draw=none},
  ticklabel style={font=\scriptsize, text=figmuted}, label style={font=\footnotesize}, title style={font=\footnotesize},
  ymajorgrids, grid style={figgrid, line width=0.4pt}, scaled ticks=false,
  xticklabel style={/pgf/number format/fixed}, yticklabel style={/pgf/number format/fixed},
  legend style={draw=none, fill=none, font=\scriptsize, /tikz/every even column/.append style={column sep=8pt}},
  legend cell align=left}}
\newlength\figW\newlength\figH
\newcommand{\fig}[3]{{\setlength\figW{#2}\setlength\figH{#3}\input{figures/#1.tex}}}
"""
MODELS = {
    "baseline": ("Zero-shot BLIP", "mbase"),
    "final": ("BLIP-LoRA", "mblip"),
    "served-rgb-eval": ("Qwen3.5-2B LoRA, RGB", "mrgb"),
    "served-ndvi-eval": ("Qwen3.5-2B LoRA, RGB + NDVI", "mndvi"),
}
LR_COLOURS = {"2e-4": "mblip", "5e-4": "mred"}
TASK_COLOURS = {"vqa": "mblip", "caption": "mgold"}
FAMILY_LABELS = {"presence": "Is X present?", "identify": "Which anomaly?", "extent": "How much area?", "location": "Where?"}
PERTURBATIONS = {"noise": "Gaussian noise, sd {}", "blur": "blur, radius {}", "jpeg": "JPEG quality {}", "light": "brightness and contrast, {}",
                 "resolution": "resolution / {}", "hflip": "horizontal flip", "rot90": "rotate 90°", "paraphrase": "unseen question wording"}


def esc(s: str) -> str:
    return s.replace("\\", r"\textbackslash{}").replace("%", r"\%").replace("&", r"\&").replace("_", r"\_").replace("#", r"\#").replace("°", r"$^\circ$")


def sci(x: float) -> str:
    m, e = f"{x:.0e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def pts(xs, ys) -> str:
    return " ".join(f"({x:.5g},{y:.5g})" for x, y in zip(xs, ys))


def labels(items) -> str:
    return ",".join("{" + esc(str(i)) + "}" for i in items)


def plot(style: str, xs, ys, legend: str | None = None) -> str:
    entry = rf" \addlegendentry{{{legend}}}" if legend else ""
    return rf"\addplot[{style}{'' if legend else ', forget plot'}] coordinates {{{pts(xs, ys)}}};" + entry


def write(out: Path, name: str, body: list[str]):
    (out / f"{name}.tex").write_text("% Written by scripts/figures.py from the run records.\n\\begin{tikzpicture}\n"
                                     + "\n".join(body) + "\n\\end{tikzpicture}\n")


def axis(opts: str, body: list[str], w: float = 0.9, h: float = 0.82) -> list[str]:
    return [rf"\begin{{axis}}[chart, width={w}\figW, height={h}\figH, {opts}]", *body, r"\end{axis}"]


def perturbation(name: str) -> str:
    kind, _, s = name.partition("-")
    return PERTURBATIONS[kind].format(f"{float(s):.0%}" if kind == "light" else s)


def mean_family(m: dict) -> float:
    return float(np.mean([m["by_family"][f] for f in FAMILIES]))


def family_bars(runs: Path, out: Path):
    have = {n: json.loads((runs / n / "f0.json").read_text())["vqa_generated"] for n in MODELS if (runs / n / "f0.json").exists()}
    rs = [r for r in rows(runs / "blip-lora" / "f0.jsonl") if r["family"] != "caption"]
    const = {}
    for f in FAMILIES:
        its = [item_of(r) for r in rs if r["family"] == f]
        const[f] = max(np.mean([vqa_credit(i, a) for i in its]) for a in {i.answer for i in its})
    groups = [FAMILY_LABELS[f] for f in FAMILIES] + ["Mean over types"]
    x = range(len(groups))
    body = []
    for name, m in have.items():
        label, colour = MODELS[name]
        body.append(plot(f"fill={colour}, draw=none, area legend", x, [m["by_family"][f] for f in FAMILIES] + [mean_family(m)], label))
    cs = [const[f] for f in FAMILIES] + [float(np.mean(list(const.values())))]
    body.append(r"\addlegendimage{legend image code/.code={\draw[figink, line width=1.2pt] (0cm,0cm) -- (0.5cm,0cm);}} \addlegendentry{Always the most common answer}")
    body += [rf"\draw[figink, line width=1.2pt] ({k - 0.42},{c:.4f}) -- ({k + 0.42},{c:.4f});" for k, c in enumerate(cs)]
    write(out, "family_bars", axis(
        rf"ybar=0pt, bar width={0.8 / len(have):.3f}, ymin=0, ymax=1, xtick={{0,...,{len(groups) - 1}}}, "
        rf"xticklabels={{{labels(groups)}}}, enlarge x limits=0.12, ylabel={{Accuracy, held-out fold 0}}, "
        r"legend columns=3, legend style={at={(0.5,1.03)}, anchor=south}", body, w=0.92, h=0.66))


def lr_curves(runs: Path, out: Path):
    body = [r"\fill[figgrid, opacity=0.6] (60,0) rectangle (120,0.75);",
            r"\node[anchor=north west, font=\scriptsize, text=figmuted] at (61,0.745) {second half: best adapter and early stopping};"]
    for lr, base in (("5e-4", runs / "archive" / "vqa-lr5e-4"), ("2e-4", runs)):
        first = True
        for f in range(5):
            p = base / f"blip-vqa-f{f}" / "val.jsonl"
            if not p.exists():
                continue
            ys = [mean_family(json.loads(l)) for l in p.read_text().splitlines()]
            body.append(plot(f"{LR_COLOURS[lr]}, line width=0.8pt", 5 * np.arange(1, len(ys) + 1), ys,
                             f"learning rate {sci(float(lr))}" if first else None))
            first = False
    write(out, "lr_curves", axis(
        "xmin=0, xmax=120, ymin=0, ymax=0.75, xlabel={Training pairs (thousands)}, "
        "ylabel={Validation, mean over types}, legend pos=south east", body, w=0.88))


def risk_coverage_plot(runs: Path, out: Path):
    body = []
    for name, (label, colour) in MODELS.items():
        rs = [r for r in rows(runs / name / "f0.jsonl") if r["family"] != "caption"]
        if not rs:
            continue
        cov, acc, aurc = risk_coverage(np.array([r["conf"] for r in rs], float), (credits(rs) == 1.0).astype(float))
        keep = np.flatnonzero(cov >= 0.05)
        keep = keep[np.unique(np.linspace(0, len(keep) - 1, 160).astype(int))]
        body.append(plot(f"{colour}, line width=1.1pt", cov[keep], acc[keep], f"{label} (AURC {aurc:.2f})"))
    write(out, "risk_coverage", axis(
        "xmin=0.05, xmax=1, ymin=0, ymax=1, xlabel={Share of questions answered, most confident first}, "
        r"ylabel={Accuracy on answered}, legend pos=south west, legend style={font=\tiny}", body, w=0.84, h=0.8))


def accuracy(rs: list[dict]) -> float:
    return float(credits(rs).mean())


def robustness(runs: Path, out: Path):
    blip = runs / f"robust-c{chosen_cycle(runs)}"
    final = next(n for n, (label, _) in MODELS.items() if label == "BLIP-LoRA")
    if not (blip / "f0-clean.jsonl").exists():
        return
    names = sorted(p.name.removesuffix(".jsonl").removeprefix("f0-") for p in blip.glob("f0-*.jsonl"))
    names = [n for n in names if n != "clean"]
    clean = accuracy(rows(blip / "f0-clean.jsonl"))
    drops = {n: accuracy(rows(blip / f"f0-{n}.jsonl")) - clean for n in names}
    order = sorted(names, key=lambda n: drops[n])
    hbar = r"xmajorgrids, ymajorgrids=false, y dir=reverse, enlarge y limits={abs=0.6}, xtick distance=0.05"
    left = [rf"\nextgroupplot[{hbar}, xbar=0pt, bar width=0.65, ytick={{0,...,{len(order) - 1}}}, yticklabels={{{labels(perturbation(n) for n in order)}}}, "
            r"yticklabel style={font=\tiny}, xlabel={Accuracy change from clean (BLIP-LoRA, 500 tiles)}]",
            plot(f"fill={MODELS[final][1]}, draw=none", [drops[n] for n in order], range(len(order))),
            rf"\draw[figmuted, line width=0.5pt] (0,-0.6) -- (0,{len(order) - 0.4});"]
    change = robustness_shared(runs)
    if not change:
        write(out, "robustness", [r"\begin{groupplot}[chart, group style={group size=1 by 1}, width=0.5\figW, height=\figH]", *left, r"\end{groupplot}"])
        return
    series = ((final, "blip"), ("served-rgb-eval", "servedRgb"), ("served-ndvi-eval", "servedNdvi"))
    shared = sorted(change["blip"], key=change["blip"].get)

    def models_panel(opts: str, legend_at: str) -> list[str]:
        body = [rf"[{hbar}, xbar=0pt, bar width={0.8 / len(series):.3f}, ytick={{0,...,{len(shared) - 1}}}, "
                rf"yticklabels={{{labels(perturbation(n) for n in shared)}}}, xlabel={{Accuracy change from clean (same 200 tiles)}}, "
                rf"legend columns=3, legend style={{{legend_at}}}, {opts}]"]
        body += [plot(f"fill={MODELS[m][1]}, draw=none, area legend", [change[k][n] for n in shared], range(len(shared)), MODELS[m][0]) for m, k in series]
        return body + [rf"\draw[figmuted, line width=0.5pt] (0,-0.6) -- (0,{len(shared) - 0.4});"]

    right = models_panel("", "at={(1,1.02)}, anchor=south east")
    write(out, "robustness", [r"\begin{groupplot}[chart, group style={group size=2 by 1, horizontal sep=0.27\figW}, width=0.24\figW, height=0.84\figH]",
                              *left, r"\nextgroupplot" + right[0], *right[1:], r"\end{groupplot}"])
    alone = models_panel(r"chart, width=0.62\figW, height=0.78\figH", "at={(0.5,1.02)}, anchor=south")
    write(out, "robustness_models", [r"\begin{axis}" + alone[0], *alone[1:], r"\end{axis}"])


def length_curve(runs: Path, out: Path):
    t = table_cycles(runs)
    x = np.array([r["pairs"] for r in t]) / 1000
    body = []
    for key, label, colour in (("ranked", "Held out, ranked", "mblip"), ("greedy", "Held out, greedy", "mrgb")):
        m, sd = np.array([r[key][0] for r in t]) * 100, np.array([r[key][1] for r in t]) * 100
        coords = " ".join(f"({xi:.5g},{mi:.4g}) +- (0,{si:.3g})" for xi, mi, si in zip(x, m, sd))
        body.append(rf"\addplot[{colour}, line width=1.1pt, mark=*, mark size=1.4pt, error bars/.cd, y dir=both, y explicit, "
                    rf"error bar style={{line width=0.5pt}}] coordinates {{{coords}}}; \addlegendentry{{{label}}}")
    v = [(xi, r["val"] * 100) for xi, r in zip(x, t) if r["val"]]
    body.append(plot("mndvi, line width=1.1pt, dashed, mark=square*, mark size=1.3pt, mark options={solid}", *zip(*v), "Validation (selection score)"))
    write(out, "length_curve", axis(
        rf"xmode=log, log basis x=2, xtick={{{','.join(f'{p:g}' for p in x)}}}, xticklabels={{{labels(f'{int(p)}k' for p in x)}}}, "
        r"enlarge x limits=0.08, xlabel={Training pairs per fold}, ylabel={Mean over question types (\%)}, legend pos=south east", body, w=0.8, h=0.8))


def sweep(runs: Path, out: Path):
    scores = {t: json.loads((runs / f"sweep-{t}" / "choice.json").read_text())["scores"] for t in ("vqa", "caption")}

    def parse(key: str) -> dict:
        r, lr, targets = re.fullmatch(r"r(\d+)-lr([\d.e-]+)-(\w+)", key.split("/")[1]).groups()
        return {"rank": int(r), "lr": float(lr), "targets": targets}

    order = ["text_attn", "attn_qv", "all_linear"]
    panels = {"rank": [], "lr": [], "targets": []}
    ticks = {}
    for t, colour in TASK_COLOURS.items():
        trials = [(parse(k), v) for k, v in scores[t].items()]
        rank = sorted((c["rank"], v) for c, v in trials if c["lr"] == 1e-4 and c["targets"] == "attn_qv")
        best_rank = max(rank, key=lambda p: p[1])[0]
        lr = sorted((c["lr"], v) for c, v in trials if c["rank"] == best_rank and c["targets"] == "attn_qv")
        best_lr = max(lr, key=lambda p: p[1])[0]
        tv = {c["targets"]: v for c, v in trials if c["rank"] == best_rank and c["lr"] == best_lr}
        label = "VQA (mean over question types)" if t == "vqa" else "Captions (fact F1)"
        line = f"{colour}, line width=1.1pt, mark=*, mark size=1.5pt"
        panels["rank"].append(plot(line, range(len(rank)), [v for _, v in rank], label))
        panels["lr"].append(plot(line, range(len(lr)), [v for _, v in lr]))
        panels["targets"].append(plot(f"{colour}, only marks, mark=*, mark size=2pt", *zip(*[(i, tv[o]) for i, o in enumerate(order) if o in tv])))
        ticks["rank"], ticks["lr"] = [str(r) for r, _ in rank], [f"{l:.0e}".replace("e-0", "e-") for l, _ in lr]
    ticks["targets"] = ["text q,v", "all q,v", "all linear"]
    titles = {"rank": r"Rank (lr $10^{-4}$, all q,v)", "lr": "Learning rate (best rank)", "targets": "Target modules (best rank, lr)"}
    body = [r"\begin{groupplot}[chart, group style={group size=3 by 1, horizontal sep=0.035\figW, yticklabels at=edge left}, "
            r"width=0.28\figW, height=0.68\figH, enlarge x limits=0.12, ymin=0.25, ymax=0.75]"]
    for k, (key, plots) in enumerate(panels.items()):
        extra = r", ylabel={Validation score}, legend pos=north west, legend style={font=\tiny}" if k == 0 else ""
        tick_labels = ",".join("{" + s + "}" for s in ticks[key])
        body += [rf"\nextgroupplot[title={{{titles[key]}}}, xtick={{0,...,{len(ticks[key]) - 1}}}, xticklabels={{{tick_labels}}}{extra}]", *plots]
    write(out, "sweep", body + [r"\end{groupplot}"])


def dataset(stats: Path, out: Path):
    from farmvlm.folds import tile_presence
    from farmvlm.scheme import CLASSES, EXTENT_CUTS, VIEWS
    from farmvlm.stats import StatsTable
    t = StatsTable.load(stats)
    pres = tile_presence(t)
    order = np.argsort(-pres.sum(0))
    full = t.row_view == VIEWS.index("full")
    area = t.area[full][t.area[full] > 0].astype(np.float32)
    counts, edges = np.histogram(100 * area, bins=60, range=(0, 100))
    top = counts.max() / 1000 * 1.12
    body = [r"\begin{groupplot}[chart, group style={group size=1 by 2, vertical sep=0.13\figW}, width=0.63\figW]",
            rf"\nextgroupplot[height=0.4\figH, xbar=0pt, bar width=0.65, y dir=reverse, xmajorgrids, ymajorgrids=false, enlarge y limits={{abs=0.6}}, "
            rf"ytick={{0,...,{len(order) - 1}}}, yticklabels={{{labels(CLASSES[k].name for k in order)}}}, xmin=0, "
            r"xlabel={Labelled tiles with the class (\%)}]",
            plot("fill=mblip, draw=none", [100 * pres[:, k].mean() for k in order], range(len(order))),
            rf"\nextgroupplot[height=0.3\figH, xmin=0, xmax=100, ymin=0, ymax={top:.4g}, xlabel={{Area of a present class (\% of the tile)}}, "
            r"ylabel={Findings (thousands)}]",
            r"\addplot[ybar interval, fill=mblip, draw=white, line width=0.2pt] coordinates {"
            + pts(edges, np.r_[counts, counts[-1]] / 1000) + "};"]
    for c, label in zip(EXTENT_CUTS, ("small | moderate", "moderate | large")):
        body += [rf"\draw[figink, dashed, line width=0.6pt] ({100 * c:.4g},0) -- ({100 * c:.4g},{top:.4g});",
                 rf"\node[rotate=90, anchor=north east, font=\tiny, text=figink] at ({100 * c:.4g},{top:.4g}) {{{label}}};"]
    write(out, "dataset", body + [r"\end{groupplot}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--stats", type=Path, default=Path("data/stats.npz"))
    ap.add_argument("--out", type=Path, default=Path("report/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "style.tex").write_text(STYLE)
    global MODELS
    MODELS = {final_run(a.runs) if k == "final" else k: v for k, v in MODELS.items()}
    for make in (family_bars, lr_curves, risk_coverage_plot, robustness, sweep, length_curve):
        try:
            make(a.runs, a.out)
            print("made", make.__name__, flush=True)
        except (FileNotFoundError, ValueError, KeyError, IndexError) as e:
            print("skipped", make.__name__, "-", type(e).__name__, str(e)[:80], flush=True)
    if a.stats.exists():
        dataset(a.stats, a.out)
        print("made dataset", flush=True)


if __name__ == "__main__":
    main()
