"""Every chart in the report, rebuilt from the run records in runs/ (charts only: no tile imagery).

    python scripts/figures.py --runs runs --out report/figures
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm.metrics import risk_coverage, vqa_credit  # noqa: E402
from farmvlm.report import FAMILIES, credits, item_of, rows  # noqa: E402

# One colour per model across every figure (validated for colour-vision deficiency on white; dataviz palette).
INK, MUTED, GRID = "#1a1a19", "#6b6a64", "#e4e3de"
MODELS = {  # run name -> (label, colour)
    "baseline": ("Zero-shot BLIP", "#9a9992"),
    "blip-lora": ("BLIP-LoRA", "#2a78d6"),
    "served-rgb-eval": ("Qwen3.5-2B LoRA, RGB", "#eb6834"),
    "served-ndvi-eval": ("Qwen3.5-2B LoRA, RGB + NDVI", "#1baf7a"),
}
LR_COLOURS = {"2e-4": "#2a78d6", "5e-4": "#e34948"}
TASK_COLOURS = {"vqa": "#2a78d6", "caption": "#eda100"}
FAMILY_LABELS = {"presence": "Is X present?", "identify": "Which anomaly?", "extent": "How much area?", "location": "Where?"}
PERTURBATIONS = {"noise": "Gaussian noise, sd {}", "blur": "blur, radius {}", "jpeg": "JPEG quality {}", "light": "brightness and contrast, {}",
                 "resolution": "resolution / {}", "hflip": "horizontal flip", "rot90": "rotate 90°", "paraphrase": "unseen question wording"}


def perturbation(name: str) -> str:
    kind, _, s = name.partition("-")
    return PERTURBATIONS[kind].format(f"{float(s):.0%}" if kind == "light" else s)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def mean_family(m: dict) -> float:
    return float(np.mean([m["by_family"][f] for f in FAMILIES]))


def family_bars(runs: Path, out: Path):
    """Held-out fold 0, the same 13,004 items for every model, against the best constant answer per question type."""
    have = {n: json.loads((runs / n / "f0.json").read_text())["vqa_generated"] for n in MODELS if (runs / n / "f0.json").exists()}
    rs = [r for r in rows(runs / "blip-lora" / "f0.jsonl") if r["family"] != "caption"]
    const = {}
    for f in FAMILIES:
        its = [item_of(r) for r in rs if r["family"] == f]
        const[f] = max(np.mean([vqa_credit(i, a) for i in its]) for a in {i.answer for i in its})
    groups = list(FAMILIES) + ["mean"]
    x, w = np.arange(len(groups)), 0.8 / len(have)
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    for k, (name, m) in enumerate(have.items()):
        label, colour = MODELS[name]
        ys = [m["by_family"][f] for f in FAMILIES] + [mean_family(m)]
        ax.bar(x - 0.4 + w * (k + 0.5), ys, w * 0.88, color=colour, label=label)
    cs = [const[f] for f in FAMILIES] + [float(np.mean(list(const.values())))]
    ax.scatter(x, cs, marker="_", s=600, color=INK, linewidths=1.6, zorder=3, label="Always the most common answer")
    ax.set_xticks(x, [FAMILY_LABELS.get(g, "Mean over types") for g in groups])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy, held-out fold 0")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.savefig(out / "family_bars.pdf")
    plt.close(fig)


def lr_curves(runs: Path, out: Path):
    """BLIP VQA validation over training: the sweep's 5e-4 (archived) against the capped 2e-4, five folds each."""
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    ax.axvspan(60, 120, color=GRID, alpha=0.5, lw=0)
    ax.text(61, 0.71, "second half: best adapter and early stopping", color=MUTED, fontsize=7)
    for lr, base in (("5e-4", runs / "archive" / "vqa-lr5e-4"), ("2e-4", runs)):
        for f in range(5):
            p = base / f"blip-vqa-f{f}" / "val.jsonl"
            if not p.exists():
                continue
            ys = [mean_family(json.loads(l)) for l in p.read_text().splitlines()]
            ax.plot(5 * np.arange(1, len(ys) + 1), ys, color=LR_COLOURS[lr], lw=1.2, alpha=0.9,
                    label=f"learning rate {lr}" if f == 0 else None)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Training pairs (thousands)")
    ax.set_ylabel("Validation, mean over question types")
    ax.legend(loc="lower right")
    fig.savefig(out / "lr_curves.pdf")
    plt.close(fig)


def risk_coverage_plot(runs: Path, out: Path):
    """Selective answering on fold 0: accuracy on the most confident share of questions."""
    fig, ax = plt.subplots(figsize=(3.1, 2.4))
    for name, (label, colour) in MODELS.items():
        rs = [r for r in rows(runs / name / "f0.jsonl") if r["family"] != "caption"]
        if not rs:
            continue
        cov, acc, aurc = risk_coverage(np.array([r["conf"] for r in rs], float), (credits(rs) == 1.0).astype(float))
        keep = cov >= 0.05
        ax.plot(cov[keep], acc[keep], color=colour, lw=1.6, label=f"{label} (AURC {aurc:.2f})")
    ax.set_xlim(0.05, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of questions answered (most confident first)")
    ax.set_ylabel("Accuracy on answered questions")
    ax.legend(loc="lower left", fontsize=6)
    fig.savefig(out / "risk_coverage.pdf")
    plt.close(fig)


def accuracy(rs: list[dict]) -> float:
    return float(credits(rs).mean())


def robustness(runs: Path, out: Path):
    """Accuracy change from clean under each perturbation; Qwen on the first 200 of BLIP's 500 tiles, so BLIP is re-scored there too."""
    blip = runs / "robust-blip-lora"
    if not (blip / "f0-clean.jsonl").exists():
        return
    names = sorted(p.name.removesuffix(".jsonl").removeprefix("f0-") for p in blip.glob("f0-*.jsonl"))
    names = [n for n in names if n != "clean"]
    clean = accuracy(rows(blip / "f0-clean.jsonl"))
    drops = {n: accuracy(rows(blip / f"f0-{n}.jsonl")) - clean for n in names}
    served = [n for n in ("served-rgb", "served-ndvi") if (runs / f"robust-{n}" / "f0-clean.jsonl").exists()]
    fig, axes = plt.subplots(1, 2 if served else 1, figsize=(6.3, 3.0), squeeze=False)
    ax = axes[0][0]
    order = sorted(names, key=lambda n: drops[n])
    ax.barh(range(len(order)), [drops[n] for n in order], 0.7, color=MODELS["blip-lora"][1])
    ax.set_yticks(range(len(order)), [perturbation(n) for n in order])
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Accuracy change from clean (BLIP-LoRA, 500 tiles)")
    if served:
        ax = axes[0][1]
        rows_of = {n: {r["row"] for r in rows(runs / f"robust-{n}" / "f0-clean.jsonl")} for n in served}
        shared = sorted(p.name.removesuffix(".jsonl").removeprefix("f0-") for p in (runs / f"robust-{served[0]}").glob("f0-*.jsonl"))
        shared = [n for n in shared if n != "clean"]
        series = [("blip-lora", blip)] + [(f"served-{n.split('-')[1]}-eval", runs / f"robust-{n}") for n in served]
        y, h = np.arange(len(shared)), 0.8 / len(series)
        for k, (model, d) in enumerate(series):
            subset = rows_of[served[0]]
            c = accuracy([r for r in rows(d / "f0-clean.jsonl") if r["row"] in subset])
            vals = [accuracy([r for r in rows(d / f"f0-{n}.jsonl") if r["row"] in subset]) - c for n in shared]
            ax.barh(y - 0.4 + h * (k + 0.5), vals, h * 0.88, color=MODELS[model][1], label=MODELS[model][0])
        ax.set_yticks(y, [perturbation(n) for n in shared])
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Accuracy change from clean (same 200 tiles)")
        ax.legend(loc="lower left", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "robustness.pdf")
    plt.close(fig)


def sweep(runs: Path, out: Path):
    """The one-knob-at-a-time LoRA sweep (10k pairs per trial): rank, then learning rate, then target modules."""
    scores = {t: json.loads((runs / f"sweep-{t}" / "choice.json").read_text())["scores"] for t in ("vqa", "caption")}
    def parse(key: str) -> dict:  # "sweep-vqa/r64-lr5e-05-attn_qv"
        r, lr, targets = re.fullmatch(r"r(\d+)-lr([\d.e-]+)-(\w+)", key.split("/")[1]).groups()
        return {"rank": int(r), "lr": float(lr), "targets": targets}

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1), sharey=True)
    for t, colour in TASK_COLOURS.items():
        trials = [(parse(k), v) for k, v in scores[t].items()]
        rank = sorted((c["rank"], v) for c, v in trials if c["lr"] == 1e-4 and c["targets"] == "attn_qv")
        best_rank = max(rank, key=lambda p: p[1])[0]
        lr = sorted((c["lr"], v) for c, v in trials if c["rank"] == best_rank and c["targets"] == "attn_qv")
        best_lr = max(lr, key=lambda p: p[1])[0]
        tg = [(c["targets"], v) for c, v in trials if c["rank"] == best_rank and c["lr"] == best_lr]
        label = "VQA (mean over question types)" if t == "vqa" else "Captions (fact F1)"
        axes[0].plot(range(len(rank)), [v for _, v in rank], "-o", color=colour, lw=1.6, ms=4, label=label)
        axes[0].set_xticks(range(len(rank)), [str(r) for r, _ in rank])
        axes[1].plot(range(len(lr)), [v for _, v in lr], "-o", color=colour, lw=1.6, ms=4)
        axes[1].set_xticks(range(len(lr)), [f"{l:g}" for l, _ in lr])
        order = ["text_attn", "attn_qv", "all_linear"]
        tv = dict(tg)
        axes[2].plot(range(3), [tv.get(o, np.nan) for o in order], "o", color=colour, ms=5)
        axes[2].set_xticks(range(3), ["text q,v", "all q,v", "all linear"])
    axes[0].set_title("Rank (lr 1e-4, all q,v)")
    axes[1].set_title("Learning rate (best rank)")
    axes[2].set_title("Target modules (best rank, lr)")
    axes[0].set_ylabel("Validation score")
    axes[0].legend(loc="upper left", fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "sweep.pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("report/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for make in (family_bars, lr_curves, risk_coverage_plot, robustness, sweep):
        make(a.runs, a.out)
        print("made", make.__name__, flush=True)


if __name__ == "__main__":
    main()
