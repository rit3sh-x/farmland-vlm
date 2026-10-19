"""Every GPU run of the project, in schedule order (issues/16). Idempotent: finished stages are skipped
(locally or on the Hub), an interrupted training run resumes from its last pushed checkpoint.

    HF_TOKEN=... python scripts/pipeline.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021
    python scripts/pipeline.py ... --only baseline sweep
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm import hub  # noqa: E402

RUNS = Path("/content/runs")
CODE = Path(__file__).resolve().parent.parent
SWEEP = {"rank": [4, 8, 16, 32, 64], "lr": [5e-5, 1e-4, 2e-4, 5e-4], "targets": ["text_attn", "attn_qv", "all_linear"]}
START = {"rank": 16, "lr": 1e-4, "targets": "attn_qv"}


def sh(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=CODE)


def fetch(name: str, marker: str = "done.json") -> bool:
    """True if the run is finished, pulling it from the Hub when it is not on this VM."""
    if (RUNS / name / marker).exists():
        return True
    if hub.has(name, marker):
        hub.pull(name, RUNS)
        return True
    return False


def resume_args(name: str) -> list:
    if hub.has(name, "last/train_state.pt"):
        hub.pull(name, RUNS)
    last = RUNS / name / "last"
    return ["--resume", last] if (last / "train_state.pt").exists() else []


def flags(**kw):
    out = []
    for k, v in kw.items():
        out += [f"--{k.replace('_', '-')}"] + ([] if v is True else [v])
    return out


def train_blip(name, a, task, **kw):
    if not fetch(name):
        sh(sys.executable, "-m", "farmvlm.train_blip", "--task", task, "--stats", a.stats, "--root", a.root,
           "--out", RUNS / name, *flags(**kw), *resume_args(name))
    return json.loads((RUNS / name / "done.json").read_text())["best_val"]


def evaluate(name, a, script, *args, marker="f4.json"):
    if not fetch(name, marker):
        sh(sys.executable, CODE / "scripts" / script, "--stats", a.stats, "--root", a.root, "--out", RUNS / name, *args)
        hub.push(RUNS / name, name)


def sweep(a, task):
    """One knob at a time from START: rank, then LR at the best rank, then targets; each trial ~10k pairs."""
    if fetch(f"sweep-{task}", "choice.json"):
        return json.loads((RUNS / f"sweep-{task}" / "choice.json").read_text())
    name = lambda c: f"sweep-{task}/r{c['rank']}-lr{c['lr']:g}-{c['targets']}"
    cfg, scores = dict(START), {}
    for knob, values in SWEEP.items():
        trials = [{**cfg, knob: v} for v in values]
        for c in trials:
            scores[name(c)] = train_blip(name(c), a, task, rank=c["rank"], lr=c["lr"], targets=c["targets"],
                                         pairs=10_000, eval_every=5_000, patience=99)
        cfg = max(trials, key=lambda c: scores[name(c)])
    (RUNS / f"sweep-{task}").mkdir(parents=True, exist_ok=True)
    (RUNS / f"sweep-{task}" / "choice.json").write_text(json.dumps({"choice": cfg, "scores": scores}, indent=1))
    hub.push(RUNS / f"sweep-{task}", f"sweep-{task}", include=("choice.json",))
    return {"choice": cfg, "scores": scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    want = lambda s: not a.only or s in a.only

    if want("baseline"):
        evaluate("baseline", a, "eval_blip.py")
    best = {}
    if want("sweep") or want("blip"):
        best = {t: sweep(a, t)["choice"] for t in ("vqa", "caption")}
    if want("bakeoff") or want("served"):
        if not fetch("bakeoff", "bakeoff.json"):
            sh(sys.executable, CODE / "scripts" / "bakeoff.py", "--stats", a.stats, "--root", a.root, "--out", RUNS / "bakeoff")
            hub.push(RUNS / "bakeoff", "bakeoff", include=("bakeoff.json",))
    if want("blip"):
        for f in range(5):
            for t in ("vqa", "caption"):
                c = best[t]
                train_blip(f"blip-{t}-f{f}", a, t, fold=f, rank=c["rank"], lr=c["lr"], targets=c["targets"], pairs=120_000)
        evaluate("blip-lora", a, "eval_blip.py", "--vqa-adapter", RUNS / "blip-vqa-f{fold}" / "best",
                 "--caption-adapter", RUNS / "blip-caption-f{fold}" / "best")
    if want("heatmaps") and not fetch("heatmaps", "pointing.json"):
        sh(sys.executable, CODE / "scripts" / "heatmaps.py", "--stats", a.stats, "--root", a.root,
           "--adapter", RUNS / "blip-vqa-f0" / "best", "--out", RUNS / "heatmaps")
        hub.push(RUNS / "heatmaps", "heatmaps")
    if want("served"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        extra = ["--vision"] if b["merged_route"] else []
        for name, flags in (("served-rgb", []), ("served-ndvi", ["--ndvi"])):
            if not fetch(name):
                sh(sys.executable, "-m", "farmvlm.train_vlm", "--model", b["winner"], "--stats", a.stats, "--root", a.root,
                   "--pairs", 60_000, "--out", RUNS / name, *extra, *flags, *resume_args(name))
            evaluate(f"{name}-eval", a, "eval_vlm.py", "--model", b["winner"], "--adapter", RUNS / name / "best", *flags, marker="f0.json")
        evaluate("served-zeroshot-eval", a, "eval_vlm.py", "--model", b["winner"], marker="f0.json")
    if want("leakage"):
        c = best.get("vqa") or json.loads((RUNS / "sweep-vqa" / "choice.json").read_text())["choice"]
        train_blip("leak-vqa-official", a, "vqa", split="official", rank=c["rank"], lr=c["lr"], targets=c["targets"], pairs=120_000)
        evaluate("leakage", a, "eval_blip.py", "--suite", "leakage", "--vqa-adapter", RUNS / "leak-vqa-official" / "best",
                 marker="unseen-field.json")
    if want("robustness"):
        evaluate("robust-baseline", a, "eval_blip.py", "--suite", "robustness", "--folds", 0, marker="f0-rot90-None.json")
        evaluate("robust-blip-lora", a, "eval_blip.py", "--suite", "robustness", "--folds", 0,
                 "--vqa-adapter", RUNS / "blip-vqa-f0" / "best", "--caption-adapter", RUNS / "blip-caption-f0" / "best",
                 marker="f0-rot90-None.json")
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for name, flags in (("served-rgb", []), ("served-ndvi", ["--ndvi"])):
            evaluate(f"robust-{name}", a, "eval_vlm.py", "--model", b["winner"], "--adapter", RUNS / name / "best",
                     "--suite", "robustness", *flags, marker="f0-rot90-None.json")
    if want("dora"):
        c = best.get("vqa") or json.loads((RUNS / "sweep-vqa" / "choice.json").read_text())["choice"]
        train_blip("blip-vqa-f0-dora", a, "vqa", fold=0, rank=c["rank"], lr=c["lr"], targets=c["targets"], dora=True, pairs=120_000)
        evaluate("dora", a, "eval_blip.py", "--folds", 0, "--vqa-adapter", RUNS / "blip-vqa-f0-dora" / "best",
                 "--caption-adapter", RUNS / "blip-caption-f0" / "best", marker="f0.json")
    if want("export"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for name in ("served-rgb", "served-ndvi"):
            out = RUNS / f"export-{name}"
            if not fetch(f"export-{name}", "lora-f16.gguf" if not b["merged_route"] else "model-Q4_K_M.gguf"):
                sh(sys.executable, CODE / "scripts" / "export_gguf.py", "--model", b["winner"], "--adapter", RUNS / name / "best",
                   "--out", out, *(["--merged"] if b["merged_route"] else []))
                hub.push(out, f"export-{name}", include=("*.gguf",))


if __name__ == "__main__":
    main()
