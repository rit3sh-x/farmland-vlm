"""Every GPU run of the project, graded BLIP work first, then the served model (issues/16).
Idempotent: finished stages are skipped (locally or on the Hub), an interrupted training run resumes
from its last pushed checkpoint.

    HF_TOKEN=... python scripts/pipeline.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021
    python scripts/pipeline.py ... --only baseline sweep

Stages: baseline sweep bakeoff blip robustness heatmaps leakage dora served export served-robustness.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm import hub  # noqa: E402

RUNS = Path("/content/runs")
CODE = Path(__file__).resolve().parent.parent
SWEEP = {"rank": [4, 8, 16, 32, 64], "lr": [5e-5, 1e-4, 2e-4, 5e-4], "targets": ["text_attn", "attn_qv", "all_linear"]}
START = {"rank": 16, "lr": 1e-4, "targets": "attn_qv"}
# The 10k-pair sweep anneals its LR within 10k pairs; held near 5e-4 for longer, BLIP VQA fold 0 collapsed
# (identify 0.34 -> 0.00 by 20k pairs). Final runs take the sweep choice with the LR capped here; the sweep
# scored 2e-4 within a point of 5e-4.
FINAL_LR_CAP = 2e-4
SERVED_PAIRS = 30_000  # per cycle: Qwen3.5-2B trains at ~146 min per 10k pairs on a T4
# Cycle 2 restarts the cosine schedule from cycle 1's best adapter (its validation was still rising at 30k pairs).
SERVED_CYCLES = 2
# A served validation costs Qwen ~40 min per 900 items on a T4, as much as 5k pairs of training: 75 tiles, 3 checks.
SERVED_EVAL_EVERY = 10_000
SERVED_VAL_TILES = 75


def sh(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=CODE)


def visible_gpus() -> list[str]:
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    try:
        n = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")
    except FileNotFoundError:
        n = 0
    return [str(i) for i in range(n)]


def sh_parallel(cmds: list[list]):
    """Each command on its own visible GPU when there are enough (Kaggle's T4 x2), else one after another."""
    gpus = visible_gpus()
    if len(cmds) < 2 or len(gpus) < len(cmds):
        for c in cmds:
            sh(*c)
        return
    procs = []
    for gpu, c in zip(gpus, cmds):
        print(f"+ [gpu {gpu}]", " ".join(map(str, c)), flush=True)
        procs.append(subprocess.Popen([str(x) for x in c], cwd=CODE, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise subprocess.CalledProcessError(max(codes), cmds[codes.index(max(codes))])


SERVED_VARIANTS = (("served-rgb", []), ("served-ndvi", ["--ndvi"]))


def served_run(name: str, cycle: int = SERVED_CYCLES) -> str:
    """Run name of a served variant's training cycle: served-rgb for cycle 1, served-rgb-c2 after."""
    return name if cycle == 1 else f"{name}-c{cycle}"


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
    ap.add_argument("--tasks", nargs="*", default=["vqa", "caption"], choices=["vqa", "caption"], help="BLIP tasks to sweep and train")
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(5)), help="BLIP folds to train")
    a = ap.parse_args()
    RUNS.mkdir(parents=True, exist_ok=True)
    want = lambda s: not a.only or s in a.only

    if want("baseline"):
        evaluate("baseline", a, "eval_blip.py")
    final = lambda c: {**c, "lr": min(c["lr"], FINAL_LR_CAP)}
    best = {}
    if want("sweep") or want("blip"):
        best = {t: final(sweep(a, t)["choice"]) for t in a.tasks}
    if want("bakeoff") or want("served"):
        if not fetch("bakeoff", "bakeoff.json"):
            sh(sys.executable, CODE / "scripts" / "bakeoff.py", "--stats", a.stats, "--root", a.root, "--out", RUNS / "bakeoff")
            hub.push(RUNS / "bakeoff", "bakeoff", include=("bakeoff.json",))
    if want("blip"):
        for f in a.folds:
            for t in a.tasks:
                c = best[t]
                train_blip(f"blip-{t}-f{f}", a, t, fold=f, rank=c["rank"], lr=c["lr"], targets=c["targets"], pairs=120_000)
        if len(a.tasks) == 2 and len(a.folds) == 5:  # held-out scoring needs every fold's adapters for both tasks
            evaluate("blip-lora", a, "eval_blip.py", "--vqa-adapter", RUNS / "blip-vqa-f{fold}" / "best",
                     "--caption-adapter", RUNS / "blip-caption-f{fold}" / "best")
    if want("robustness"):
        evaluate("robust-baseline", a, "eval_blip.py", "--suite", "robustness", "--folds", 0, marker="f0-rot90-None.json")
        evaluate("robust-blip-lora", a, "eval_blip.py", "--suite", "robustness", "--folds", 0,
                 "--vqa-adapter", RUNS / "blip-vqa-f0" / "best", "--caption-adapter", RUNS / "blip-caption-f0" / "best",
                 marker="f0-rot90-None.json")
    if want("heatmaps") and not fetch("heatmaps", "pointing.json"):
        sh(sys.executable, CODE / "scripts" / "heatmaps.py", "--stats", a.stats, "--root", a.root,
           "--adapter", RUNS / "blip-vqa-f0" / "best", "--out", RUNS / "heatmaps")
        hub.push(RUNS / "heatmaps", "heatmaps")
    if want("leakage"):
        c = best.get("vqa") or final(sweep(a, "vqa")["choice"])
        train_blip("leak-vqa-official", a, "vqa", split="official", rank=c["rank"], lr=c["lr"], targets=c["targets"], pairs=120_000)
        evaluate("leakage", a, "eval_blip.py", "--suite", "leakage", "--vqa-adapter", RUNS / "leak-vqa-official" / "best",
                 marker="unseen-field.json")
    if want("dora"):
        c = best.get("vqa") or final(sweep(a, "vqa")["choice"])
        train_blip("blip-vqa-f0-dora", a, "vqa", fold=0, rank=c["rank"], lr=c["lr"], targets=c["targets"], dora=True, pairs=120_000)
        evaluate("dora", a, "eval_blip.py", "--folds", 0, "--vqa-adapter", RUNS / "blip-vqa-f0-dora" / "best",
                 "--caption-adapter", RUNS / "blip-caption-f0" / "best", marker="f0.json")
    if want("served"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        extra = ["--vision"] if b["merged_route"] else []
        for cycle in range(1, SERVED_CYCLES + 1):
            cmds = []
            for name, flags in SERVED_VARIANTS:
                run = served_run(name, cycle)
                if fetch(run):
                    continue
                warm = [] if cycle == 1 else ["--resume", hub.pull(served_run(name, cycle - 1), RUNS) / "best"]
                eval_flags = [] if cycle == 1 else ["--eval-every", SERVED_EVAL_EVERY, "--val-tiles", SERVED_VAL_TILES]
                cmds.append([sys.executable, "-m", "farmvlm.train_vlm", "--model", b["winner"], "--stats", a.stats, "--root", a.root,
                             "--pairs", SERVED_PAIRS, *eval_flags, "--out", RUNS / run, *extra, *flags, *(resume_args(run) or warm)])
            sh_parallel(cmds)
        for name, flags in SERVED_VARIANTS:
            evaluate(f"{name}-eval", a, "eval_vlm.py", "--model", b["winner"], "--adapter", RUNS / served_run(name) / "best",
                     *flags, marker="f0.json")
        evaluate("served-zeroshot-eval", a, "eval_vlm.py", "--model", b["winner"], marker="f0.json")
    if want("export"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for name, _ in SERVED_VARIANTS:
            out = RUNS / f"export-{name}"
            if not fetch(f"export-{name}", "lora-f16.gguf" if not b["merged_route"] else "model-Q4_K_M.gguf"):
                sh(sys.executable, CODE / "scripts" / "export_gguf.py", "--model", b["winner"], "--adapter", RUNS / served_run(name) / "best",
                   "--out", out, *(["--merged"] if b["merged_route"] else []))
                hub.push(out, f"export-{name}", include=("*.gguf",))
    if want("served-robustness"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for name, flags in SERVED_VARIANTS:
            evaluate(f"robust-{name}", a, "eval_vlm.py", "--model", b["winner"], "--adapter", RUNS / served_run(name) / "best",
                     "--suite", "robustness", *flags, marker="f0-rot90-None.json")


if __name__ == "__main__":
    main()
