import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from farmvlm import hub
from farmvlm.report import chosen_cycle, final_run

RUNS = hub.RUNS
CODE = Path(__file__).resolve().parent.parent
SWEEP = {"rank": [4, 8, 16, 32, 64], "lr": [5e-5, 1e-4, 2e-4, 5e-4], "targets": ["text_attn", "attn_qv", "all_linear"]}
START = {"rank": 16, "lr": 1e-4, "targets": "attn_qv"}
FINAL_LR_CAP = 2e-4
STUDY_REF = {"rank": 64, "lr": FINAL_LR_CAP, "targets": "attn_qv"}
VQA, BOTH = ("vqa",), ("vqa", "caption")
STUDY = [("ref", {}, BOTH), ("size512", {"size": 512}, BOTH), ("rank128", {"rank": 128}, BOTH),
         ("nrg", {"bands": "nrg"}, BOTH), ("large", {"base": "large"}, ("caption",)),
         ("all_linear-lr1e-4", {"targets": "all_linear", "lr": 1e-4}, BOTH), ("rank256", {"rank": 256}, VQA),
         ("all_linear-lr2e-4", {"targets": "all_linear"}, VQA), ("norms", {"train_norms": True}, VQA),
         ("alpha1x", {"alpha": 64}, VQA), ("alpha4x", {"alpha": 256}, VQA),
         ("dropout0", {"dropout": 0.0}, VQA), ("dropout0.1", {"dropout": 0.1}, VQA), ("dropout0.2", {"dropout": 0.2}, VQA),
         ("pairs60k", {"pairs": 60_000}, VQA), ("pairs120k", {"pairs": 120_000}, VQA),
         ("pairs240k", {"pairs": 240_000}, VQA), ("pairs480k", {"pairs": 480_000}, VQA)]
STUDY2 = [("ref-seed1", {"train_seed": 1}, VQA), ("ref-seed2", {"train_seed": 2}, VQA),
          ("rank256-pairs120k", {"rank": 256, "pairs": 120_000}, VQA), ("rank256-seed1", {"rank": 256, "train_seed": 1}, VQA),
          ("all_linear-lr2e-4-pairs120k", {"targets": "all_linear", "pairs": 120_000}, VQA),
          ("pairs120k-seed1", {"pairs": 120_000, "train_seed": 1}, VQA),
          ("pairs480k", {"pairs": 480_000}, ("caption",))]
STUDY_PAIRS, STUDY_VAL_TILES = 30_000, 1000
CYCLE3 = {**STUDY_REF, "pairs": 240_000}
CYCLE4 = {**STUDY_REF, "pairs": 480_000}
CYCLE5 = {**STUDY_REF, "pairs": 960_000}
CYCLE6 = {**STUDY_REF, "pairs": 1_920_000}
CYCLE7 = {**STUDY_REF, "pairs": 3_840_000}
SERVED_PAIRS = 30_000
SERVED_CYCLES = 2
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
    return name if cycle == 1 else f"{name}-c{cycle}"


def fetch(name: str, marker: str = "done.json") -> bool:
    if (RUNS / name / marker).exists():
        return True
    if hub.has(name, marker):
        hub.pull(name, RUNS, last=False)
        return True
    return False


def adapter(name: str) -> Path:
    assert fetch(name), f"{name} is not finished here or on the Hub"
    return RUNS / name / "best"


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
        hub.pull(name, RUNS)
        sh(sys.executable, CODE / "scripts" / script, "--stats", a.stats, "--root", a.root, "--out", RUNS / name, *args)
        hub.push(RUNS / name, name)


def sweep(a, task):
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
    ap.add_argument("--variants", nargs="*", default=[n for n, _ in SERVED_VARIANTS], choices=[n for n, _ in SERVED_VARIANTS],
                    help="served variants to train, score, export and stress-test (one per GPU when there are enough)")
    a = ap.parse_args()
    variants = [v for v in SERVED_VARIANTS if v[0] in a.variants]
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
                train_blip(f"blip-{t}-f{f}", a, t, fold=f, rank=c["rank"], lr=c["lr"], targets=c["targets"], pairs=120_000, patience=99)
        if len(a.tasks) == 2 and len(a.folds) == 5:  # held-out scoring needs every fold's adapters for both tasks
            evaluate("blip-lora", a, "eval_blip.py", "--vqa-adapter", RUNS / "blip-vqa-f{fold}" / "best",
                     "--caption-adapter", RUNS / "blip-caption-f{fold}" / "best")
    if want("robustness"):
        evaluate("robust-baseline", a, "eval_blip.py", "--suite", "robustness", "--folds", 0, marker="f0-rot90-None.json")
        evaluate("robust-blip-lora", a, "eval_blip.py", "--suite", "robustness", "--folds", 0,
                 "--vqa-adapter", adapter("blip-vqa-f0"), "--caption-adapter", adapter("blip-caption-f0"),
                 marker="f0-rot90-None.json")
    if want("heatmaps") and not fetch("heatmaps", "pointing.json"):
        sh(sys.executable, CODE / "scripts" / "heatmaps.py", "--stats", a.stats, "--root", a.root,
           "--adapter", adapter("blip-vqa-f0"), "--out", RUNS / "heatmaps")
        hub.push(RUNS / "heatmaps", "heatmaps")
    if want("leakage"):
        c = best.get("vqa") or final(sweep(a, "vqa")["choice"])
        train_blip("leak-vqa-official", a, "vqa", split="official", rank=c["rank"], lr=c["lr"], targets=c["targets"], pairs=120_000)
        evaluate("leakage", a, "eval_blip.py", "--suite", "leakage", "--vqa-adapter", adapter("leak-vqa-official"),
                 marker="unseen-field.json")
    if want("dora"):
        c = best.get("vqa") or final(sweep(a, "vqa")["choice"])
        train_blip("blip-vqa-f0-dora", a, "vqa", fold=0, rank=c["rank"], lr=c["lr"], targets=c["targets"], dora=True, pairs=120_000)
        evaluate("dora", a, "eval_blip.py", "--folds", 0, "--vqa-adapter", adapter("blip-vqa-f0-dora"),
                 "--caption-adapter", adapter("blip-caption-f0"), marker="f0.json")
    if want("balanced"):  # the location-balanced sampler, fold 0: an ablation of the final VQA recipe
        c = best.get("vqa") or final(sweep(a, "vqa")["choice"])
        train_blip("blip-vqa-f0-balanced", a, "vqa", fold=0, rank=c["rank"], lr=c["lr"], targets=c["targets"], pairs=120_000,
                   patience=99, location_cell_share=0.65)
        evaluate("balanced", a, "eval_blip.py", "--folds", 0, "--vqa-adapter", adapter("blip-vqa-f0-balanced"),
                 "--caption-adapter", adapter("blip-caption-f0"), marker="f0.json")
    for stage, trials in (("study", STUDY), ("study2", STUDY2)):
        for trial, change, tasks in trials if want(stage) else []:
            for t in tasks:
                kw = {**STUDY_REF, "pairs": STUDY_PAIRS, **change}
                train_blip(f"study-{t}/{trial}", a, t, fold=0, **kw, eval_every=max(10_000, kw["pairs"] // 6),
                           val_tiles=STUDY_VAL_TILES, patience=99)
    if want("study"):
        evaluate("study-decode-beams3", a, "eval_blip.py", "--folds", 0, "--beams", 3, "--vqa-adapter", adapter("blip-vqa-f0"),
                 "--caption-adapter", adapter("blip-caption-f0"), marker="f0.json")
    if want("captions4"):
        for f in a.folds:
            train_blip(f"c4-caption-f{f}", a, "caption", fold=f, **CYCLE4, eval_every=80_000, val_tiles=STUDY_VAL_TILES,
                       val_protocol="test", patience=99)
    for c, recipe in (("c3", CYCLE3), ("c4", CYCLE4), ("c5", CYCLE5), ("c6", CYCLE6), ("c7", CYCLE7)):
        if not want(f"cycle{c[1]}"):
            continue
        for f in a.folds:
            train_blip(f"{c}-vqa-f{f}", a, "vqa", fold=f, **recipe, eval_every=recipe["pairs"] // 6, val_tiles=STUDY_VAL_TILES,
                       val_protocol="test", select="ranked", patience=99)
        evaluate(f"{c}-blip", a, "eval_blip.py", "--folds", *a.folds, "--vqa-adapter", RUNS / f"{c}-vqa-f{{fold}}" / "best",
                 "--caption-adapter", RUNS / "blip-caption-f{fold}" / "best", marker=f"f{a.folds[-1]}.json")
    if want("final"):
        c = chosen_cycle(RUNS)
        evaluate(f"robust-c{c}", a, "eval_blip.py", "--suite", "robustness", "--folds", 0, "--vqa-adapter", adapter(f"c{c}-vqa-f0"),
                 "--caption-adapter", adapter("c4-caption-f0"), marker="f0-rot90-None.json")
        if not fetch(f"heatmaps-c{c}", "pointing.json"):
            sh(sys.executable, CODE / "scripts" / "heatmaps.py", "--stats", a.stats, "--root", a.root,
               "--adapter", adapter(f"c{c}-vqa-f0"), "--out", RUNS / f"heatmaps-c{c}")
            hub.push(RUNS / f"heatmaps-c{c}", f"heatmaps-c{c}")
        evaluate(final_run(RUNS), a, "eval_blip.py", "--folds", *a.folds, "--vqa-adapter", RUNS / f"c{c}-vqa-f{{fold}}" / "best",
                 "--caption-adapter", RUNS / "c4-caption-f{fold}" / "best", marker=f"f{a.folds[-1]}.json")
        evaluate(f"train-c{c}", a, "eval_blip.py", "--suite", "train", "--folds", 0, "--vqa-adapter", adapter(f"c{c}-vqa-f0"),
                 "--caption-adapter", adapter("c4-caption-f0"), marker="train-f0.json")
    for c, recipe, stage in (("c4", CYCLE4, "augment"), ("c6", CYCLE6, "augment6")):
        if not want(stage):
            continue
        for f in a.folds if c == "c6" else [0]:
            train_blip(f"aug-{c}-vqa-f{f}", a, "vqa", fold=f, **recipe, augment=0.5, eval_every=recipe["pairs"] // 6,
                       val_tiles=STUDY_VAL_TILES, val_protocol="test", select="ranked", patience=99)
            evaluate(f"aug-{c}-blip", a, "eval_blip.py", "--folds", f, "--vqa-adapter", adapter(f"aug-{c}-vqa-f{f}"),
                     "--caption-adapter", adapter(f"c4-caption-f{f}"), marker=f"f{f}.json")
        if 0 not in a.folds:
            continue
        evaluate(f"robust-aug-{c}", a, "eval_blip.py", "--suite", "robustness", "--folds", 0, "--vqa-adapter",
                 adapter(f"aug-{c}-vqa-f0"), "--caption-adapter", adapter("c4-caption-f0"), marker="f0-rot90-None.json")
        evaluate(f"train-aug-{c}", a, "eval_blip.py", "--suite", "train", "--folds", 0, "--vqa-adapter", adapter(f"aug-{c}-vqa-f0"),
                 "--caption-adapter", adapter("c4-caption-f0"), marker="train-f0.json")  # does augmentation narrow the fit gap?
    if want("served"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for cycle in range(1, SERVED_CYCLES + 1):
            cmds = []
            for name, flags in variants:
                run = served_run(name, cycle)
                if fetch(run):
                    continue
                warm = [] if cycle == 1 else ["--resume", hub.pull(served_run(name, cycle - 1), RUNS) / "best"]
                eval_flags = [] if cycle == 1 else ["--eval-every", SERVED_EVAL_EVERY, "--val-tiles", SERVED_VAL_TILES]
                cmds.append([sys.executable, "-m", "farmvlm.train_vlm", "--model", b["winner"], "--stats", a.stats, "--root", a.root,
                             "--pairs", SERVED_PAIRS, *eval_flags, "--out", RUNS / run, *flags, *(resume_args(run) or warm)])
            sh_parallel(cmds)
        for name, flags in variants:
            evaluate(f"{name}-eval", a, "eval_vlm.py", "--model", b["winner"], "--adapter", adapter(served_run(name)),
                     *flags, marker="f0.json")
        evaluate("served-zeroshot-eval", a, "eval_vlm.py", "--model", b["winner"], marker="f0.json")
    if want("export"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for name, _ in variants:
            out = RUNS / f"export-{name}"
            if not fetch(f"export-{name}", "lora-f16.gguf"):
                sh(sys.executable, CODE / "scripts" / "export_gguf.py", "--model", b["winner"], "--adapter", adapter(served_run(name)),
                   "--out", out)
                hub.push(out, f"export-{name}", include=("*.gguf",))
    if want("served-robustness"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for name, flags in variants:
            evaluate(f"robust-{name}", a, "eval_vlm.py", "--model", b["winner"], "--adapter", adapter(served_run(name)),
                     "--suite", "robustness", *flags, marker="f0-rot90-None.json")


if __name__ == "__main__":
    main()
