"""Every GPU run of the project, graded BLIP work first, then the served model (issues/16).
Idempotent: finished stages are skipped (locally or on the Hub), an interrupted training run resumes
from its last pushed checkpoint.

    HF_TOKEN=... python scripts/pipeline.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021
    python scripts/pipeline.py ... --only baseline sweep

Stages: baseline sweep bakeoff blip robustness heatmaps leakage dora balanced study study2 captions4 cycle3-cycle6 final augment served
export served-robustness.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm import hub  # noqa: E402
from farmvlm.report import chosen_cycle, final_run  # noqa: E402

RUNS = hub.RUNS
CODE = Path(__file__).resolve().parent.parent
SWEEP = {"rank": [4, 8, 16, 32, 64], "lr": [5e-5, 1e-4, 2e-4, 5e-4], "targets": ["text_attn", "attn_qv", "all_linear"]}
START = {"rank": 16, "lr": 1e-4, "targets": "attn_qv"}
# The 10k-pair sweep anneals its LR within 10k pairs; held near 5e-4 for longer, BLIP VQA fold 0 collapsed
# (identify 0.34 -> 0.00 by 20k pairs). Final runs take the sweep choice with the LR capped here; the sweep
# scored 2e-4 within a point of 5e-4.
FINAL_LR_CAP = 2e-4
# The study (evaluator review, 2026-11-04): one change at a time from the final recipe on fold 0, 30k pairs unless a
# trial says otherwise, scored on 1,000 inner-validation tiles (fixed by the seed, so every trial sees the same items).
# The 10k-pair sweep stopped at its top rank while still rising, never varied alpha, dropout, LayerNorm, input size or
# bands, and tried all_linear only at an LR that derails long runs; 300 validation tiles left 1-3 point effects inside the
# noise. Trials run most informative first; captions only where a change could plausibly move them. Selection happens
# on validation only: the held-out folds score the final recipe and never choose it.
STUDY_REF = {"rank": 64, "lr": FINAL_LR_CAP, "targets": "attn_qv"}
VQA, BOTH = ("vqa",), ("vqa", "caption")
# The memory-heavy trials (size512 peaks at 19 GB, against 8 GB at 384 px; BLIP-large) share a third of the list, so the
# three parallel parts (--study-part k/3 takes every third trial) never run them at once beside Qwen training.
STUDY = [("ref", {}, BOTH), ("size512", {"size": 512}, BOTH), ("rank128", {"rank": 128}, BOTH),
         ("nrg", {"bands": "nrg"}, BOTH), ("large", {"base": "large"}, ("caption",)),
         ("all_linear-lr1e-4", {"targets": "all_linear", "lr": 1e-4}, BOTH), ("rank256", {"rank": 256}, VQA),
         ("all_linear-lr2e-4", {"targets": "all_linear"}, VQA), ("norms", {"train_norms": True}, VQA),
         ("alpha1x", {"alpha": 64}, VQA), ("alpha4x", {"alpha": 256}, VQA),
         ("dropout0", {"dropout": 0.0}, VQA), ("dropout0.1", {"dropout": 0.1}, VQA), ("dropout0.2", {"dropout": 0.2}, VQA),
         ("pairs60k", {"pairs": 60_000}, VQA), ("pairs120k", {"pairs": 120_000}, VQA),
         ("pairs240k", {"pairs": 240_000}, VQA), ("pairs480k", {"pairs": 480_000}, VQA)]
# Round 2, from the first results: two training-seed replicates of the reference measure run-to-run noise, which the
# field bootstrap does not cover; rank 256 (+1.3 at 30k) at the cycle-2 length checks the gain survives longer training.
# Then, since the three reference seeds spread 56.5 to 58.2: rank 256 under a second seed; all_linear at LR 2e-4 (the
# best 30k trial) at 120k, since every capacity knob gained at 30k and the length test tells faster convergence from a
# higher ceiling; and the 120k trial under a second seed (the noise both 120k knobs are read against).
STUDY2 = [("ref-seed1", {"train_seed": 1}, VQA), ("ref-seed2", {"train_seed": 2}, VQA),
          ("rank256-pairs120k", {"rank": 256, "pairs": 120_000}, VQA), ("rank256-seed1", {"rank": 256, "train_seed": 1}, VQA),
          ("all_linear-lr2e-4-pairs120k", {"targets": "all_linear", "pairs": 120_000}, VQA),
          ("pairs120k-seed1", {"pairs": 120_000, "train_seed": 1}, VQA),
          # Captions: cycle 2's 120k caption run was still near its best at the end (fact F1 72 at 30k, ~83.5 at 100k on
          # 300 tiles). Read against blip-caption-f0 on the same 1,000 validation tiles (eval_blip --suite validation).
          ("pairs480k", {"pairs": 480_000}, ("caption",))]
STUDY_PAIRS, STUDY_VAL_TILES = 30_000, 1000
# Cycle 3, from the study: the cycle-2 recipe at 240k pairs. Length was the one change that raised the ceiling (+1.1 over
# 120k on validation); rank, alpha, all_linear, dropout, LayerNorm, 512 px and NRG did not at length or beyond seed noise.
# Checks score the ranked readout (chosen on validation) on a validation draw made by the held-out sample's rules.
CYCLE3 = {**STUDY_REF, "pairs": 240_000}
# Cycle 4: the 480k run then validated +1.4 over 240k (65.1 against 63.7), the curve still rising. Cycle 3 keeps
# running, so the length curve is read on the held-out folds too (cycle 2 is its 120k point).
CYCLE4 = {**STUDY_REF, "pairs": 480_000}
# Cycle 5 probe: 960k pairs on fold 0 first. Doubling compute through length paid +1.9 at 240k -> 480k (fold 0), against
# +0.8 for averaging two seeds of one length; the other folds follow only if fold 0's validation rises again.
CYCLE5 = {**STUDY_REF, "pairs": 960_000}
# Cycle 6 probe: 1.92M pairs on fold 0. Cycle 5 led cycle 4 on every fold's validation (fold 0: 70.7 against 67.8).
CYCLE6 = {**STUDY_REF, "pairs": 1_920_000}
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
        hub.pull(name, RUNS, last=False)
        return True
    return False


def adapter(name: str) -> Path:
    """A finished run's best adapter, pulled from the Hub first when another machine trained it."""
    assert fetch(name), f"{name} is not finished on this VM or the Hub"
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
        hub.pull(name, RUNS)  # results an earlier VM pushed before it was recycled; the script skips them
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
    ap.add_argument("--study-part", default="0/1", help="k/n: this pipeline trains study trials k, k+n, k+2n, ... (parallel slots)")
    ap.add_argument("--variants", nargs="*", default=[n for n, _ in SERVED_VARIANTS], choices=[n for n, _ in SERVED_VARIANTS],
                    help="served variants to train, score, export and stress-test (one per GPU on Kaggle)")
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
                # The full budget, best second-half checkpoint kept: patience 3 stopped runs on validation noise (300 tiles).
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
    k, n = map(int, a.study_part.split("/"))
    for stage, trials in (("study", STUDY), ("study2", STUDY2)):
        for trial, change, tasks in trials[k::n] if want(stage) else []:
            for t in tasks:
                kw = {**STUDY_REF, "pairs": STUDY_PAIRS, **change}
                train_blip(f"study-{t}/{trial}", a, t, fold=0, **kw, eval_every=max(10_000, kw["pairs"] // 6),
                           val_tiles=STUDY_VAL_TILES, patience=99)
    if want("study"):
        # Decoding needs no training: the cycle-2 fold-0 adapters with beam search, against blip-lora/f0 (greedy)
        evaluate("study-decode-beams3", a, "eval_blip.py", "--folds", 0, "--beams", 3, "--vqa-adapter", adapter("blip-vqa-f0"),
                 "--caption-adapter", adapter("blip-caption-f0"), marker="f0.json")
    # Captions at 480k pairs: fold 0's probe (study-caption/pairs480k, copied to c4-caption-f0: training is deterministic
    # and validation never touches the weights) gained fact F1 +2.8 (81.1 against 78.3), BLEU-4 0.579 -> 0.591 and
    # ROUGE-L 0.742 -> 0.760 over cycle 2's caption adapter on the same validation tiles.
    if want("captions4"):
        for f in a.folds:
            train_blip(f"c4-caption-f{f}", a, "caption", fold=f, **CYCLE4, eval_every=80_000, val_tiles=STUDY_VAL_TILES,
                       val_protocol="test", patience=99)
    for c, recipe in (("c3", CYCLE3), ("c4", CYCLE4), ("c5", CYCLE5), ("c6", CYCLE6)):
        if not want(f"cycle{c[1]}"):
            continue
        for f in a.folds:
            train_blip(f"{c}-vqa-f{f}", a, "vqa", fold=f, **recipe, eval_every=recipe["pairs"] // 6, val_tiles=STUDY_VAL_TILES,
                       val_protocol="test", select="ranked", patience=99)
        # Captions keep cycle 2's adapters: the study tuned VQA, the primary metric.
        evaluate(f"{c}-blip", a, "eval_blip.py", "--folds", *a.folds, "--vqa-adapter", RUNS / f"{c}-vqa-f{{fold}}" / "best",
                 "--caption-adapter", RUNS / "blip-caption-f{fold}" / "best", marker=f"f{a.folds[-1]}.json")
    if want("final"):  # the cycle the validation rule chose, with the 480k caption adapters: robustness, heatmaps, held out
        c = chosen_cycle(RUNS)
        evaluate(f"robust-c{c}", a, "eval_blip.py", "--suite", "robustness", "--folds", 0, "--vqa-adapter", adapter(f"c{c}-vqa-f0"),
                 "--caption-adapter", adapter("c4-caption-f0"), marker="f0-rot90-None.json")
        if not fetch(f"heatmaps-c{c}", "pointing.json"):
            sh(sys.executable, CODE / "scripts" / "heatmaps.py", "--stats", a.stats, "--root", a.root,
               "--adapter", adapter(f"c{c}-vqa-f0"), "--out", RUNS / f"heatmaps-c{c}")
            hub.push(RUNS / f"heatmaps-c{c}", f"heatmaps-c{c}")
        evaluate(final_run(RUNS), a, "eval_blip.py", "--folds", *a.folds, "--vqa-adapter", RUNS / f"c{c}-vqa-f{{fold}}" / "best",
                 "--caption-adapter", RUNS / "c4-caption-f{fold}" / "best", marker=f"f{a.folds[-1]}.json")
    # Robustness by length (robust-c2..c5): the image drop grows with every doubling, and training damages no images. Cycle
    # 4's fold-0 recipe with half the training images damaged at random (the draws match c4-vqa-f0's), scored against it on
    # validation, held-out fold 0 and the robustness suite.
    if want("augment"):
        train_blip("aug-c4-vqa-f0", a, "vqa", fold=0, **CYCLE4, augment=0.5, eval_every=CYCLE4["pairs"] // 6,
                   val_tiles=STUDY_VAL_TILES, val_protocol="test", select="ranked", patience=99)
        evaluate("aug-c4-blip", a, "eval_blip.py", "--folds", 0, "--vqa-adapter", adapter("aug-c4-vqa-f0"),
                 "--caption-adapter", adapter("c4-caption-f0"), marker="f0.json")
        evaluate("robust-aug-c4", a, "eval_blip.py", "--suite", "robustness", "--folds", 0, "--vqa-adapter",
                 adapter("aug-c4-vqa-f0"), "--caption-adapter", adapter("c4-caption-f0"), marker="f0-rot90-None.json")
    if want("served"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        extra = ["--vision"] if b["merged_route"] else []
        for cycle in range(1, SERVED_CYCLES + 1):
            cmds = []
            for name, flags in variants:
                run = served_run(name, cycle)
                if fetch(run):
                    continue
                warm = [] if cycle == 1 else ["--resume", hub.pull(served_run(name, cycle - 1), RUNS) / "best"]
                eval_flags = [] if cycle == 1 else ["--eval-every", SERVED_EVAL_EVERY, "--val-tiles", SERVED_VAL_TILES]
                cmds.append([sys.executable, "-m", "farmvlm.train_vlm", "--model", b["winner"], "--stats", a.stats, "--root", a.root,
                             "--pairs", SERVED_PAIRS, *eval_flags, "--out", RUNS / run, *extra, *flags, *(resume_args(run) or warm)])
            sh_parallel(cmds)
        for name, flags in variants:
            evaluate(f"{name}-eval", a, "eval_vlm.py", "--model", b["winner"], "--adapter", adapter(served_run(name)),
                     *flags, marker="f0.json")
        evaluate("served-zeroshot-eval", a, "eval_vlm.py", "--model", b["winner"], marker="f0.json")
    if want("export"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for name, _ in variants:
            out = RUNS / f"export-{name}"
            if not fetch(f"export-{name}", "lora-f16.gguf" if not b["merged_route"] else "model-Q4_K_M.gguf"):
                sh(sys.executable, CODE / "scripts" / "export_gguf.py", "--model", b["winner"], "--adapter", adapter(served_run(name)),
                   "--out", out, *(["--merged"] if b["merged_route"] else []))
                hub.push(out, f"export-{name}", include=("*.gguf",))
    if want("served-robustness"):
        b = json.loads((RUNS / "bakeoff" / "bakeoff.json").read_text())
        for name, flags in variants:
            evaluate(f"robust-{name}", a, "eval_vlm.py", "--model", b["winner"], "--adapter", adapter(served_run(name)),
                     "--suite", "robustness", *flags, marker="f0-rot90-None.json")


if __name__ == "__main__":
    main()
