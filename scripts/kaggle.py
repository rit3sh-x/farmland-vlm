"""Kaggle entry point (GPU T4 x2): two pipelines over disjoint stages, one per GPU, so neither card idles.

    python3 scripts/kaggle.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021

The notebook cell only fetches the code and runs this, so what a Kaggle run does is set here, in the repo.
Logs: /kaggle/working/pipeline-gpu<N>.out.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import CODE, visible_gpus  # noqa: E402

# Pipeline arguments per GPU, run in parallel; with fewer GPUs than groups, the groups queue on the GPUs there are.
# Week of 2026-10-26 (about 3 quota hours left): finish caption fold 0 v2 and train VQA fold 0 v2 as the v2 checks.
GROUPS = (["--only", "blip", "--tasks", "caption", "--folds", "0"],
          ["--only", "blip", "--tasks", "vqa", "--folds", "0"])


def main():
    gpus = visible_gpus()
    queues = {gpu: [] for gpu in gpus}
    for k, group in enumerate(GROUPS):
        queues[gpus[k % len(gpus)]].append(group)
    procs = []
    for gpu, groups in queues.items():
        log = open(f"/kaggle/working/pipeline-gpu{gpu}.out", "w")
        cmds = [[sys.executable, "-u", str(CODE / "scripts" / "pipeline.py"), *sys.argv[1:], *g] for g in groups]
        print(f"gpu {gpu}: " + " && ".join(" ".join(g) for g in groups), flush=True)
        procs.append(subprocess.Popen(" && ".join(shlex.join(c) for c in cmds), shell=True, cwd=CODE, stdout=log,
                                      stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}))
    codes = [p.wait() for p in procs]
    print("exit codes:", codes, flush=True)
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
