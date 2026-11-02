"""Kaggle (GPU T4 x2) and molab (one 96 GB Blackwell) entry point: two pipelines over disjoint stages in parallel,
one per GPU, or both on one card with room for them, so no card idles.

    python3 scripts/kaggle.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021

The notebook cell only fetches the code and runs this, so what a run does is set here, in the repo.
Logs: pipeline-<N>.out in /kaggle/working on Kaggle, /content on molab, one per pipeline.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import CODE, visible_gpus  # noqa: E402

KAGGLE = Path("/kaggle/working").exists()
LOGS = Path("/kaggle/working" if KAGGLE else "/content")
GROUP_MIB = 20_000  # a T4-sized pipeline peaks under 16 GB

# Pipeline arguments per GPU slot, run in parallel; with fewer slots than groups, the groups queue on the slots there are.
# What remains after Kaggle session 1, split by machine so the two can run at once on disjoint runs. molab extracts
# only the RGB images (its sandbox died extracting all 21 GB), so it takes every RGB stage: the BLIP caption folds 3-4,
# the five-fold scoring, fold-0 robustness and balanced scoring beside Qwen RGB's scoring, export and stress test.
# Kaggle takes what needs NIR or the masks: Qwen NDVI cycle 2 (session 1 was cancelled before its first checkpoint)
# with its scoring, export and stress test, and the heatmaps. Colab is off.
GROUPS = ((["--only", "served", "export", "served-robustness", "--variants", "served-ndvi"], ["--only", "heatmaps"]) if KAGGLE
          else (["--only", "blip", "robustness", "balanced"],
                ["--only", "served", "export", "served-robustness", "--variants", "served-rgb"]))


def slots() -> list[str]:
    """The GPU of each parallel pipeline: one per card, or every group on a single card with memory for them all."""
    gpus = visible_gpus()
    if len(gpus) == 1:
        mib = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()[0])
        if mib >= GROUP_MIB * len(GROUPS):
            return gpus * len(GROUPS)
    return gpus


def main():
    gpus = slots()
    queues = {k: (gpu, []) for k, gpu in enumerate(gpus)}
    for k, group in enumerate(GROUPS):
        queues[k % len(gpus)][1].append(group)
    LOGS.mkdir(parents=True, exist_ok=True)
    procs = []
    for k, (gpu, groups) in queues.items():
        log = open(LOGS / f"pipeline-{k}.out", "w")
        cmds = [[sys.executable, "-u", str(CODE / "scripts" / "pipeline.py"), *sys.argv[1:], *g] for g in groups]
        print(f"gpu {gpu}: " + " && ".join(" ".join(g) for g in groups), flush=True)
        procs.append(subprocess.Popen(" && ".join(shlex.join(c) for c in cmds), shell=True, cwd=CODE, stdout=log,
                                      stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}))
    codes = [p.wait() for p in procs]
    print("exit codes:", codes, flush=True)
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
