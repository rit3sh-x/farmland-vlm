"""Kaggle (GPU T4 x2) and molab (one 96 GB Blackwell) entry point: two pipelines over disjoint stages in parallel,
one per GPU, or both on one card with room for them, so no card idles.

    python3 scripts/kaggle.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021

The notebook cell only fetches the code and runs this, so what a run does is set here, in the repo.
Logs: $FARMVLM_LOGS/pipeline-<N>.out (default /kaggle/working), one per pipeline.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import CODE, visible_gpus  # noqa: E402

LOGS = Path(os.environ.get("FARMVLM_LOGS", "/kaggle/working"))
GROUP_MIB = 20_000  # a T4-sized pipeline peaks under 16 GB

# Pipeline arguments per GPU slot, run in parallel; with fewer slots than groups, the groups queue on the slots there are.
# What remains after Kaggle session 1: the first group trains Qwen NDVI cycle 2 (session 1 was cancelled before its
# first checkpoint) and scores, exports and stress-tests it. The second finishes the BLIP caption folds 3-4 and the
# five-fold scoring, then runs the fold-0 robustness suite, heatmaps and balanced scoring, then scores, exports and
# stress-tests Qwen RGB. Colab is off; run these on one machine at a time (molab from 2026-11-03, or Kaggle), never both.
GROUPS = (["--only", "served", "export", "served-robustness", "--variants", "served-ndvi"],
          ["--only", "blip", "robustness", "heatmaps", "balanced", "served", "export", "served-robustness",
           "--variants", "served-rgb"])


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
