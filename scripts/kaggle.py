"""Multi-GPU entry point for Kaggle (T4 x2), molab (one 96 GB Blackwell) and the svm-ws workstation (one 48 GB
Blackwell): pipelines over disjoint stages in parallel, one per GPU or several on one card with room for them.

    [FARMVLM_MACHINE=workstation] python3 scripts/kaggle.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021

The notebook cell only fetches the code and runs this, so what a run does is set here, in the repo.
Logs: pipeline-<N>.out in /kaggle/working on Kaggle, else in FARMVLM_HOME, one per GPU slot.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import CODE, visible_gpus  # noqa: E402
from farmvlm.hub import HOME  # noqa: E402

MACHINE = os.environ.get("FARMVLM_MACHINE") or ("kaggle" if Path("/kaggle/working").exists() else "molab")
LOGS = Path("/kaggle/working") if MACHINE == "kaggle" else HOME
GROUP_MIB = 16_000  # a T4-sized pipeline peaks under 16 GB (BLIP base 8 GB, Qwen3.5-2B under 15 GB)

# Pipeline arguments per group; groups run in parallel, one per GPU slot, and queue when there are more groups than slots.
# Kaggle and molab split what remained after Kaggle session 1 so both could run at once on disjoint runs: molab extracts
# only the RGB images (its sandbox died extracting all 21 GB) and takes every RGB stage; Kaggle takes what needs NIR or
# the masks. The workstation (from 2026-11-04, Colab off) takes everything left on three slots of its 48 GB card: the
# BLIP scoring and suites, then a third of the study; another third of the study, then the last; Qwen RGB scoring and
# export, then Qwen NDVI cycle 2. One BLIP run keeps the GPU only about half busy (kernel launches bound it), so
# parallel runs, not a bigger batch, use the rest; the batch stays part of the recipe.
GROUPS = {
    "kaggle": (["--only", "served", "export", "served-robustness", "--variants", "served-ndvi"], ["--only", "heatmaps"]),
    "molab": (["--only", "blip", "robustness", "balanced"],
              ["--only", "served", "export", "served-robustness", "--variants", "served-rgb"]),
    "workstation": (["--only", "blip", "robustness", "heatmaps", "balanced"],
                    ["--only", "study", "--study-part", "0/3"],
                    ["--only", "served", "export", "served-robustness", "--variants", "served-rgb"],
                    ["--only", "study", "--study-part", "1/3"],
                    ["--only", "study", "--study-part", "2/3"],
                    ["--only", "served", "export", "served-robustness", "--variants", "served-ndvi"]),
}[MACHINE]


def slots() -> list[str]:
    """The GPU of each parallel pipeline: one per card, or as many on a single card as its memory holds."""
    gpus = visible_gpus()
    if len(gpus) == 1:
        mib = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True).stdout.split()[0])
        return gpus * max(1, min(len(GROUPS), mib // GROUP_MIB))
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
