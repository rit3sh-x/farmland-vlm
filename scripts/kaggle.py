"""Kaggle entry point (GPU T4 x2): two pipelines over disjoint stages, one per GPU, so neither card idles.

    python3 scripts/kaggle.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021

The notebook cell only fetches the code and runs this, so what a Kaggle run does is set here, in the repo.
Logs: /kaggle/working/pipeline-gpu<N>.out.
"""
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import CODE, visible_gpus  # noqa: E402

GROUPS = (["bakeoff", "served", "export", "served-robustness"],
          ["baseline", "sweep", "blip", "robustness", "heatmaps", "leakage", "dora"])


def main():
    gpus = visible_gpus()
    groups = GROUPS if len(gpus) >= len(GROUPS) else [sum(GROUPS, [])]
    procs = []
    for gpu, stages in zip(gpus, groups):
        log = open(f"/kaggle/working/pipeline-gpu{gpu}.out", "w")
        print(f"gpu {gpu}: {' '.join(stages)}", flush=True)
        procs.append(subprocess.Popen([sys.executable, "-u", CODE / "scripts" / "pipeline.py", *sys.argv[1:], "--only", *stages],
                                      cwd=CODE, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu}))
    codes = [p.wait() for p in procs]
    print("exit codes:", codes, flush=True)
    sys.exit(max(codes))


if __name__ == "__main__":
    main()
