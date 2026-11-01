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
# Second session of 2026-11-01 (start it only after the first has ended): GPU 0 finishes Qwen NDVI cycle 2 and scores,
# exports and stress-tests it; GPU 1 trains whichever BLIP folds are left (caption 3-4 unless Colab finished them; stop
# the Colab keepalive first so no run is on two machines), scores all five folds, then scores, exports and stress-tests
# Qwen RGB, whose training the first session finished. Stages already on the Hub are skipped.
GROUPS = (["--only", "served", "export", "served-robustness", "--variants", "served-ndvi"],
          ["--only", "blip", "robustness", "heatmaps", "balanced", "served", "export", "served-robustness",
           "--variants", "served-rgb"])


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
