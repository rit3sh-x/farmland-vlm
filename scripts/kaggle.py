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
# Weekend of 2026-10-31 (two 12 h sessions; everything resumes): GPU 0 trains Qwen cycle 2 (RGB, then NDVI) and scores,
# exports and stress-tests it; GPU 1 trains whichever v2 BLIP folds Colab has not finished, then scores all five folds,
# runs the robustness suite and the heatmaps on fold 0.
GROUPS = (["--only", "served", "export", "served-robustness"],
          ["--only", "blip", "robustness", "heatmaps"])


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
