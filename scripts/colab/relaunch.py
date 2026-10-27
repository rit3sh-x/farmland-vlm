"""Runs inside the Colab VM (via `colab exec`): set the session up, then start the pipeline detached.
Environment: HF_TOKEN (Kaggle-style secret, never written anywhere), STAGES (pipeline arguments after --only)."""
import os
import subprocess

assert os.environ.get("HF_TOKEN"), "HF_TOKEN missing in the kernel"
stages = os.environ.get("STAGES", "blip")
cmd = ("curl -sL https://raw.githubusercontent.com/rit3sh-x/farmland-vlm/main/scripts/session_setup.py | python3 - >> /content/setup.out 2>&1; "
       "[ -f /content/av/.done ] && cd /content/farmland-vlm && git fetch -q origin && git checkout -q origin/main && "
       f"python3 scripts/pipeline.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 --only {stages} >> /content/pipeline.out 2>&1")
print("launched", subprocess.Popen(cmd, shell=True, start_new_session=True).pid, "|", stages)
