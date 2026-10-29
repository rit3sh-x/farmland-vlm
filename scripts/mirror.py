"""Mirror the run records of the private Hub repo (json and jsonl only, no weights) into .hubmirror/runs, laid out as the
Hub lays them out, so report.py and figures.py read the latest cycle: python scripts/mirror.py && python -m farmvlm.report --runs .hubmirror/runs"""
import argparse
import os
import pathlib
import shutil

from huggingface_hub import snapshot_download

from farmvlm.hub import REPO

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path(".hubmirror"))
a = ap.parse_args()
fresh = a.out.with_name(a.out.name + ".new")
shutil.rmtree(fresh, ignore_errors=True)
snapshot_download(REPO, allow_patterns=["runs/*.json", "runs/*.jsonl"], local_dir=str(fresh), token=os.environ.get("HF_TOKEN"))
shutil.rmtree(a.out, ignore_errors=True)
fresh.rename(a.out)
print(" ".join(sorted(p.name for p in (a.out / "runs").iterdir())))
print("archive/v1:", " ".join(sorted(p.name for p in (a.out / "runs" / "archive" / "v1").iterdir())) if (a.out / "runs" / "archive" / "v1").exists() else "none")
