"""Bring a fresh Colab VM to a ready state: code, packages, data, stats. Safe to rerun at any point.

    curl -sL https://raw.githubusercontent.com/rit3sh-x/farmland-vlm/main/scripts/session_setup.py | python3 - [--commit SHA]
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/rit3sh-x/farmland-vlm.git"
CODE, DATA, STATS = Path("/content/farmland-vlm"), Path("/content/av"), Path("/content/stats.npz")


def sh(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", default="main")
    ap.add_argument("--no-data", action="store_true")
    a = ap.parse_args()
    if not CODE.exists():
        sh("git", "clone", "-q", REPO, CODE)
    sh("git", "-C", CODE, "fetch", "-q", "origin")
    sh("git", "-C", CODE, "checkout", "-q", a.commit if a.commit != "main" else "origin/main")
    sh(sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao")
    sh(sys.executable, "-m", "pip", "install", "-q", "-e", CODE)
    if a.no_data:
        return
    sh(sys.executable, CODE / "scripts" / "fetch_data.py", DATA)
    if not STATS.exists():
        sh(sys.executable, "-m", "farmvlm.stats", "--root", DATA / "Agriculture-Vision-2021",
           "--manifest", DATA / "data_manifest.csv", "--out", STATS)


if __name__ == "__main__":
    main()
