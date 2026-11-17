import argparse
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TAR = "https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz"
MANIFEST = "https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/neurips_benchmarks_2021/data_manifest.csv"
UA = {"User-Agent": "farmland-vlm/0.1"}
EXPECTED_TILES = {"train": 56944, "val": 18334}
PARTS = ("images/rgb", "images/nir", "labels", "boundaries", "masks")


def fetch(dest: Path):
    done = dest / ".done"
    if done.exists():
        print(f"{dest}: already extracted")
        return
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    (dest / "data_manifest.csv").write_bytes(
        urllib.request.urlopen(urllib.request.Request(MANIFEST, headers=UA), timeout=60).read())
    tar = subprocess.Popen(["tar", "-xz", "-C", str(dest), "--wildcards",
                            *(f"Agriculture-Vision-2021/{s}/{p}/*" for s in EXPECTED_TILES for p in PARTS)], stdin=subprocess.PIPE)
    t0, n = time.time(), 0
    with urllib.request.urlopen(urllib.request.Request(TAR, headers=UA), timeout=120) as r:
        while chunk := r.read(8 << 20):
            tar.stdin.write(chunk)
            n += len(chunk)
            if n % (2 << 30) < len(chunk):
                print(f"{n / 1e9:.0f} GB downloaded", flush=True)
    tar.stdin.close()
    if tar.wait() != 0:
        sys.exit(f"tar failed after {n / 1e9:.2f} GB")
    root = dest / "Agriculture-Vision-2021"
    counts = {s: sum(1 for _ in (root / s / "images" / "rgb").glob("*.jpg")) for s in EXPECTED_TILES}
    if counts != EXPECTED_TILES:
        sys.exit(f"unexpected tile counts {counts}")
    done.write_text(f"{n} bytes in {time.time() - t0:.0f} s\n")
    print(f"{n / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min; tiles {counts}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download and extract Agriculture-Vision 2021 (21 GB).")
    ap.add_argument("dest", type=Path, nargs="?", default=Path("data/av"))
    fetch(ap.parse_args().dest)
