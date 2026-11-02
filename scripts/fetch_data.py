"""Stream the official Agriculture-Vision 2021 tar into DEST (train + val, RGB + NIR + masks) and fetch the manifest.

Only the official link is used (dataset terms section 1). Idempotent: a completed extraction leaves DEST/.done
naming the parts it holds, and a rerun that asks for no more returns at once; otherwise the extraction is wiped and
restarted. Naming PARTS extracts only those: molab's sandbox died twice extracting all 21 GB, and RGB-only stages
need just images/rgb (8 GB); the masks feed only the statistics and the heatmaps.

    python scripts/fetch_data.py /content/av [PART ...]
"""
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


def fetch(dest: Path, parts=PARTS):
    assert "images/rgb" in parts and set(parts) <= set(PARTS), parts
    done = dest / ".done"
    have = done.read_text().splitlines()[1:2] if done.exists() else []
    if done.exists() and set(parts) <= set(have[0].split()[1:] if have else PARTS):
        print(f"{dest}: already extracted")
        return
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    (dest / "data_manifest.csv").write_bytes(
        urllib.request.urlopen(urllib.request.Request(MANIFEST, headers=UA), timeout=60).read())
    tar = subprocess.Popen(["tar", "-xz", "-C", str(dest), "--wildcards",
                            *(f"Agriculture-Vision-2021/{s}/{p}/*" for s in EXPECTED_TILES for p in parts)], stdin=subprocess.PIPE)
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
    done.write_text(f"{n} bytes in {time.time() - t0:.0f} s\nparts: {' '.join(parts)}\n")
    print(f"{n / 1e9:.2f} GB in {(time.time() - t0) / 60:.1f} min; tiles {counts}")


if __name__ == "__main__":
    fetch(Path(sys.argv[1] if len(sys.argv) > 1 else "/content/av"), tuple(sys.argv[2:]) or PARTS)
