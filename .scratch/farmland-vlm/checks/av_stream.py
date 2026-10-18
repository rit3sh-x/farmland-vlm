# Runs on the Colab VM: probe S3 vs HF throughput, then one full S3 pass extracting train+val (no test).
import time, urllib.request, subprocess, json, os, sys
UA = {"User-Agent": "farmland-vlm-research/0.1"}
S3 = "https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz"
HF = "https://huggingface.co/datasets/shi-labs/Agriculture-Vision/resolve/main/Agriculture-Vision-2021.tar.gz"
LOG = open("/content/av_stream.log", "a", buffering=1)
def log(*a): print(time.strftime("%H:%M:%S"), *a, file=LOG)

def probe(name, url, secs=30):
    r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60)
    t0 = time.time(); n = 0
    while time.time() - t0 < secs:
        b = r.read(8 << 20)
        if not b: break
        n += len(b)
    r.close(); dt = time.time() - t0
    log(f"probe {name}: {n/1e6:.0f} MB in {dt:.1f}s = {n/1e6/dt:.1f} MB/s")

for name, url in (("s3", S3), ("hf", HF)):
    try: probe(name, url)
    except Exception as e: log(f"probe {name} failed: {e!r}")

os.makedirs("/content/av", exist_ok=True)
urllib.request.urlretrieve  # manifest is tiny; fetch with UA
m = urllib.request.urlopen(urllib.request.Request("https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/neurips_benchmarks_2021/data_manifest.csv", headers=UA), timeout=60).read()
open("/content/av/data_manifest.csv", "wb").write(m)

tar = subprocess.Popen(["tar", "-xz", "-C", "/content/av", "--wildcards",
                        "Agriculture-Vision-2021/train/*", "Agriculture-Vision-2021/val/*"], stdin=subprocess.PIPE)
r = urllib.request.urlopen(urllib.request.Request(S3, headers=UA), timeout=120)
t0 = time.time(); n = 0; last = t0
while True:
    b = r.read(8 << 20)
    if not b: break
    tar.stdin.write(b); n += len(b)
    if time.time() - last > 30:
        last = time.time(); log(f"stream {n/1e9:.2f} GB, {n/1e6/(last-t0):.1f} MB/s avg")
tar.stdin.close(); rc = tar.wait(); dt = time.time() - t0
log(f"stream done rc={rc}: {n/1e9:.2f} GB in {dt/60:.1f} min = {n/1e6/dt:.1f} MB/s")
du = subprocess.run("du -sh /content/av/Agriculture-Vision-2021/*/* /content/av/Agriculture-Vision-2021/*/images/* | sort -k2", shell=True, capture_output=True, text=True).stdout
log("du:\n" + du)
log("DONE")
