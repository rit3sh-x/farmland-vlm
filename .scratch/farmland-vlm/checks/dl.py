# Download files from a HF repo: dl.py <repo> <outdir> <file>...
import sys, os, time, urllib.request
repo, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
for f in files:
    dst = os.path.join(out, os.path.basename(f))
    if os.path.exists(dst): continue
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/{repo}/resolve/main/{f}", headers={"User-Agent": "farmland-vlm-research/0.1"}), timeout=60)
    with open(dst + ".part", "wb") as w:
        while b := r.read(4 << 20): w.write(b)
    os.rename(dst + ".part", dst)
    print(f"{f}: {os.path.getsize(dst)/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)
