"""Put everything the laptop demo needs into ~/.cache/farmvlm: llama.cpp binaries, the served model's base GGUF
and projector, and the trained adapters from the private runs repo.

    HF_TOKEN=$(cat ~/.config/farmvlm/hf_token) python scripts/laptop_setup.py --served lfm-450m
"""
import argparse
import io
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm import hub  # noqa: E402

HOME = Path(os.environ.get("FARMVLM_HOME", Path.home() / ".cache" / "farmvlm"))
LLAMA_TAG = "b11719"
GGUF = {  # official quantised base models: (repo, model file, projector file)
    "lfm-450m": ("LiquidAI/LFM2.5-VL-450M-GGUF", "LFM2.5-VL-450M-Q4_K_M.gguf", "mmproj-LFM2.5-VL-450m-Q8_0.gguf"),
    "lfm-1.6b": ("LiquidAI/LFM2.5-VL-1.6B-GGUF", "LFM2.5-VL-1.6B-Q4_K_M.gguf", "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf"),
    "qwen35-2b": ("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", "mmproj-F16.gguf"),
}


def llama():
    dst = HOME / f"llama-{LLAMA_TAG}"
    if (dst / "llama-server").exists():
        return
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_TAG}/llama-{LLAMA_TAG}-bin-ubuntu-x64.zip"
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read()))
    tmp = HOME / "llama-unzip"
    z.extractall(tmp)
    src = next(tmp.rglob("llama-server")).parent
    shutil.move(str(src), dst)
    shutil.rmtree(tmp, ignore_errors=True)
    for f in dst.iterdir():
        f.chmod(0o755)


def models(key):
    from huggingface_hub import hf_hub_download
    repo, model, proj = GGUF[key]
    (HOME / "models").mkdir(parents=True, exist_ok=True)
    for f, name in ((model, f"{key}.gguf"), (proj, f"{key}-mmproj.gguf")):
        if not (HOME / "models" / name).exists():
            shutil.copy(hf_hub_download(repo, f), HOME / "models" / name)


def adapters():
    (HOME / "adapters").mkdir(parents=True, exist_ok=True)
    for run, name in (("export-served-rgb", "rgb-lora-f16.gguf"), ("export-served-ndvi", "ndvi-lora-f16.gguf")):
        src = hub.pull(run, HOME / "runs") / "lora-f16.gguf"
        if src.exists():
            shutil.copy(src, HOME / "adapters" / name)
            print("adapter", name)
        else:
            print(f"no {run} on the Hub yet: the demo serves the base model for that variant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--served", choices=list(GGUF), default="lfm-450m")
    a = ap.parse_args()
    HOME.mkdir(parents=True, exist_ok=True)
    llama()
    models(a.served)
    adapters()
    print(f"ready: FARMVLM_SERVED={a.served} .venv/bin/uvicorn app.backend.main:app --port 8000")


if __name__ == "__main__":
    main()
