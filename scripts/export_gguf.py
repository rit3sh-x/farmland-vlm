import argparse
import os
import subprocess
import sys
from pathlib import Path

from farmvlm.hub import HOME
from farmvlm.vlm import MODELS

LCPP_TAG = "b11719"
LCPP = HOME / "llama.cpp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if not LCPP.exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", LCPP_TAG, "https://github.com/ggml-org/llama.cpp", str(LCPP)],
                       check=True)
    from huggingface_hub import snapshot_download
    base = snapshot_download(MODELS[a.model], allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
    subprocess.run([sys.executable, LCPP / "convert_lora_to_gguf.py", "--base", base, "--outtype", "f16",
                    "--outfile", a.out / "lora-f16.gguf", a.adapter], check=True, env={**os.environ, "PYTHONPATH": str(LCPP / "gguf-py")})


if __name__ == "__main__":
    main()
