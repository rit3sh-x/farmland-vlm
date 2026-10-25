"""Turn a trained served-model adapter into the files the laptop's llama-server loads.

Default (language-model-only adapter): a runtime LoRA GGUF next to the official base GGUF.
--merged (vision-tower adapter): merge, convert model + mmproj, quantize the model to Q4_K_M.

    python scripts/export_gguf.py --model lfm-450m --adapter /content/runs/final-rgb/best --out /content/export/rgb
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from farmvlm.vlm import MODELS  # noqa: E402

LCPP_TAG = "b11719"
LCPP = Path("/content/llama.cpp")
LBIN = Path(f"/content/llamabin/llama-{LCPP_TAG}")
LBIN_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{LCPP_TAG}/llama-{LCPP_TAG}-bin-ubuntu-x64.zip"


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, env={**os.environ, "PYTHONPATH": f"{LCPP}/gguf-py"})


def tools(binaries: bool):
    if not LCPP.exists():
        # Not llama.cpp's requirements file: it pins transformers < Qwen3.5 and protobuf 4, which broke every later
        # stage on the VM. The converter needs only gguf-py (on PYTHONPATH in sh) and what the VM already has.
        sh(f"git clone -q --depth 1 --branch {LCPP_TAG} https://github.com/ggml-org/llama.cpp {LCPP}")
    if binaries and not LBIN.exists():
        sh(f"mkdir -p /content/llamabin && cd /content/llamabin && curl -sL -o lb.zip {LBIN_URL} && unzip -q lb.zip")
        found = next(Path("/content/llamabin").rglob("llama-quantize"))
        LBIN.parent.mkdir(parents=True, exist_ok=True)
        if found.parent != LBIN:
            os.symlink(found.parent, LBIN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--merged", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    tools(binaries=a.merged)
    from huggingface_hub import snapshot_download
    base = snapshot_download(MODELS[a.model], allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model"])
    if not a.merged:
        sh(f"{sys.executable} {LCPP}/convert_lora_to_gguf.py --base {base} --outtype f16 --outfile {a.out}/lora-f16.gguf {a.adapter}")
        return
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    merged = a.out / "merged"
    m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    PeftModel.from_pretrained(m, a.adapter).merge_and_unload().save_pretrained(merged)
    AutoProcessor.from_pretrained(base).save_pretrained(merged)
    sh(f"{sys.executable} {LCPP}/convert_hf_to_gguf.py {merged} --outtype f16 --outfile {a.out}/model-f16.gguf")
    sh(f"{sys.executable} {LCPP}/convert_hf_to_gguf.py {merged} --mmproj --outtype f16 --outfile {a.out}/mmproj-f16.gguf")
    sh(f"{LBIN}/llama-quantize {a.out}/model-f16.gguf {a.out}/model-Q4_K_M.gguf Q4_K_M")
    os.remove(a.out / "model-f16.gguf")


if __name__ == "__main__":
    main()
