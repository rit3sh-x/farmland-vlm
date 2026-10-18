#!/bin/bash
# Pre-download all checkpoints, then benchmark each model in its own process.
mkdir -p /content/bench /content/adapters
python3 - <<'PY' > /content/bench/predl.log 2>&1
import time
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import snapshot_download
ms = ["Salesforce/blip-vqa-base", "Salesforce/blip-image-captioning-base", "LiquidAI/LFM2.5-VL-450M", "LiquidAI/LFM2.5-VL-1.6B", "Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "google/gemma-4-E2B-it"]
t = time.time()
def dl(m):
    snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.py"]); return m, round(time.time() - t)
with ThreadPoolExecutor(3) as ex:
    for r in ex.map(dl, ms): print(r, flush=True)
PY
B=/content/train_bench.py; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { timeout 2700 python3 $B "$@" > /content/bench/$(basename $1).log 2>&1; }
run Salesforce/blip-vqa-base 200 32,16,8,4
run Salesforce/blip-image-captioning-base 200 32,16,8,4
run LiquidAI/LFM2.5-VL-450M 200 16,8,4,2,1
run LiquidAI/LFM2.5-VL-1.6B 200 16,8,4,2,1 /content/adapters/lfm16b
run Qwen/Qwen3.5-0.8B 200 16,8,4,2,1
run Qwen/Qwen3.5-2B 200 16,8,4,2,1 /content/adapters/qwen2b
run google/gemma-4-E2B-it 200 8,4,2,1 /content/adapters/gemma4e2b
echo ALLDONE >> /content/bench/results.jsonl
