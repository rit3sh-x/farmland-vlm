# Colab: turn each PEFT adapter into (a) an fp16 PEFT copy for mistral.rs and (b) a llama.cpp LoRA GGUF.
import subprocess, os, json, shutil, torch
from safetensors.torch import load_file, save_file
LCPP = "/content/llama.cpp"
out = {}
for tag, base in [("lfm16b", "LiquidAI/LFM2.5-VL-1.6B"), ("qwen2b", "Qwen/Qwen3.5-2B")]:
    a = f"/content/adapters/{tag}"; d = f"/content/export/{tag}"; os.makedirs(d, exist_ok=True)
    sd = load_file(f"{a}/adapter_model.safetensors")
    save_file({k: v.half().contiguous() for k, v in sd.items()}, f"{d}/adapter_model.safetensors")
    cfg = json.load(open(f"{a}/adapter_config.json")); json.dump(cfg, open(f"{d}/adapter_config.json", "w"))
    from huggingface_hub import snapshot_download
    bdir = snapshot_download(base, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt"])
    r = subprocess.run(f"python3 {LCPP}/convert_lora_to_gguf.py --base {bdir} --outtype f16 --outfile {d}/lora-f16.gguf {a}",
                       shell=True, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": f"{LCPP}/gguf-py"})
    out[tag] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-600:] if r.returncode else "",
                "files_mb": {f: round(os.path.getsize(f"{d}/{f}") / 1e6, 1) for f in os.listdir(d)}}
print(json.dumps(out, indent=1))
