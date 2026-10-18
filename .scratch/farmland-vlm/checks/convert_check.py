# Colab: merge a trained LM LoRA, export GGUF (+ mmproj), quantize, and check behaviour survives;
# then merge a dummy *vision* LoRA and check the mmproj carries it.
# usage: convert_check.py <base_id> <adapter_dir> <tag>
import sys, os, json, glob, subprocess, torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel, LoraConfig, get_peft_model
base, adapter, tag = sys.argv[1:4]
OUT = f"/content/merged/{tag}"; os.makedirs(OUT, exist_ok=True)
LCPP = "/content/llama.cpp"; LBIN = "/content/llamabin/llama-b11719"
TILES = sorted(glob.glob("/content/sample/*.jpg"))
Q = "Is there drydown in this tile?"
res = {"tag": tag}
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True, errors="replace", env={**os.environ, "PYTHONPATH": f"{LCPP}/gguf-py"})
    if r.returncode: print(r.stdout[-1500:], r.stderr[-3000:], flush=True); raise SystemExit(f"failed: {c}")
    return r.stdout + r.stderr
proc = AutoProcessor.from_pretrained(base)
def hf_answers(model):
    outs = {}
    for t in TILES:
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
        p = proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        enc = proc(text=[p], images=[[Image.open(t).convert("RGB")]], return_tensors="pt").to(model.device)
        with torch.no_grad(): g = model.generate(**enc, max_new_tokens=12, do_sample=False)
        outs[os.path.basename(t)] = proc.tokenizer.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return outs
# 1. base vs fine-tuned answers in HF (GPU fp16)
m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float16, device_map="cuda")
res["hf_base"] = hf_answers(m)
m = PeftModel.from_pretrained(m, adapter); res["hf_lora"] = hf_answers(m)
del m; torch.cuda.empty_cache()
# 2. merge on CPU in bf16, save, convert, quantize
if not os.path.exists(f"{OUT}/model-Q4_K_M.gguf"):  # resumable: skip a merge/convert that already finished
    m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, adapter).merge_and_unload()
    m.save_pretrained(OUT); proc.save_pretrained(OUT); del m
    sh(f"python3 {LCPP}/convert_hf_to_gguf.py {OUT} --outtype f16 --outfile {OUT}/model-f16.gguf")
    sh(f"python3 {LCPP}/convert_hf_to_gguf.py {OUT} --mmproj --outtype f16 --outfile {OUT}/mmproj-f16.gguf")
    sh(f"{LBIN}/llama-quantize {OUT}/model-f16.gguf {OUT}/model-Q4_K_M.gguf Q4_K_M")
if os.path.exists(f"{OUT}/model-f16.gguf"): os.remove(f"{OUT}/model-f16.gguf")
res["gguf_sizes_gb"] = {os.path.basename(f): round(os.path.getsize(f) / 1e9, 2) for f in glob.glob(f"{OUT}/*.gguf")}
# 3. same question through llama.cpp on the merged Q4_K_M
res["gguf_q4km"] = {}
for t in TILES:
    o = subprocess.run(f"{LBIN}/llama-mtmd-cli -m {OUT}/model-Q4_K_M.gguf --mmproj {OUT}/mmproj-f16.gguf --image {t} -p '{Q}' -n 12 --temp 0 -t 2 --log-disable",
                       shell=True, capture_output=True, text=True, errors="replace").stdout
    res["gguf_q4km"][os.path.basename(t)] = o.strip()[-80:]
print("PARTIAL " + json.dumps(res), flush=True)
# 4. dummy vision LoRA -> merge -> mmproj; compare tensors with the LM-only merge's mmproj
sys.path.insert(0, f"{LCPP}/gguf-py"); from gguf import GGUFReader
m = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16)
vis = sorted({n.split(".")[-1] for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and "visual" in n or ("vision" in n and isinstance(mod, torch.nn.Linear))})
cfg = LoraConfig(r=8, lora_alpha=16, target_modules=r".*(visual|vision).*\.(" + "|".join(vis) + r")$", init_lora_weights=False)
m = get_peft_model(m, cfg).merge_and_unload()
VOUT = OUT + "-vis"; m.save_pretrained(VOUT); proc.save_pretrained(VOUT); del m
sh(f"python3 {LCPP}/convert_hf_to_gguf.py {VOUT} --mmproj --outtype f16 --outfile {VOUT}/mmproj-f16.gguf")
a = {t.name: np.array(t.data, dtype=np.float32) for t in GGUFReader(f"{OUT}/mmproj-f16.gguf").tensors}
b = {t.name: np.array(t.data, dtype=np.float32) for t in GGUFReader(f"{VOUT}/mmproj-f16.gguf").tensors}
diff = [k for k in a if k in b and a[k].shape == b[k].shape and not np.allclose(a[k], b[k])]
res["vision_lora_targets"] = vis
res["mmproj_tensors"] = len(a); res["mmproj_tensors_changed_by_vision_lora"] = len(diff); res["changed_examples"] = diff[:6]
print("RESULT " + json.dumps(res, indent=1), flush=True)
open("/content/bench/convert.jsonl", "a").write(json.dumps(res) + "\n")
