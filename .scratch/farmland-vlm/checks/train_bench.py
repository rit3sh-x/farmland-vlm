# Colab T4: fp16 LoRA throughput + stability on real Agriculture-Vision tiles.
# usage: train_bench.py <model_id> <total_steps> <bs candidates, e.g. 16,8,4,2,1> [save_adapter_dir]
import sys, os, time, json, glob, random, math, torch
from PIL import Image
from peft import LoraConfig, get_peft_model
mid, total, bss = sys.argv[1], int(sys.argv[2]), [int(x) for x in sys.argv[3].split(",")]
save_dir = sys.argv[4] if len(sys.argv) > 4 else None
ROOT = "/content/av/Agriculture-Vision-2021/train"
CLASSES = ["double_plant", "drydown", "endrow", "nutrient_deficiency", "planter_skip", "storm_damage", "water", "waterway", "weed_cluster"]
EMPTY = 1102  # byte size of an all-zero 512x512 label PNG (research heuristic)
random.seed(0); torch.manual_seed(0)
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{ROOT}/images/rgb/*.jpg"))
random.shuffle(names); names = names[:6000]
def present(n): return [c for c in CLASSES if os.path.getsize(f"{ROOT}/labels/{c}/{n}.png") != EMPTY]
IS_BLIP = "blip" in mid.lower()
KIND = ("vqa" if "vqa" in mid else "caption") if IS_BLIP else "mixed"

class DS(torch.utils.data.Dataset):
    def __len__(self): return len(names)
    def __getitem__(self, i):
        n = names[i]; img = Image.open(f"{ROOT}/images/rgb/{n}.jpg").convert("RGB"); pr = present(n)
        cls = random.choice(CLASSES)
        if KIND == "vqa" or (KIND == "mixed" and i % 2):
            return img, f"Is there {cls.replace('_', ' ')} in this tile?", "yes" if cls in pr else "no"
        cap = "This tile shows " + (", ".join(c.replace("_", " ") for c in pr) or "no labelled anomaly") + "."
        return img, "Describe the anomalies in this aerial farmland tile.", cap

t_load = time.time()
if IS_BLIP:
    from transformers import BlipProcessor, BlipForQuestionAnswering, BlipForConditionalGeneration
    proc = BlipProcessor.from_pretrained(mid)
    model = (BlipForQuestionAnswering if KIND == "vqa" else BlipForConditionalGeneration).from_pretrained(mid, dtype=torch.float16).cuda()
    target = r".*\.(query|value|qkv)$"
else:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModelForImageTextToText.from_pretrained(mid, dtype=torch.float16, device_map="cuda", attn_implementation="sdpa")
    leaf = sorted({n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "language_model" in n and "lm_head" not in n})
    target = r".*language_model.*\.(" + "|".join(leaf) + r")$"
load_s = time.time() - t_load
tok = proc.tokenizer; tok.padding_side = "right"

def collate(batch):
    imgs, qs, ans = zip(*batch)
    if IS_BLIP:
        if KIND == "vqa":
            enc = proc(images=list(imgs), text=list(qs), padding=True, return_tensors="pt")
            lab = tok(list(ans), padding=True, return_tensors="pt").input_ids
        else:
            enc = proc(images=list(imgs), text=list(ans), padding=True, return_tensors="pt")
            lab = enc["input_ids"].clone()
        lab[lab == tok.pad_token_id] = -100; enc["labels"] = lab; return enc
    full, pre = [], []
    for q, a in zip(qs, ans):
        m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        pre.append(proc.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False))
        full.append(proc.apply_chat_template(m + [{"role": "assistant", "content": [{"type": "text", "text": a}]}], tokenize=False, enable_thinking=False))
    try: enc = proc(text=full, images=[[im] for im in imgs], padding=True, return_tensors="pt")
    except Exception: enc = proc(text=full, images=list(imgs), padding=True, return_tensors="pt")
    lab = enc["input_ids"].clone(); lab[enc["attention_mask"] == 0] = -100
    for i, (f, p) in enumerate(zip(full, pre)):
        nsuf = len(tok(f[len(p):], add_special_tokens=False).input_ids) if f.startswith(p) else 8
        lab[i, : int(enc["attention_mask"][i].sum()) - nsuf] = -100
    enc["labels"] = lab; return enc

if not IS_BLIP:  # BLIP is small; its BERT embeddings add in place, which breaks enable_input_require_grads
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=target))
for p in model.parameters():
    if p.requires_grad: p.data = p.data.float()
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
params = [p for p in model.parameters() if p.requires_grad]

def run(bs, steps, log_every=0):
    opt = torch.optim.AdamW(params, lr=1e-4); scaler = torch.amp.GradScaler("cuda")
    dl = torch.utils.data.DataLoader(DS(), batch_size=bs, shuffle=True, num_workers=2, collate_fn=collate, prefetch_factor=4, persistent_workers=False, drop_last=True)
    it = iter(dl); hist = []; t_wait = 0.0; seq = None
    torch.cuda.synchronize(); t0 = time.time(); tm = None
    for s in range(steps):
        if s == 10: torch.cuda.synchronize(); tm = time.time(); t_wait = 0.0
        tw = time.time(); b = next(it); t_wait += time.time() - tw
        b = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in b.items()}
        seq = b["input_ids"].shape[1]
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**b).loss
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        hist.append((float(loss), float(gn), scaler.get_scale()))
        if log_every and s % log_every == 0: print(f"step {s} loss {hist[-1][0]:.4f} gn {hist[-1][1]:.3f} scale {hist[-1][2]:.0f}", flush=True)
    torch.cuda.synchronize(); dt = time.time() - (tm or t0)
    return hist, dt, t_wait, seq

res = {"model": mid, "load_s": round(load_s, 1), "trainable_M": round(trainable / 1e6, 2), "lora_target": target}
bs_ok = None
for bs in bss:
    try:
        torch.cuda.reset_peak_memory_stats(); run(bs, 2); bs_ok = bs; break
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache(); print(f"bs {bs} OOM", flush=True)
res["batch_size"] = bs_ok
if bs_ok:
    torch.cuda.reset_peak_memory_stats()
    hist, dt, wait, seq = run(bs_ok, total, log_every=20)
    meas = total - 10
    losses = [h[0] for h in hist]
    res.update(steps=total, seq_len=seq, samples_per_s=round(meas * bs_ok / dt, 2), s_per_step=round(dt / meas, 2),
               dataloader_wait_frac=round(wait / dt, 2), peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
               nonfinite_loss_steps=sum(not math.isfinite(l) for l in losses),
               scale_drops=sum(1 for a, b in zip(hist, hist[1:]) if b[2] < a[2]), final_scale=hist[-1][2],
               loss_first20=round(sum(losses[:20]) / 20, 4), loss_last20=round(sum(l for l in losses[-20:] if math.isfinite(l)) / 20, 4))
    if save_dir: model.save_pretrained(save_dir); proc.save_pretrained(save_dir); res["adapter"] = save_dir
print("RESULT " + json.dumps(res), flush=True)
open("/content/bench/results.jsonl", "a").write(json.dumps(res) + "\n")
