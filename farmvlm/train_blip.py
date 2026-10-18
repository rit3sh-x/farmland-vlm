"""BLIP-LoRA training: the sweep, the 5 folds, the DoRA row and the leakage study all run through here.

    python -m farmvlm.train_blip --task vqa --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 \
        --fold 0 --seed 0 --rank 16 --lr 1e-4 --targets attn_qv --pairs 120000 --out /content/runs/vqa-f0
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from farmvlm import folds as F
from farmvlm.blip import CAPTION_ID, VQA_ID, BlipReader
from farmvlm.evalset import eval_items
from farmvlm.images import rgb_view
from farmvlm.items import caption_refs, render_question
from farmvlm.metrics import caption_facts, vqa_scores
from farmvlm.sampler import Sampler
from farmvlm.scheme import VIEWS
from farmvlm.stats import StatsTable

TARGETS = {
    "text_attn": r"^(text_encoder|text_decoder)\..*\.(query|value)$",
    "attn_qv": r".*\.(query|value|qkv)$",
    "all_linear": r"^(?!.*\bcls\b).*\.(query|key|value|qkv|projection|dense|fc1|fc2)$",
}
TRAIN_TEMPLATES = 5


class Draws(IterableDataset):
    """An endless stream of (image, question, answer) from the balanced sampler; each worker draws with its own seed."""

    def __init__(self, t, root, tiles, mix, seed):
        self.t, self.root, self.tiles, self.mix, self.seed = t, root, tiles, mix, seed

    def __iter__(self):
        wid = get_worker_info().id if get_worker_info() else 0
        s = Sampler(self.t, self.tiles, self.mix, self.seed * 1000 + wid)
        rng = np.random.default_rng(self.seed * 1000 + wid)
        while True:
            item = s.draw()
            ti = self.t.row_tile[item.row]
            img = rgb_view(self.root, self.t.split[ti], self.t.tile[ti], VIEWS[self.t.row_view[item.row]])
            if item.family == "caption":
                refs = caption_refs(item)
                yield img, "", refs[rng.integers(len(refs))]
            else:
                yield img, render_question(item, int(rng.integers(TRAIN_TEMPLATES))), item.answer


def build_model(task, rank, alpha, targets, dora, resume=None):
    from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering
    base = (BlipForQuestionAnswering.from_pretrained(VQA_ID) if task == "vqa"
            else BlipForConditionalGeneration.from_pretrained(CAPTION_ID)).to("cuda")
    base.half()
    if resume:
        model = PeftModel.from_pretrained(base, resume, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                                target_modules=TARGETS[targets], use_dora=dora))
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def collate(proc, task, bos):
    def fn(batch):
        imgs, qs, ans = zip(*batch)
        if task == "vqa":
            x = proc(images=list(imgs), text=list(qs), return_tensors="pt", padding=True)
        else:
            x = proc(images=list(imgs), return_tensors="pt")
        a = proc.tokenizer(list(ans), return_tensors="pt", padding=True)
        ids = a["input_ids"].clone()
        ids[:, 0] = bos
        x["answer_ids"], x["answer_mask"] = ids, a["attention_mask"]
        return x
    return fn


def loss_of(model, task, x):
    labels = x["answer_ids"].masked_fill(x["answer_mask"] == 0, -100)
    pv = x["pixel_values"].half()
    if task == "vqa":
        return model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], pixel_values=pv,
                     decoder_input_ids=x["answer_ids"], decoder_attention_mask=x["answer_mask"], labels=labels).loss
    return model(pixel_values=pv, input_ids=x["answer_ids"], attention_mask=x["answer_mask"], labels=labels).loss


def validate(model, task, t, root, items, batch=48):
    model.eval()
    reader = BlipReader.__new__(BlipReader)
    from transformers import BlipProcessor
    reader.device, reader.dtype = "cuda", torch.float16
    reader.proc = BlipProcessor.from_pretrained(VQA_ID)
    reader.cap_proc = BlipProcessor.from_pretrained(CAPTION_ID)
    reader.vqa = reader.cap = model
    reader.bos = model.config.text_config.bos_token_id
    img = lambda r: rgb_view(root, t.split[t.row_tile[r]], t.tile[t.row_tile[r]], VIEWS[t.row_view[r]])
    preds = []
    for i in range(0, len(items), batch):
        b = items[i:i + batch]
        if task == "vqa":
            preds += [p for p, _ in reader.answer([img(x.row) for x in b], [render_question(x, 0) for x in b])]
        else:
            preds += [p for p, _ in reader.caption([img(x.row) for x in b])]
    model.train()
    if task == "vqa":
        return vqa_scores(items, preds)["accuracy"]
    return caption_facts(items, preds)["f1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vqa", "caption"], required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["folds", "official"], default="folds",
                    help="official = train on the dataset's own train split (leakage study)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--targets", choices=list(TARGETS), default="attn_qv")
    ap.add_argument("--dora", action="store_true")
    ap.add_argument("--pairs", type=int, default=120_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-tiles", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    a = ap.parse_args()
    a.alpha = a.alpha or 2 * a.rank
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)

    t = StatsTable.load(a.stats)
    if a.split == "official":
        train_tiles = np.flatnonzero(t.split == "train")
        val_pool = np.flatnonzero(t.split == "val")
    else:
        fold = F.make_folds(t, a.seed)
        inner = F.inner_validation(t, fold, a.fold, a.seed)
        train_tiles = np.flatnonzero((fold != a.fold) & ~inner)
        val_pool = np.flatnonzero(inner)
    val_tiles = np.random.default_rng(a.seed).choice(val_pool, min(a.val_tiles, len(val_pool)), replace=False)
    val_items = [i for i in eval_items(t, val_tiles, np.array([], int))
                 if (i.family == "caption") == (a.task == "caption")]

    from transformers import BlipProcessor
    proc = BlipProcessor.from_pretrained(VQA_ID if a.task == "vqa" else CAPTION_ID)
    model = build_model(a.task, a.rank, a.alpha, a.targets, a.dora, a.resume)
    bos = model.config.text_config.bos_token_id
    params = [p for p in model.parameters() if p.requires_grad]
    steps = math.ceil(a.pairs / a.batch)
    opt = torch.optim.AdamW(params, lr=a.lr)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", init_scale=2 ** 12)
    state = {"step": 0, "best": -1.0, "since_best": 0, "history": []}
    if a.resume and (a.resume / "train_state.pt").exists():
        ck = torch.load(a.resume / "train_state.pt", weights_only=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
    mix = "blip_vqa" if a.task == "vqa" else "blip_caption"
    dl = DataLoader(Draws(t, a.root, train_tiles, mix, a.seed + state["step"]), batch_size=a.batch,
                    num_workers=2, collate_fn=collate(proc, a.task, bos), prefetch_factor=4)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update(train_tiles=int(len(train_tiles)), val_items=len(val_items), trainable=sum(p.numel() for p in params))
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    log = open(a.out / "metrics.jsonl", "a")
    model.train()
    t0, skipped = time.time(), 0
    for x in dl:
        if state["step"] >= steps:
            break
        x = {k: v.to("cuda") for k, v in x.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_of(model, a.task, x)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scale = scaler.get_scale()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        skipped += scaler.get_scale() < scale
        state["step"] += 1
        pairs = state["step"] * a.batch
        if pairs % a.eval_every < a.batch or state["step"] == steps:
            score = validate(model, a.task, t, a.root, val_items)
            rec = {"pairs": pairs, "loss": float(loss), "val": score, "lr": sched.get_last_lr()[0],
                   "skipped_steps": int(skipped), "minutes": round((time.time() - t0) / 60, 1)}
            state["history"].append(rec)
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(json.dumps(rec), flush=True)
            if score > state["best"]:
                state["best"], state["since_best"] = score, 0
                model.save_pretrained(a.out / "best")
            else:
                state["since_best"] += 1
            model.save_pretrained(a.out / "last")
            torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                        "state": state}, a.out / "last" / "train_state.pt")
            if skipped > 0.05 * a.eval_every / a.batch:
                print("loss-scale overflows above 5% of steps: rerun with half the LR from last/", flush=True)
            skipped = 0
            if state["since_best"] >= a.patience:
                break
    (a.out / "done.json").write_text(json.dumps({"best_val": state["best"], "pairs": state["step"] * a.batch}))


if __name__ == "__main__":
    main()
