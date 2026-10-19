"""The fine-tuning loop shared by BLIP and the served VLM (issues/13-peft-configuration.md, "Every final run").

fp16 autocast with a capped initial loss scale, gradient clipping, cosine schedule with 3% warmup,
inner-validation every eval_every pairs, best/last adapters, resumable state, early stopping.
"""
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class LoopConfig:
    out: Path
    pairs: int
    batch: int            # pairs per optimizer step (micro-batch x accumulation)
    accum: int = 1
    lr: float = 1e-4
    eval_every: int = 5000
    patience: int = 3
    resume: Path | None = None
    on_checkpoint: object = None  # called after each checkpoint is written (e.g. push to the Hub)


def run(model, batches, loss_fn, validate, cfg: LoopConfig):
    params = [p for p in model.parameters() if p.requires_grad]
    steps = math.ceil(cfg.pairs / cfg.batch)
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    warm = max(1, int(0.03 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler("cuda", init_scale=2 ** 12)
    state = {"step": 0, "best": -1.0, "since_best": 0}
    if cfg.resume and (cfg.resume / "train_state.pt").exists():
        ck = torch.load(cfg.resume / "train_state.pt", weights_only=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
    log = open(cfg.out / "metrics.jsonl", "a")
    model.train()
    t0, skipped, micro = time.time(), 0, 0
    for x in batches:
        if state["step"] >= steps:
            break
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_fn(x) / cfg.accum
        scaler.scale(loss).backward()
        micro += 1
        if micro % cfg.accum:
            continue
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scale = scaler.get_scale()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        skipped += scaler.get_scale() < scale
        state["step"] += 1
        pairs = state["step"] * cfg.batch
        if pairs % cfg.eval_every >= cfg.batch and state["step"] != steps:
            continue
        model.eval()
        score = validate()
        model.train()
        rec = {"pairs": pairs, "loss": float(loss) * cfg.accum, "val": score, "lr": sched.get_last_lr()[0],
               "skipped_steps": int(skipped), "minutes": round((time.time() - t0) / 60, 1)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if score > state["best"]:
            state["best"], state["since_best"] = score, 0
            model.save_pretrained(cfg.out / "best")
        else:
            state["since_best"] += 1
        model.save_pretrained(cfg.out / "last")
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "state": state},
                   cfg.out / "last" / "train_state.pt")
        if cfg.on_checkpoint:
            cfg.on_checkpoint()
        if skipped > 0.05 * cfg.eval_every / cfg.batch:
            print("loss-scale overflows above 5% of steps: rerun with half the LR from last/", flush=True)
        skipped = 0
        if state["since_best"] >= cfg.patience:
            break
    (cfg.out / "done.json").write_text(json.dumps({"best_val": state["best"], "pairs": state["step"] * cfg.batch}))
    if cfg.on_checkpoint:
        cfg.on_checkpoint()
    return state
