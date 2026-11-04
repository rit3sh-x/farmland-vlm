"""The fine-tuning loop shared by BLIP and the served VLM (issues/13-peft-configuration.md, "Every final run").

fp16 autocast with a capped initial loss scale, gradient clipping, cosine schedule with 3% warmup,
inner-validation every eval_every pairs, best/last adapters, resumable state. The best adapter and early
stopping use second-half checks only: validation swings while the cosine LR is near its peak, so a first-half
peak is noise (blip-caption-f2 once kept a 5k-pair adapter that named no locations).

Telemetry (2026-11-04): every log_every steps train.jsonl gets the mean loss, the gradient norm before clipping
(overall and per tower), loss scale, throughput and GPU memory; each validation adds the LoRA update's size per
tower (||BA||*scale, and relative to the frozen weight). Alerts flag non-finite losses, loss-scale overflows and a
validation collapse; profile_steps records a torch.profiler window; done.json carries the run's resource use.
"""
import json
import math
import time
from collections import defaultdict
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
    on_log: object = None  # called with every train.jsonl / metrics.jsonl record (e.g. a Trackio dashboard)
    on_alert: object = None  # called with (title, text) when training looks unhealthy
    log_every: int = 50  # optimizer steps between train.jsonl records
    profile_steps: int = 0  # >0: profile that many steps after 20 warm-up steps into out/profile/


def tower(name: str) -> str:
    """The part of the model a parameter belongs to, for per-tower gradient and LoRA statistics."""
    for key, label in (("vision", "vision"), ("visual", "vision"), ("text_encoder", "text_encoder"),
                       ("text_decoder", "decoder"), ("language_model", "decoder")):
        if key in name:
            return label
    return "other"


@torch.no_grad()
def lora_stats(model) -> dict:
    """Per tower, the Frobenius norm of the LoRA update (B @ A * scaling) and its ratio to the frozen weight's norm."""
    upd, base = defaultdict(float), defaultdict(float)
    for name, m in model.named_modules():
        if "default" not in getattr(m, "lora_A", {}):
            continue
        delta = (m.lora_B["default"].weight.float() @ m.lora_A["default"].weight.float()) * m.scaling["default"]
        upd[tower(name)] += float(delta.pow(2).sum())
        base[tower(name)] += float(m.base_layer.weight.float().pow(2).sum())
    return {**{f"lora/{k}_update": math.sqrt(v) for k, v in upd.items()},
            **{f"lora/{k}_relative": math.sqrt(upd[k] / base[k]) for k in upd if base[k]}}


class Profile:
    """A torch.profiler window over profile_steps optimizer steps: a Chrome trace, CUDA and CPU stacks for a
    flamegraph, and the top operators, written to out/profile/."""

    def __init__(self, cfg):
        self.out, self.start, self.stop, self.prof = cfg.out / "profile", 20, 20 + cfg.profile_steps, None

    def step(self, step):
        if step == self.start:
            from torch.profiler import ProfilerActivity, profile
            self.prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True,
                                profile_memory=True, with_stack=True)
            self.prof.start()
        elif step == self.stop and self.prof:
            self.prof.stop()
            self.out.mkdir(parents=True, exist_ok=True)
            self.prof.export_chrome_trace(str(self.out / "trace.json"))
            self.prof.export_stacks(str(self.out / "stacks_cuda.txt"), "self_cuda_time_total")
            self.prof.export_stacks(str(self.out / "stacks_cpu.txt"), "self_cpu_time_total")
            (self.out / "top_ops.txt").write_text(self.prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))
            print(f"profile written to {self.out}", flush=True)
            self.prof = None


def grad_norms(named) -> dict:
    """Gradient L2 norm per tower (gradients already unscaled)."""
    sq = defaultdict(float)
    for t, p in named:
        if p.grad is not None:
            sq[t] += float(p.grad.float().pow(2).sum())
    return {k: math.sqrt(v) for k, v in sq.items()}


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
    log, tlog = open(cfg.out / "metrics.jsonl", "a"), open(cfg.out / "train.jsonl", "a")
    named = [(tower(n), p) for n, p in model.named_parameters() if p.requires_grad]
    alert = cfg.on_alert or (lambda title, text: print(f"ALERT {title}: {text}", flush=True))
    prof = Profile(cfg) if cfg.profile_steps else None
    torch.cuda.reset_peak_memory_stats()
    model.train()
    t0, skipped, micro = time.time(), 0, 0
    window, losses, gnorms, tw = defaultdict(list), [], [], time.time()
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
        if (state["step"] + 1) % cfg.log_every == 0:  # per-tower gradient norms, before clipping
            for k, v in grad_norms(named).items():
                window[f"grad/{k}"].append(v)
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)  # a tensor: read once per log window, not every step
        scale = scaler.get_scale()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        skipped += scaler.get_scale() < scale
        state["step"] += 1
        losses.append(loss.detach() * cfg.accum); gnorms.append(gnorm.detach())
        if prof:
            prof.step(state["step"])
        pairs = state["step"] * cfg.batch
        if state["step"] % cfg.log_every == 0:
            lv, gv = torch.stack(losses).float(), torch.stack(gnorms).float()
            if not torch.isfinite(lv).all():
                alert("non-finite loss", f"step {state['step']}: {int((~torch.isfinite(lv)).sum())} of {len(lv)} losses")
            gv = gv[torch.isfinite(gv)]  # overflow steps (skipped by the scaler) have infinite norms
            trec = {"step": state["step"], "pairs": pairs, **{k: sum(v) / len(v) for k, v in window.items() if v},
                    "train/loss": float(lv[torch.isfinite(lv)].mean()) if torch.isfinite(lv).any() else float("nan"),
                    "train/grad_norm": float(gv.mean()) if len(gv) else float("nan"),
                    "train/grad_norm_max": float(gv.max()) if len(gv) else float("nan"),
                    "train/lr": sched.get_last_lr()[0], "train/loss_scale": scaler.get_scale(),
                    "train/pairs_per_s": cfg.log_every * cfg.batch / (time.time() - tw),
                    "gpu/mem_gb": torch.cuda.memory_allocated() / 2**30, "gpu/peak_gb": torch.cuda.max_memory_allocated() / 2**30}
            tlog.write(json.dumps(trec) + "\n"); tlog.flush()
            if cfg.on_log:
                cfg.on_log(trec)
            window, losses, gnorms, tw = defaultdict(list), [], [], time.time()
        if pairs % cfg.eval_every >= cfg.batch and state["step"] != steps:
            continue
        model.eval()
        score = validate()
        model.train()
        tw = time.time()  # validation time is not training throughput
        rec = {"pairs": pairs, "loss": float(loss) * cfg.accum, "val": score, "lr": sched.get_last_lr()[0],
               "skipped_steps": int(skipped), "minutes": round((time.time() - t0) / 60, 1), **lora_stats(model)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        if cfg.on_log:
            cfg.on_log({"step": state["step"], **{f"val/{k}" if k in ("val", "loss") else k: v for k, v in rec.items()}})
        if state["best"] >= 0 and score < state["best"] - 0.05:
            alert("validation collapse", f"{pairs} pairs: validation {score:.3f}, best {state['best']:.3f}")
        if pairs >= cfg.pairs // 2 and score > state["best"]:
            state["best"], state["since_best"] = score, 0
            model.save_pretrained(cfg.out / "best")
        elif pairs >= cfg.pairs // 2:
            state["since_best"] += 1
        model.save_pretrained(cfg.out / "last")
        torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(), "state": state},
                   cfg.out / "last" / "train_state.pt")
        if cfg.on_checkpoint:
            cfg.on_checkpoint()
        if skipped > 0.05 * cfg.eval_every / cfg.batch:
            print("loss-scale overflows above 5% of steps: rerun with half the LR from last/", flush=True)
            alert("loss-scale overflows", f"{skipped} skipped steps in the last {cfg.eval_every} pairs")
        skipped = 0
        if state["since_best"] >= cfg.patience:
            break
    minutes = (time.time() - t0) / 60
    (cfg.out / "done.json").write_text(json.dumps({
        "best_val": state["best"], "pairs": state["step"] * cfg.batch, "minutes_this_session": round(minutes, 1),
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "gpu": torch.cuda.get_device_name()}))
    if cfg.on_checkpoint:
        cfg.on_checkpoint()
    return state
