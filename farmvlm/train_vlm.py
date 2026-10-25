"""Served-model LoRA training: the bake-off, the vision test, and the final RGB / NDVI-variant runs.

    python -m farmvlm.train_vlm --model lfm-450m --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 \
        --fold 0 --pairs 10000 --out /content/runs/bake-lfm-450m
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from farmvlm import hub, loop, vlm
from farmvlm.evalset import eval_items
from farmvlm.evaluate import predict, summarise
from farmvlm.images import ndvi_view, rgb_view
from farmvlm.items import caption_refs, render_question
from farmvlm.sampler import Sampler
from farmvlm.scheme import VIEWS
from farmvlm.stats import StatsTable
from farmvlm.train_blip import split_tiles


def view_image(t, root, row, ndvi: bool):
    ti, view = t.row_tile[row], VIEWS[t.row_view[row]]
    rgb = rgb_view(root, t.split[ti], t.tile[ti], view)
    return (rgb, ndvi_view(root, t.split[ti], t.tile[ti], view)) if ndvi else rgb


class Draws(IterableDataset):
    def __init__(self, t, root, tiles, seed, ndvi):
        self.t, self.root, self.tiles, self.seed, self.ndvi = t, root, tiles, seed, ndvi

    def __iter__(self):
        wid = get_worker_info().id if get_worker_info() else 0
        s = Sampler(self.t, self.tiles, "served", self.seed * 1000 + wid)
        rng = np.random.default_rng(self.seed * 1000 + wid)
        while True:
            item = s.draw()
            img = view_image(self.t, self.root, item.row, self.ndvi)
            if item.family == "caption":
                refs = caption_refs(item)
                yield img, "", refs[rng.integers(len(refs))]
            else:
                yield img, render_question(item, int(rng.integers(1 << 30))), item.answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(vlm.MODELS), required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ndvi", action="store_true", help="NDVI variant: RGB + NDVI map per prompt")
    ap.add_argument("--vision", action="store_true", help="also adapt the vision tower (vision test)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--pairs", type=int, default=30_000)
    ap.add_argument("--batch", type=int, default=16, help="effective batch (micro-batch x accumulation)")
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-tiles", type=int, default=150)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    t = StatsTable.load(a.stats)
    train_tiles, val_pool = split_tiles(t, "folds", a.fold, a.seed)
    val_tiles = np.random.default_rng(a.seed).choice(val_pool, min(a.val_tiles, len(val_pool)), replace=False)
    val_items = eval_items(t, val_tiles, np.array([], int))
    model, proc = vlm.load(a.model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if a.resume:
        model = PeftModel.from_pretrained(model, a.resume, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                                 target_modules=vlm.lora_targets(model, a.vision)))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    micro = min(vlm.MICRO_BATCH[a.model] // (2 if a.ndvi else 1) or 1, a.batch)
    reader = vlm.VlmReader(model, proc)
    image = lambda r: view_image(t, a.root, r, a.ndvi)

    def validate():
        m = summarise(predict(reader, val_items, image, batch=micro * 2))
        fam = m["vqa_generated"]["by_family"]
        return float(np.mean(list(fam.values()))) + 1e-3 * m["caption_facts"]["f1"]

    def loss_fn(batch):
        imgs, qs, ans = zip(*batch)
        enc = vlm.train_batch(proc, list(imgs), list(qs), list(ans)).to("cuda")
        return model(**enc).loss

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update(model_id=vlm.MODELS[a.model], micro_batch=micro, train_tiles=int(len(train_tiles)),
               val_items=len(val_items), trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    dl = DataLoader(Draws(t, a.root, train_tiles, a.seed + (7919 if a.resume else 0), a.ndvi), batch_size=micro, num_workers=2,
                    collate_fn=lambda b: b, prefetch_factor=4)
    loop.run(model, dl, loss_fn, validate,
             loop.LoopConfig(out=a.out, pairs=a.pairs, batch=a.batch, accum=max(1, a.batch // micro), lr=a.lr,
                             eval_every=a.eval_every, patience=a.patience, resume=a.resume,
                             on_checkpoint=lambda: hub.push(a.out, hub.name_of(a.out))))
    proc.save_pretrained(a.out / "best")


if __name__ == "__main__":
    main()
