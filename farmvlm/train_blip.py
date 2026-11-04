"""BLIP-LoRA training: the sweep, the 5 folds, the DoRA row and the leakage study all run through here.

    python -m farmvlm.train_blip --task vqa --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 \
        --fold 0 --seed 0 --rank 16 --lr 1e-4 --targets attn_qv --pairs 120000 --out /content/runs/vqa-f0
"""
import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from farmvlm import folds as F
from farmvlm import hub, loop, track
from farmvlm.blip import BASES, BlipReader, processor
from farmvlm.evalset import eval_items
from farmvlm.evaluate import predict, summarise, write
from farmvlm.images import SIZE, rgb_view
from farmvlm.items import caption_refs, render_question
from farmvlm.sampler import Sampler
from farmvlm.scheme import VIEWS
from farmvlm.stats import StatsTable

TARGETS = {
    "text_attn": r"^(text_encoder|text_decoder)\..*\.(query|value)$",
    "attn_qv": r".*\.(query|value|qkv)$",
    "all_linear": r"^(?!.*\bcls\b).*\.(query|key|value|qkv|projection|dense|fc1|fc2)$",
}
NORMS = ["layer_norm1", "layer_norm2", "post_layernorm", "LayerNorm"]  # every LayerNorm in BLIP's vision and text towers


class Draws(IterableDataset):
    """An endless stream of (image, question, answer) from the balanced sampler; each worker draws with its own seed."""

    def __init__(self, t, root, tiles, mix, seed, location_cell_share=None, size=SIZE, bands="rgb"):
        self.t, self.root, self.tiles, self.mix, self.seed = t, root, tiles, mix, seed
        self.location_cell_share, self.size, self.bands = location_cell_share, size, bands

    def __iter__(self):
        wid = get_worker_info().id if get_worker_info() else 0
        s = Sampler(self.t, self.tiles, self.mix, self.seed * 1000 + wid, location_cell_share=self.location_cell_share)
        rng = np.random.default_rng(self.seed * 1000 + wid)
        while True:
            item = s.draw()
            ti = self.t.row_tile[item.row]
            img = rgb_view(self.root, self.t.split[ti], self.t.tile[ti], VIEWS[self.t.row_view[item.row]], self.size, self.bands)
            if item.family == "caption":
                refs = caption_refs(item)
                yield img, "", refs[rng.integers(len(refs))]
            else:
                yield img, render_question(item, int(rng.integers(1 << 30))), item.answer


def split_tiles(t, how, fold_index, seed):
    """(training tiles, validation tiles) for fold-based runs or the official-split leakage run."""
    if how == "official":
        train = np.flatnonzero(t.split == "train")
        _, val = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(train, groups=t.field[train]))
        return np.setdiff1d(train, train[val]), train[val]
    fold = F.make_folds(t, seed)
    inner = F.inner_validation(t, fold, fold_index, seed)
    train, val, test = (np.flatnonzero((fold != fold_index) & ~inner), np.flatnonzero(inner), np.flatnonzero(fold == fold_index))
    for x, y, what in ((train, val, "validation"), (train, test, "the held-out test fold"), (val, test, "the held-out test fold")):
        shared = set(t.field[x]) & set(t.field[y])
        assert not shared, f"leakage: {len(shared)} fields in both a training-side split and {what}"
    return train, val


def build_model(task, rank, alpha, targets, dora, resume=None, dropout=0.05, train_norms=False, base_model="base"):
    from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering
    vqa_id, cap_id = BASES[base_model]
    base = (BlipForQuestionAnswering.from_pretrained(vqa_id) if task == "vqa"
            else BlipForConditionalGeneration.from_pretrained(cap_id)).to("cuda").half()
    if resume:
        model = PeftModel.from_pretrained(base, resume, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=TARGETS[targets],
                                                use_dora=dora, modules_to_save=NORMS if train_norms else None))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def collate(proc, task, bos):
    def fn(batch):
        imgs, qs, ans = zip(*batch)
        x = (proc(images=list(imgs), text=list(qs), return_tensors="pt", padding=True) if task == "vqa"
             else proc(images=list(imgs), return_tensors="pt"))
        a = proc.tokenizer(list(ans), return_tensors="pt", padding=True)
        ids = a["input_ids"].clone()
        ids[:, 0] = bos
        x["answer_ids"], x["answer_mask"] = ids, a["attention_mask"]
        return x
    return fn


def loss_of(model, task, hi_res=False):
    def fn(x):
        x = {k: v.to("cuda") for k, v in x.items()}
        labels = x["answer_ids"].masked_fill(x["answer_mask"] == 0, -100)
        pv = x["pixel_values"].half()
        if task == "vqa":
            return model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], pixel_values=pv,
                         decoder_input_ids=x["answer_ids"], decoder_attention_mask=x["answer_mask"], labels=labels,
                         interpolate_pos_encoding=hi_res).loss
        return model(pixel_values=pv, input_ids=x["answer_ids"], attention_mask=x["answer_mask"], labels=labels,
                     interpolate_pos_encoding=hi_res).loss
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["vqa", "caption"], required=True)
    ap.add_argument("--stats", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="fixes the folds and the validation items")
    ap.add_argument("--train-seed", type=int, help="LoRA init and training draws only (default: --seed): replicates of one "
                    "recipe on the same validation items")
    ap.add_argument("--split", choices=["folds", "official"], default="folds",
                    help="official = train on the dataset's own train split (leakage study)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--targets", choices=list(TARGETS), default="attn_qv")
    ap.add_argument("--dora", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--train-norms", action="store_true", help="also train every LayerNorm (saved with the adapter)")
    ap.add_argument("--size", type=int, default=SIZE, help="input px; BLIP was pretrained at 384, tiles are 512")
    ap.add_argument("--base", choices=list(BASES), default="base", help="BLIP base, or large (ViT-L; captioning only)")
    ap.add_argument("--bands", choices=["rgb", "nrg"], default="rgb", help="nrg: near-infrared, red, green false colour")
    ap.add_argument("--log-every", type=int, default=50, help="optimizer steps between train.jsonl records")
    ap.add_argument("--profile", type=int, default=0, help="profile this many steps into out/profile/ (0: off)")
    ap.add_argument("--pairs", type=int, default=120_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--val-tiles", type=int, default=300)
    ap.add_argument("--location-cell-share", type=float, help="share of location draws answered by a grid cell (default: the natural mix)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", type=Path)
    a = ap.parse_args()
    if BASES[a.base][a.task == "caption"] is None:
        ap.error(f"no BLIP-{a.base} checkpoint for {a.task}")
    a.alpha = a.alpha or 2 * a.rank
    a.train_seed = a.seed if a.train_seed is None else a.train_seed
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.train_seed)
    t = StatsTable.load(a.stats)
    train_tiles, val_pool = split_tiles(t, a.split, a.fold, a.seed)
    val_tiles = np.random.default_rng(a.seed).choice(val_pool, min(a.val_tiles, len(val_pool)), replace=False)
    val_items = [i for i in eval_items(t, val_tiles, np.array([], int)) if (i.family == "caption") == (a.task == "caption")]
    proc = processor(BASES[a.base][a.task == "caption"], a.size)
    model = build_model(a.task, a.rank, a.alpha, a.targets, a.dora, a.resume, a.dropout, a.train_norms, a.base)
    reader = BlipReader(vqa=model if a.task == "vqa" else None, captioner=model if a.task == "caption" else None, load=False,
                        size=a.size, base=a.base)
    image = lambda r: rgb_view(a.root, t.split[t.row_tile[r]], t.tile[t.row_tile[r]], VIEWS[t.row_view[r]], a.size, a.bands)
    key = "vqa_generated" if a.task == "vqa" else "caption_facts"
    checks = len(list((a.out / "val").glob("*.json"))) if (a.out / "val").exists() else 0

    def validate():
        """Mean accuracy over question families (as the served VLM's loop), so no family's item count dominates.
        Every check keeps its per-item predictions (val/NN.jsonl): trials share the validation items, so any two
        checkpoints or runs compare item by item."""
        nonlocal checks
        with torch.autocast("cuda", dtype=torch.float16):  # as in training: trained LayerNorm copies are fp32
            rows = predict(reader, val_items, image)
        full = summarise(rows)
        (a.out / "val").mkdir(exist_ok=True)
        write(rows, full, a.out / "val" / f"{checks:02d}", lambda r: str(t.field[t.row_tile[r]]))
        checks += 1
        m = full[key]
        with open(a.out / "val.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        return float(np.mean(list(m["by_family"].values()))) if a.task == "vqa" else m["f1"]

    git = lambda *c: subprocess.run(["git", "-C", str(Path(__file__).parent), *c], capture_output=True, text=True).stdout.strip()
    import peft
    import transformers
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update(train_tiles=int(len(train_tiles)), val_items=len(val_items),
               trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
               commit=git("rev-parse", "--short", "HEAD"), dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
               stats_sha256=hashlib.sha256(a.stats.read_bytes()).hexdigest()[:16], gpu=torch.cuda.get_device_name(),
               host=platform.node(), versions={"torch": torch.__version__, "transformers": transformers.__version__,
                                               "peft": peft.__version__})
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    mix = "blip_vqa" if a.task == "vqa" else "blip_caption"
    dl = DataLoader(Draws(t, a.root, train_tiles, mix, a.train_seed + (7919 if a.resume else 0), a.location_cell_share, a.size,
                          a.bands),
                    batch_size=a.batch, num_workers=2,
                    collate_fn=collate(proc, a.task, model.config.text_config.bos_token_id), prefetch_factor=4)
    name = hub.name_of(a.out)
    tracker = track.start(name, cfg, group=name.split("/")[0])
    loop.run(model, dl, loss_of(model, a.task, a.size != SIZE), validate,
             loop.LoopConfig(out=a.out, pairs=a.pairs, batch=a.batch, lr=a.lr, eval_every=a.eval_every,
                             patience=a.patience, resume=a.resume, log_every=a.log_every, profile_steps=a.profile,
                             on_checkpoint=lambda: hub.push(a.out, name),
                             on_log=tracker.log if tracker else None, on_alert=tracker.alert if tracker else None))


if __name__ == "__main__":
    main()
