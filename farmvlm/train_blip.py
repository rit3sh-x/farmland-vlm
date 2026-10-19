"""BLIP-LoRA training: the sweep, the 5 folds, the DoRA row and the leakage study all run through here.

    python -m farmvlm.train_blip --task vqa --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021 \
        --fold 0 --seed 0 --rank 16 --lr 1e-4 --targets attn_qv --pairs 120000 --out /content/runs/vqa-f0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from farmvlm import folds as F
from farmvlm import hub, loop
from farmvlm.blip import CAPTION_ID, VQA_ID, BlipReader
from farmvlm.evalset import eval_items
from farmvlm.evaluate import predict, summarise
from farmvlm.images import rgb_view
from farmvlm.items import caption_refs, render_question
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


def split_tiles(t, how, fold_index, seed):
    """(training tiles, validation tiles) for fold-based runs or the official-split leakage run."""
    if how == "official":
        train = np.flatnonzero(t.split == "train")
        _, val = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed).split(train, groups=t.field[train]))
        return np.setdiff1d(train, train[val]), train[val]
    fold = F.make_folds(t, seed)
    inner = F.inner_validation(t, fold, fold_index, seed)
    return np.flatnonzero((fold != fold_index) & ~inner), np.flatnonzero(inner)


def build_model(task, rank, alpha, targets, dora, resume=None):
    from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering
    base = (BlipForQuestionAnswering.from_pretrained(VQA_ID) if task == "vqa"
            else BlipForConditionalGeneration.from_pretrained(CAPTION_ID)).to("cuda").half()
    if resume:
        model = PeftModel.from_pretrained(base, resume, is_trainable=True)
    else:
        model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05,
                                                target_modules=TARGETS[targets], use_dora=dora))
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


def loss_of(model, task):
    def fn(x):
        x = {k: v.to("cuda") for k, v in x.items()}
        labels = x["answer_ids"].masked_fill(x["answer_mask"] == 0, -100)
        pv = x["pixel_values"].half()
        if task == "vqa":
            return model(input_ids=x["input_ids"], attention_mask=x["attention_mask"], pixel_values=pv,
                         decoder_input_ids=x["answer_ids"], decoder_attention_mask=x["answer_mask"], labels=labels).loss
        return model(pixel_values=pv, input_ids=x["answer_ids"], attention_mask=x["answer_mask"], labels=labels).loss
    return fn


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
    train_tiles, val_pool = split_tiles(t, a.split, a.fold, a.seed)
    val_tiles = np.random.default_rng(a.seed).choice(val_pool, min(a.val_tiles, len(val_pool)), replace=False)
    val_items = [i for i in eval_items(t, val_tiles, np.array([], int)) if (i.family == "caption") == (a.task == "caption")]
    from transformers import BlipProcessor
    proc = BlipProcessor.from_pretrained(VQA_ID if a.task == "vqa" else CAPTION_ID)
    model = build_model(a.task, a.rank, a.alpha, a.targets, a.dora, a.resume)
    reader = BlipReader(vqa=model if a.task == "vqa" else None, captioner=model if a.task == "caption" else None, load=False)
    image = lambda r: rgb_view(a.root, t.split[t.row_tile[r]], t.tile[t.row_tile[r]], VIEWS[t.row_view[r]])
    key = "vqa_generated" if a.task == "vqa" else "caption_facts"
    field = "accuracy" if a.task == "vqa" else "f1"

    def validate():
        m = summarise(predict(reader, val_items, image))[key]
        with open(a.out / "val.jsonl", "a") as f:
            f.write(json.dumps(m) + "\n")
        return m[field]

    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    cfg.update(train_tiles=int(len(train_tiles)), val_items=len(val_items),
               trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    (a.out / "config.json").write_text(json.dumps(cfg, indent=1))
    mix = "blip_vqa" if a.task == "vqa" else "blip_caption"
    dl = DataLoader(Draws(t, a.root, train_tiles, mix, a.seed + (7919 if a.resume else 0)), batch_size=a.batch, num_workers=2,
                    collate_fn=collate(proc, a.task, model.config.text_config.bos_token_id), prefetch_factor=4)
    loop.run(model, dl, loss_of(model, a.task), validate,
             loop.LoopConfig(out=a.out, pairs=a.pairs, batch=a.batch, lr=a.lr, eval_every=a.eval_every,
                             patience=a.patience, resume=a.resume,
                             on_checkpoint=lambda: hub.push(a.out, hub.name_of(a.out))))


if __name__ == "__main__":
    main()
