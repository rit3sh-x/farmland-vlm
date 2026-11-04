"""Scoring a reader on a list of items, shared by the BLIP and VLM evaluators.

A reader has answer(images, questions) and caption(images), each returning [(text, confidence)];
BLIP readers also have rank(images, questions, candidates). An image is a PIL image, or a tuple of
them for the NDVI variant (RGB, NDVI map).
"""
import json
from pathlib import Path

from farmvlm.items import caption_refs, render_question
from farmvlm.metrics import caption_facts, caption_overlap, vqa_scores
from farmvlm.scheme import CLASSES, EXTENTS, POSSIBLY


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def candidates(items):
    by = {"presence": ["yes", "no", POSSIBLY],
          "extent": list(EXTENTS) + [f"{a} to {b}" for a, b in zip(EXTENTS, EXTENTS[1:])]}
    for fam in ("location", "identify"):
        by[fam] = sorted({i.answer for i in items if i.family == fam})
    return by


def predict(reader, items, image, batch=48, heldout=False, ranked=False):
    vqa = [i for i in items if i.family != "caption"]
    caps = [i for i in items if i.family == "caption"]
    cands = candidates(vqa) if ranked else None
    rows = []
    for b in chunks(vqa, batch):
        imgs = [image(i.row) for i in b]
        qs = [render_question(i, k if heldout else 0, heldout=heldout) for k, i in enumerate(b)]
        gen = reader.answer(imgs, qs)
        rk = reader.rank(imgs, qs, [cands[i.family] for i in b]) if ranked else [(None, None, None, None)] * len(b)
        rows += [{"item": i, "question": q, "pred": g, "conf": gc, "ranked": r, "ranked_conf": rc, "ranked2": r2, "ranked2_conf": rc2}
                 for i, q, (g, gc), (r, rc, r2, rc2) in zip(b, qs, gen, rk)]
    for b in chunks(caps, batch):
        rows += [{"item": i, "pred": c, "conf": cc} for i, (c, cc) in zip(b, reader.caption([image(i.row) for i in b]))]
    return rows


# Views of a tile as sequences of perturb's geometric ops (counter-clockwise rot90): the first four are the flips and
# the half turn, all eight the dihedral group. Aerial tiles have no "up", so each view is as valid as the original.
VIEWS_D4 = ((), ("hflip",), ("rot90", "rot90"), ("hflip", "rot90", "rot90"),
            ("rot90",), ("rot90", "rot90", "rot90"), ("hflip", "rot90"), ("hflip", "rot90", "rot90", "rot90"))


def predict_tta(reader, items, image, views, batch=48):
    """Ranked answers averaged over views of each tile. In every view each candidate is scored as the masks would read
    there (location candidates and the question remapped), and the mean log-probability picks the answer. Captions are
    read once, from the original view."""
    from functools import reduce

    from farmvlm.perturb import apply, remap, remap_loc
    vqa = [i for i in items if i.family != "caption"]
    cands = candidates(vqa)
    rows = []
    for b in chunks(vqa, batch):
        total = None
        for ops in views:
            imgs = [reduce(lambda im, k: apply(im, k, None), ops, image(i.row)) for i in b]
            qs = [render_question(reduce(remap, ops, i), 0) for i in b]
            cs = [[reduce(lambda a, k: remap_loc(a, k), ops, c) for c in cands[i.family]] if i.family == "location"
                  else cands[i.family] for i in b]
            lps = reader.scores(imgs, qs, cs)
            total = lps if total is None else [t + x for t, x in zip(total, lps)]
        for i, t in zip(b, total):
            p = (t / len(views)).softmax(0)
            j = int(p.argmax())
            rows.append({"item": i, "question": render_question(i, 0), "pred": cands[i.family][j], "conf": float(p[j]),
                         "ranked": cands[i.family][j], "ranked_conf": float(p[j])})
    for b in chunks([i for i in items if i.family == "caption"], batch):
        rows += [{"item": i, "pred": c, "conf": cc} for i, (c, cc) in zip(b, reader.caption([image(i.row) for i in b]))]
    return rows


def summarise(rows, ranked=False) -> dict:
    vqa = [r for r in rows if r["item"].family != "caption"]
    caps = [r for r in rows if r["item"].family == "caption"]
    m = {"items": len(rows),
         "vqa_generated": vqa_scores([r["item"] for r in vqa], [r["pred"] for r in vqa]),
         "caption_facts": caption_facts([r["item"] for r in caps], [r["pred"] for r in caps])}
    if caps:
        m["caption_overlap"] = caption_overlap([caption_refs(r["item"]) for r in caps], [r["pred"] for r in caps])
    if ranked:
        m["vqa_ranked"] = vqa_scores([r["item"] for r in vqa], [r["ranked"] for r in vqa])
    return m


def write(rows, metrics, out: Path, field_of):
    with open(out.parent / f"{out.name}.jsonl", "w") as fh:  # not with_suffix: "f0-light-0.1" has a dot
        for r in rows:
            i = r["item"]
            rec = {"row": i.row, "field": field_of(i.row), "family": i.family,
                   "cls": CLASSES[i.cls].key if i.cls >= 0 else None, "truth": i.answer, "ambiguous": i.ambiguous,
                   **{k: v for k, v in r.items() if k != "item"}}
            if i.family == "caption":
                rec["refs"] = caption_refs(i)
            fh.write(json.dumps(rec) + "\n")
    (out.parent / f"{out.name}.json").write_text(json.dumps(metrics, indent=1))
