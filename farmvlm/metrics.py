"""Scoring per issues/06-evaluation-protocol.md: exact match, hedge credit, caption facts, calibration, bootstrap."""
from collections import defaultdict

import numpy as np

from farmvlm.items import Item
from farmvlm.scheme import EXTENTS, POSSIBLY
from farmvlm.text import identify_set, normalise, parse_caption


def is_hedge(answer: str) -> bool:
    a = normalise(answer)
    return a.startswith(POSSIBLY) or f", {POSSIBLY}" in a or " to " in a or " or " in a


def _sides(truth: str) -> set[str]:
    """The committed answers that each earn half credit on a borderline item."""
    if truth == POSSIBLY:
        return {"yes"}
    for sep in (" to ", " or "):
        if sep in truth:
            return set(truth.split(sep))
    return set()


def vqa_credit(item: Item, pred: str) -> float:
    """1 for the right answer (the hedge on ambiguous items), 0.5 for one side of a borderline, else 0."""
    p = normalise(pred)
    if item.family == "identify":
        want, got = identify_set(item.answer), identify_set(p)
        if got == want:
            return 1.0
        strip = lambda s: frozenset(x.removeprefix(POSSIBLY + " ") for x in s)
        firm = lambda s: frozenset(x for x in s if not x.startswith(POSSIBLY))
        return 0.5 if item.ambiguous and got in (strip(want), firm(want)) else 0.0
    if p == item.answer or (" or " in p and " or " in item.answer and set(p.split(" or ")) == set(item.answer.split(" or "))):
        return 1.0
    return 0.5 if item.ambiguous and p in _sides(item.answer) else 0.0


def margin_hedge(family: str, top: str, second: str, p_top: float, p_second: float, delta: float) -> str:
    """The ranked answer, or the scheme's hedge between the top two candidates when their probabilities are within delta."""
    if p_top - p_second >= delta:
        return top
    pair = {top, second}
    if family == "presence" and pair == {"yes", "no"}:
        return POSSIBLY
    if family == "extent":
        for a, b in zip(EXTENTS, EXTENTS[1:]):
            if pair == {a, b}:
                return f"{a} to {b}"
    if family == "location" and " or " not in top and " or " not in second:
        return f"{top} or {second}"
    return top


def vqa_scores(items: list[Item], preds: list[str]) -> dict:
    if not items:
        return {"accuracy": None, "by_family": {}, "ambiguous_credit": None, "clear_accuracy": None,
                "hedge_rate_ambiguous": None, "hedge_rate_clear": None}
    credit = np.array([vqa_credit(i, p) for i, p in zip(items, preds)], float)
    amb = np.array([i.ambiguous for i in items], bool)
    hedged = np.array([is_hedge(p) for p in preds], bool)
    by_family = defaultdict(list)
    for i, c in zip(items, credit):
        by_family[i.family].append(c)
    return {
        "accuracy": float(credit.mean()),
        "by_family": {f: float(np.mean(v)) for f, v in by_family.items()},
        "ambiguous_credit": float(credit[amb].mean()) if amb.any() else None,
        "clear_accuracy": float(credit[~amb].mean()) if (~amb).any() else None,
        "hedge_rate_ambiguous": float(hedged[amb].mean()) if amb.any() else None,
        "hedge_rate_clear": float(hedged[~amb].mean()) if (~amb).any() else None,
    }


def caption_facts(items: list[Item], captions: list[str]) -> dict:
    """Class mentions as precision/recall/F1 against the masks; extent and location accuracy on matched classes."""
    tp = fp = fn = 0
    ext_ok, loc_ok, matched = 0, 0, 0
    for item, cap in zip(items, captions):
        truth = {f.cls: f for f in item.findings}
        said = {m.cls: m for m in parse_caption(cap)}
        hit = truth.keys() & said.keys()
        tp, fp, fn = tp + len(hit), fp + len(said.keys() - hit), fn + len(truth.keys() - hit)
        for k in hit:
            matched += 1
            ext_ok += said[k].extent == truth[k].extent
            loc_ok += said[k].loc == truth[k].loc
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "extent_acc": ext_ok / matched if matched else None, "location_acc": loc_ok / matched if matched else None}


def caption_overlap(refs: list[tuple[str, ...]], captions: list[str]) -> dict:
    """BLEU-1/4, ROUGE-L and CIDEr against all reference phrasings (pycocoevalcap scorers, whitespace tokens)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    tok = lambda s: " ".join(normalise(s).replace(",", " , ").split())
    gts = {i: [tok(r) for r in rs] for i, rs in enumerate(refs)}
    res = {i: [tok(c)] for i, c in enumerate(captions)}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    rouge, _ = Rouge().compute_score(gts, res)
    cider, _ = Cider().compute_score(gts, res)
    return {"bleu1": bleu[0], "bleu4": bleu[3], "rougeL": rouge, "cider": cider}


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    return float(sum(abs(correct[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def risk_coverage(conf: np.ndarray, correct: np.ndarray):
    """Accuracy when answering only the most confident share of items; returns (coverage, accuracy, AURC)."""
    order = np.argsort(-conf, kind="stable")
    acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    cov = np.arange(1, len(order) + 1) / len(order)
    return cov, acc, float(np.mean(1 - acc))


def field_resample_means(values: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0) -> np.ndarray:
    """n means of values, each over whole fields resampled with replacement."""
    uniq, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, values), np.bincount(inv)
    pick = np.random.default_rng(seed).integers(0, len(uniq), size=(n, len(uniq)))
    return sums[pick].sum(1) / counts[pick].sum(1)


def grouped_bootstrap(values: np.ndarray, groups: np.ndarray, n: int = 2000, seed: int = 0, other: np.ndarray | None = None):
    """95% interval of mean(values), or of mean(values - other) when paired, resampling whole fields."""
    v = values if other is None else values - other
    stats = field_resample_means(v, groups, n, seed)
    return float(v.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
