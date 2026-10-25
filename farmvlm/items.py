"""QA items: what a view's masks say, as one question family's answer. Text rendering is separate.

An Item is the semantic answer (class, extent, location, hedge). render_question / caption_refs turn
it into words, so the same Item can be asked with a training template or a held-out paraphrase.
"""
from dataclasses import dataclass

import numpy as np

from farmvlm.scheme import (CELLS, CLASSES, EXTENT_CUTS, EXTENT_MARGIN, EXTENTS, LOCATION_TIE_MARGIN, NONE,
                            POSSIBLY, THROUGHOUT, THROUGHOUT_CUT, THROUGHOUT_MARGIN, TINY_AREA)


@dataclass(frozen=True)
class Finding:
    cls: int
    area: float
    extent: str          # "small" | "moderate" | "large" | "small to moderate" | "moderate to large"
    loc: str             # a cell word, "throughout", or "A or B"
    tiny: bool           # present but under TINY_AREA -> "possibly"

    @property
    def hedged(self):
        return self.tiny or " to " in self.extent or " or " in self.loc


@dataclass(frozen=True)
class Item:
    row: int
    family: str          # scheme.FAMILIES
    cls: int             # -1 for identify / caption
    answer: str          # closed-vocabulary answer; captions use caption_refs
    ambiguous: bool
    findings: tuple[Finding, ...] = ()


def extent_of(area: float, cuts=EXTENT_CUTS, margin=EXTENT_MARGIN) -> str:
    for i, c in enumerate(cuts):
        if abs(area - c) <= margin * c:
            return f"{EXTENTS[i]} to {EXTENTS[i + 1]}"
    return EXTENTS[int(np.searchsorted(cuts, area, side="right"))]


def location_of(shares: np.ndarray) -> str:
    order = np.argsort(shares)[::-1]
    top, second = int(order[0]), int(order[1])
    s, s2 = float(shares[top]), float(shares[second])
    if abs(s - THROUGHOUT_CUT) < THROUGHOUT_MARGIN:
        return f"{CELLS[top]} or {THROUGHOUT}"
    if s < THROUGHOUT_CUT:
        return THROUGHOUT
    if s - s2 < LOCATION_TIE_MARGIN:
        return f"{CELLS[top]} or {CELLS[second]}"
    return CELLS[top]


def findings_of(area: np.ndarray, cells: np.ndarray, cuts=EXTENT_CUTS) -> tuple[Finding, ...]:
    present = [k for k in np.argsort(area)[::-1] if area[k] > 0]
    return tuple(Finding(int(k), float(area[k]), extent_of(float(area[k]), cuts),
                         location_of(cells[k].astype(np.float32)), bool(area[k] < TINY_AREA)) for k in present)


def make_item(row: int, family: str, cls: int, area: np.ndarray, cells: np.ndarray, cuts=EXTENT_CUTS) -> Item:
    fs = findings_of(area, cells, cuts)
    hit = next((f for f in fs if f.cls == cls), None)
    if family == "presence":
        if hit is None:
            return Item(row, family, cls, "no", False)
        return Item(row, family, cls, POSSIBLY if hit.tiny else "yes", hit.tiny, (hit,))
    if family in ("extent", "location"):
        if hit is None:
            raise ValueError(f"{family} item for absent class {CLASSES[cls].key}")
        ans = hit.extent if family == "extent" else hit.loc
        return Item(row, family, cls, ans, " to " in ans or " or " in ans, (hit,))
    if family == "identify":
        ans = ", ".join((POSSIBLY + " " if f.tiny else "") + CLASSES[f.cls].name for f in fs) or NONE
        return Item(row, family, -1, ans, any(f.tiny for f in fs), fs)
    if family == "caption":
        return Item(row, family, -1, "", any(f.hedged for f in fs), fs)
    raise ValueError(family)


# ---- text ----

# Per family: training phrasings (the first is the one every evaluation asks), then the held-out phrasings with class
# synonyms for the paraphrase test. v1 trained on 5 phrasings and lost 11 points to the held-out ones.
QUESTION_TEMPLATES = {
    "presence": (
        ("Is there {name} in this tile?", "Does this tile show {name}?", "Can you see {name} here?",
         "Is any {name} visible in this field?", "Does this image contain {name}?",
         "Is {name} present in this tile?", "Do you see any {name} in this image?", "Is there any {name} here?",
         "Does this field have {name}?", "Is {name} visible?", "Are there signs of {name} in this tile?",
         "Is this tile affected by {name}?"),
        ("Do you spot {syn} in this aerial view?", "Would you say {syn} is present here?",
         "Any sign of {syn} in this picture?"),
    ),
    "identify": (
        ("What anomalies are visible?", "Which anomalies does this tile show?", "What field anomalies can you see?",
         "List the anomalies in this tile.", "What problems are visible in this field?",
         "What anomalies are present in this tile?", "Which field problems can you see here?",
         "What field issues are visible here?", "Which anomalies are there?", "Identify the anomalies in this image.",
         "What anomalies does this field show?", "What kinds of anomaly appear in this tile?"),
        ("Which crop issues show up in this aerial image?", "What is wrong with this field, if anything?",
         "Name every field pattern you can see here."),
    ),
    "extent": (
        ("How much of the tile does the {name} cover?", "How large is the {name}?", "How extensive is the {name}?",
         "Is the {name} small, moderate or large?", "How big is the {name} area?",
         "How much of the field has {name}?", "What is the size of the {name} area?", "How much {name} is there?",
         "Is the {name} area small, moderate or large?", "How extensive is the {name} in this tile?",
         "How much of this tile is covered by {name}?", "Would you call the {name} small, moderate or large?"),
        ("How much ground does the {syn} take up?", "What share of the field is affected by {syn}?",
         "How widespread is the {syn}?"),
    ),
    "location": (
        ("Where is the {name}?", "Where in the tile is the {name}?", "Which part of the tile has the {name}?",
         "Where does the {name} appear?", "In which part of the image is the {name}?",
         "Where in this image is the {name}?", "Where is the {name} located?", "Which part of the field has the {name}?",
         "In which area of the tile is the {name}?", "Where should I look for the {name}?",
         "Which region of the tile shows the {name}?", "Where is the {name} in this tile?"),
        ("Where can the {syn} be found?", "Point to where the {syn} is.", "Which area shows the {syn}?"),
    ),
}

# Captions end on their last word. BLIP's caption head cannot emit "." or ":" (base NLL ~13 nats, and LoRA leaves
# the head frozen), so v1 captions that ended in "." learned "[SEP] after the full stop" and never stopped.
CAPTION_TEMPLATES = (
    "{first}{rest}",
    "This tile shows {first_l}{rest}",
    "Visible here is {first_l}{rest}",
    "The field has {first_l}{rest}",
    "In this tile there is {first_l}{rest}",
)
CLEAN_CAPTIONS = ("No anomalies visible", "This tile shows no anomalies", "Visible here is a field with no anomalies",
                  "The field looks clean", "In this tile there are no anomalies")


def render_question(item: Item, variant: int, heldout: bool = False) -> str:
    templates = QUESTION_TEMPLATES[item.family][1 if heldout else 0]
    t = templates[variant % len(templates)]
    if item.cls < 0:
        return t
    c = CLASSES[item.cls]
    return t.format(name=c.name, syn=c.synonyms[variant % len(c.synonyms)])


def _phrase(f: Finding) -> str:
    loc = {"throughout": "throughout the tile", "centre": "in the centre"}.get(f.loc, f"at the {f.loc}")
    loc = loc.replace(" or throughout", " or throughout the tile")
    return f"{POSSIBLY + ' ' if f.tiny else ''}{f.extent} {CLASSES[f.cls].name} {loc}"


def caption_refs(item: Item) -> tuple[str, ...]:
    """Every phrasing of the caption. Training samples one; BLEU/ROUGE/CIDEr score against all."""
    if not item.findings:
        return CLEAN_CAPTIONS
    first, *others = map(_phrase, item.findings)
    rest = (", with " + " and ".join(others)) if others else ""
    return tuple(t.format(first=first[0].upper() + first[1:], first_l=first, rest=rest) for t in CAPTION_TEMPLATES)
