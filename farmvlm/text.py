"""Reading model output back into the scheme: strict answer normalising and caption parsing.

Shared by the evaluation (exact match, caption fact F1) and the API (finding rows under a caption).
"""
import re
from dataclasses import dataclass

from farmvlm.scheme import CELLS, CLASSES, EXTENTS, THROUGHOUT

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = re.compile(r"[^\w\s,-]")


def normalise(answer: str) -> str:
    """Case, punctuation, articles, centre/center and plural class names. No synonym map, on purpose."""
    s = answer.lower().replace("center", "centre")
    s = _PUNCT.sub(" ", s)
    s = _ARTICLES.sub(" ", s)
    for c in CLASSES:
        s = re.sub(rf"\b{re.escape(c.name)}s\b", c.name, s)
    return re.sub(r"\s+", " ", s).strip(" ,")


def identify_set(answer: str) -> frozenset[str]:
    """Identify answers are scored as a set of (possibly-prefixed) class names."""
    parts = [p.strip() for p in normalise(answer).replace(" and ", ",").split(",")]
    return frozenset(p for p in parts if p)


@dataclass(frozen=True)
class Mention:
    cls: int
    extent: str | None
    loc: str | None
    tiny: bool


_NAMES = sorted(((c.name, i) for i, c in enumerate(CLASSES)), key=lambda x: -len(x[0]))
_NAME_RE = "|".join(re.escape(n) for n, _ in _NAMES)
_EXT_RE = "small to moderate|moderate to large|" + "|".join(EXTENTS)
_CELL_RE = "|".join(re.escape(c) for c in sorted(CELLS, key=len, reverse=True))
_LOC_RE = (rf"throughout the tile|in the centre(?: or (?:{_CELL_RE}|throughout the tile))?"
           rf"|at the (?:{_CELL_RE})(?: or (?:{_CELL_RE}|throughout the tile))?")
_MENTION = re.compile(rf"(?P<poss>possibly )?(?:(?P<ext>{_EXT_RE}) )?(?P<name>{_NAME_RE})(?: (?P<loc>{_LOC_RE}))?")
_NAME_INDEX = dict(_NAMES)


def _loc_word(phrase: str | None) -> str | None:
    if phrase is None:
        return None
    s = phrase.replace("throughout the tile", THROUGHOUT).replace("in the centre", "centre")
    return re.sub(r"^at the ", "", s)


def mention_spans(caption: str) -> list[tuple[Mention, int, int]]:
    """Each class mention with its character span in the caption."""
    s = caption.lower().replace("center", "centre")
    return [(Mention(_NAME_INDEX[m["name"]], m["ext"], _loc_word(m["loc"]), m["poss"] is not None), m.start(), m.end())
            for m in _MENTION.finditer(s)]


def parse_caption(caption: str) -> tuple[Mention, ...]:
    """Every class mention in a caption with whatever extent and location words sit around it."""
    return tuple(m for m, _, _ in mention_spans(caption))


WHY = {  # the one-line reason shown under a hedged answer, per hedge type
    "possibly": "It covers only a sliver of the field, so a few pixels decide it.",
    "range": "Its area is close to the cut-off between those two sizes.",
    "or": "Two parts of the tile hold almost the same share of it.",
}


def why(answer: str) -> str | None:
    a = normalise(answer)
    if a.startswith("possibly") or ", possibly" in a:
        return WHY["possibly"]
    if " to " in a:
        return WHY["range"]
    if " or " in a:
        return WHY["or"]
    return None


_FAMILY_CUES = (
    ("location", ("where", "which part", "which area", "which region", "point to", "in which part")),
    ("extent", ("how much", "how large", "how big", "how extensive", "how widespread", "what share", "small, moderate",
                "what is the size")),
    ("identify", ("what anomal", "which anomal", "what field", "which field", "list ", "what problems", "which crop issues",
                  "what is wrong", "name every", "identify the", "what kinds of anomal")),
)


_YES = re.compile(r"^(yes|yes[,.!]|there (is|are)\b|i can see\b)")
_NO = re.compile(r"^(no|no[,.!]|there (is|are) no\b|i (can't|cannot|do not|don't) see\b|none\b)")


def lenient_answer(family: str, text: str) -> str:
    """The closed-vocabulary answer a free-text reply amounts to, for models that answer in sentences.

    Presence: a leading yes or no. Identify: every class named, largest first is unknown so alphabetical. Extent: the
    first extent word. Location: the first grid cell or "throughout". Anything else stays as written (and scores 0).
    """
    s = normalise(text)
    if family == "presence":
        return "no" if _NO.match(s) else "yes" if _YES.match(s) else text
    if family == "identify":
        names = sorted({m.cls for m, _, _ in mention_spans(s)})
        return ", ".join(CLASSES[c].name for c in names) if names else ("none" if "no anomal" in s or _NO.match(s) else text)
    if family == "extent":
        m = re.search(_EXT_RE, s)
        return m.group(0) if m else text
    if family == "location":
        m = re.search(rf"throughout|{_CELL_RE}", s)
        return m.group(0) if m else text
    return text


def question_family(question: str) -> tuple[str, int]:
    """(family, class index or -1) of a free-text question, so the UI can draw the right answer widget."""
    q = question.lower()
    names = sorted(((n, i) for i, c in enumerate(CLASSES) for n in (c.name, *c.synonyms, c.key.replace("_", " "))),
                   key=lambda x: -len(x[0]))
    cls = next((i for n, i in names if n in q), -1)
    for fam, cues in _FAMILY_CUES:
        if any(q.startswith(c) or f" {c}" in q for c in cues):
            return fam, cls
    return "presence", cls
