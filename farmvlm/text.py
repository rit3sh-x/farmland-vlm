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


def parse_caption(caption: str) -> tuple[Mention, ...]:
    """Every class mention in a caption with whatever extent and location words sit around it."""
    s = caption.lower().replace("center", "centre")
    return tuple(Mention(_NAME_INDEX[m["name"]], m["ext"], _loc_word(m["loc"]), m["poss"] is not None)
                 for m in _MENTION.finditer(s))
