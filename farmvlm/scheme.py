"""The QA-pair scheme's vocabulary: classes, views, location cells, cut points and answer words.

Locked in .scratch/farmland-vlm/issues/05-qa-pair-scheme.md. Thresholds marked "week 1" are set
from the measured stats and then frozen.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AnomalyClass:
    key: str
    name: str
    synonyms: tuple[str, ...]


CLASSES = (
    AnomalyClass("double_plant", "double plant", ("double planting", "doubled rows")),
    AnomalyClass("drydown", "drydown", ("dry-down", "drying crop")),
    AnomalyClass("endrow", "endrow", ("end rows", "end-row pattern")),
    AnomalyClass("nutrient_deficiency", "nutrient deficiency", ("nutrient-deficient crop", "lack of nutrients")),
    AnomalyClass("planter_skip", "planter skip", ("skipped rows", "planter gaps")),
    AnomalyClass("storm_damage", "storm damage", ("wind damage", "storm-damaged crop")),
    AnomalyClass("water", "standing water", ("water", "ponding")),
    AnomalyClass("waterway", "waterway", ("grassed waterway", "drainage channel")),
    AnomalyClass("weed_cluster", "weed cluster", ("weeds", "weed patch")),
)
CLASS_INDEX = {c.key: i for i, c in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)

# Views of a tile: the full 512 px tile and its four 256 px quadrants (top-left, top-right, bottom-left, bottom-right).
VIEWS = ("full", "q_tl", "q_tr", "q_bl", "q_br")
VIEW_SLICES = {
    "full": (slice(0, 512), slice(0, 512)),
    "q_tl": (slice(0, 256), slice(0, 256)),
    "q_tr": (slice(0, 256), slice(256, 512)),
    "q_bl": (slice(256, 512), slice(0, 256)),
    "q_br": (slice(256, 512), slice(256, 512)),
}
MIN_VALID_SHARE = 0.5  # a view needs this share of in-field pixels to be used

CELLS = ("top left", "top", "top right", "left", "centre", "right", "bottom left", "bottom", "bottom right")
THROUGHOUT = "throughout"
THROUGHOUT_CUT = 0.4  # no cell holds this share of the mask -> "throughout"
LOCATION_TIE_MARGIN = 0.05  # top cell beats runner-up by less than this -> "A or B" (week 1: ~10% of findings)
THROUGHOUT_MARGIN = 0.025  # top share within this of THROUGHOUT_CUT -> "A or throughout"

EXTENTS = ("small", "moderate", "large")
EXTENT_CUTS = (0.10, 0.35)  # week 1: rounded tertiles (0.109, 0.374) of present-class area over 72,661 full views
EXTENT_MARGIN = 0.12  # relative distance to a cut that counts as borderline (week 1: ~12% of findings)
TINY_AREA = 0.01  # present but under this share of the field -> "possibly"

POSSIBLY = "possibly"
NONE = "none"
FAMILIES = ("presence", "identify", "extent", "location", "caption")
