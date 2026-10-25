"""Balanced draws from the implicit QA pool (see issues/13-peft-configuration.md, "Sampler").

The pool is every applicable (view, family, class) item; it is never materialised. Each draw picks a
family by the run's mix, a class by weight 1/sqrt(frequency), yes/no 50/50 for presence, and a clean
quadrant view CLEAN_SHARE of the time.
"""
import numpy as np

from farmvlm.items import Item, make_item
from farmvlm.scheme import EXTENT_CUTS, N_CLASSES, THROUGHOUT, THROUGHOUT_CUT, THROUGHOUT_MARGIN, VIEWS
from farmvlm.stats import StatsTable

MIXES = {
    "served": {"caption": 0.30, "presence": 0.30, "identify": 0.15, "extent": 0.15, "location": 0.10},
    "blip_vqa": {"presence": 0.43, "identify": 0.21, "extent": 0.21, "location": 0.15},
    "blip_caption": {"caption": 1.0},
}
CLEAN_SHARE = 0.12
# Location draws: share whose answer is a grid cell rather than "throughout". In the data 55% of location truths
# are "throughout"; v1 BLIP learnt that prior and answered it for 68% of the items whose anomaly was localised.
LOCATION_CELL_SHARE = 0.65
CLEAN_FAMILIES = ("presence", "identify", "caption")
FULL = VIEWS.index("full")


class Sampler:
    def __init__(self, t: StatsTable, tiles: np.ndarray, mix: str, seed: int, cuts=EXTENT_CUTS):
        self.t, self.cuts, self.mix = t, cuts, MIXES[mix]
        self.rng = np.random.default_rng(seed)
        rows = np.flatnonzero(np.isin(t.row_tile, tiles))
        present = t.area[rows] > 0
        is_clean = ~present.any(1)
        self.clean = rows[is_clean & (t.row_view[rows] != FULL)]
        anom, pres = rows[~is_clean], present[~is_clean]
        freq = pres.sum(0)
        w = np.where(freq > 0, 1 / np.sqrt(np.maximum(freq, 1)), 0.0)
        self.class_w = w / w.sum()
        self.yes = [anom[pres[:, c]] for c in range(N_CLASSES)]
        self.no = [anom[~pres[:, c]] for c in range(N_CLASSES)]
        row_w = (pres * w[None, :]).max(1)
        self.anom, self.anom_p = anom, row_w / row_w.sum()
        localised = t.cells[anom].astype(np.float32).max(-1) > THROUGHOUT_CUT - THROUGHOUT_MARGIN  # location_of names a cell
        self.loc = [(anom[pres[:, c] & localised[:, c]], anom[pres[:, c] & ~localised[:, c]]) for c in range(N_CLASSES)]

    def _family(self, allowed):
        fams = [f for f in self.mix if f in allowed]
        p = np.array([self.mix[f] for f in fams])
        return fams[self.rng.choice(len(fams), p=p / p.sum())]

    def draw(self) -> Item:
        r = self.rng
        if len(self.clean) and r.random() < CLEAN_SHARE and any(f in self.mix for f in CLEAN_FAMILIES):
            row = int(r.choice(self.clean))
            fam = self._family(CLEAN_FAMILIES)
            cls = int(r.integers(N_CLASSES)) if fam == "presence" else -1
        else:
            fam = self._family(self.mix)
            if fam in ("identify", "caption"):
                row, cls = int(r.choice(self.anom, p=self.anom_p)), -1
            else:
                cls = int(r.choice(N_CLASSES, p=self.class_w))
                row = int(r.choice(self._rows(fam, cls)))
        return make_item(row, fam, cls, self.t.area[row], self.t.cells[row], self.cuts)

    def _rows(self, fam: str, cls: int) -> np.ndarray:
        if fam == "presence" and len(self.no[cls]) and self.rng.random() < 0.5:
            return self.no[cls]
        if fam == "location":
            cell, throughout = self.loc[cls]
            if len(cell) and len(throughout):
                return cell if self.rng.random() < LOCATION_CELL_SHARE else throughout
        return self.yes[cls]

    def draws(self, n: int) -> list[Item]:
        return [self.draw() for _ in range(n)]
