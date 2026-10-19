"""The fixed evaluation items for a fold: every applicable item of the test-sample tiles and the clean slice.

Presence is asked yes/no balanced: every class in the view, plus as many absent classes (at least one),
picked by a generator seeded on the row so every model sees the same questions.
"""
import numpy as np

from farmvlm.items import Item, make_item
from farmvlm.scheme import EXTENT_CUTS, N_CLASSES, VIEWS
from farmvlm.stats import StatsTable

FULL = VIEWS.index("full")


def eval_items(t: StatsTable, tiles: np.ndarray, clean_rows: np.ndarray, cuts=EXTENT_CUTS) -> list[Item]:
    rows = np.concatenate([np.flatnonzero(np.isin(t.row_tile, tiles) & (t.row_view == FULL)), clean_rows])
    items = []
    for r in rows:
        area, cells = t.area[r], t.cells[r]
        present, absent = np.flatnonzero(area > 0), np.flatnonzero(area == 0)
        asked = np.sort(np.concatenate([present, np.random.default_rng(int(r)).choice(
            absent, min(len(absent), max(1, len(present))), replace=False)]))
        items += [make_item(int(r), "presence", int(k), area, cells, cuts) for k in asked]
        items.append(make_item(int(r), "identify", -1, area, cells, cuts))
        for k in present:
            items += [make_item(int(r), "extent", int(k), area, cells, cuts), make_item(int(r), "location", int(k), area, cells, cuts)]
        items.append(make_item(int(r), "caption", -1, area, cells, cuts))
    return items
