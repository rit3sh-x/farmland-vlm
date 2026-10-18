"""The fixed evaluation items for a fold: every applicable item of the test-sample tiles and the clean slice."""
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
        items += [make_item(int(r), "presence", k, area, cells, cuts) for k in range(N_CLASSES)]
        items.append(make_item(int(r), "identify", -1, area, cells, cuts))
        for k in np.flatnonzero(area > 0):
            items += [make_item(int(r), "extent", int(k), area, cells, cuts), make_item(int(r), "location", int(k), area, cells, cuts)]
        items.append(make_item(int(r), "caption", -1, area, cells, cuts))
    return items
