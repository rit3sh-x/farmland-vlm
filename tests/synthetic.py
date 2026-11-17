"""A synthetic stats table shaped like Agriculture-Vision's, so no test needs the dataset."""
import numpy as np

from farmvlm.scheme import CLASS_INDEX, N_CLASSES, VIEWS
from farmvlm.stats import StatsTable

DRY, WEED, STORM = CLASS_INDEX["drydown"], CLASS_INDEX["weed_cluster"], CLASS_INDEX["storm_damage"]


def synthetic_table(n_fields=60, tiles_per_field=8, seed=0):
    rng = np.random.default_rng(seed)
    tiles, fields, rt, rv, areas = [], [], [], [], []
    for f in range(n_fields):
        for k in range(tiles_per_field):
            ti = len(tiles)
            tiles.append(f"F{f}_{k}")
            fields.append(f"field{f}")
            a = np.zeros(N_CLASSES, np.float32)
            a[rng.choice(N_CLASSES, p=np.r_[np.full(8, 0.98 / 8), 0.02])] = rng.uniform(0.002, 0.8)
            for v in range(len(VIEWS)):
                rt.append(ti); rv.append(v)
                areas.append(a if v == 0 or rng.random() < 0.5 else np.zeros(N_CLASSES, np.float32))
    area = np.array(areas)
    cells = np.zeros((len(area), N_CLASSES, 9), np.float16)
    cells[..., 4] = (area > 0)
    T = len(tiles)
    return StatsTable(tile=np.array(tiles), split=np.array(["train"] * T), field=np.array(fields),
                      year=np.full(T, 2019, np.int16), row_tile=np.array(rt, np.int32),
                      row_view=np.array(rv, np.uint8), valid_px=np.full(len(rt), 65536, np.int32),
                      area=area, cells=cells)
