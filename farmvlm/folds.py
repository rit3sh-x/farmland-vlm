"""Dynamic, field-grouped cross-validation over train+val tiles (see issues/06-evaluation-protocol.md).

Folds are regenerated from a seed at the start of every experiment run and never saved as split
files. Quadrant views always follow their tile.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from farmvlm.scheme import N_CLASSES, VIEWS
from farmvlm.stats import StatsTable

FULL = VIEWS.index("full")


def tile_presence(t: StatsTable) -> np.ndarray:
    """bool[T, 9]: class present in any usable view of the tile."""
    p = np.zeros((len(t.tile), N_CLASSES), bool)
    np.logical_or.at(p, t.row_tile, t.area > 0)
    return p


def rarest_class(presence: np.ndarray) -> np.ndarray:
    """Per tile, its rarest present class (by tile frequency); -1 if none. The stratification label."""
    freq = presence.sum(0)
    order = np.argsort(freq)
    rank = np.empty(N_CLASSES, int)
    rank[order] = np.arange(N_CLASSES)
    ranked = np.where(presence, rank[None, :], N_CLASSES)
    best = ranked.min(1)
    return np.where(best < N_CLASSES, order[np.minimum(best, N_CLASSES - 1)], -1)


def make_folds(t: StatsTable, seed: int, k: int = 5) -> np.ndarray:
    """int8[T]: test-fold index of each tile."""
    y = rarest_class(tile_presence(t))
    fold = np.full(len(t.tile), -1, np.int8)
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for f, (_, test) in enumerate(skf.split(np.zeros(len(y)), y, groups=t.field)):
        fold[test] = f
    return fold


def inner_validation(t: StatsTable, fold: np.ndarray, f: int, seed: int, share: float = 0.1) -> np.ndarray:
    """bool[T]: the ~10% of fold f's training fields held back for early stopping and the sweep."""
    train = np.flatnonzero(fold != f)
    split = GroupShuffleSplit(n_splits=1, test_size=share, random_state=seed)
    _, val = next(split.split(train, groups=t.field[train]))
    mask = np.zeros(len(t.tile), bool)
    mask[train[val]] = True
    return mask


def pick_test_sample(t: StatsTable, fold: np.ndarray, f: int, seed: int, n: int = 2000, per_class: int = 50) -> np.ndarray:
    """Tile indices of fold f's fixed test sample: up to per_class tiles of every class, rarest first, then filled at random."""
    rng = np.random.default_rng(seed)
    pres = tile_presence(t)
    pool = np.flatnonzero(fold == f)
    chosen = []
    for c in np.argsort(pres[pool].sum(0)):
        have = pres[np.array(chosen, int), c].sum() if chosen else 0
        cand = np.setdiff1d(pool[pres[pool, c]], chosen)
        need = max(0, per_class - int(have))
        chosen += rng.choice(cand, min(need, len(cand)), replace=False).tolist()
    rest = np.setdiff1d(pool, chosen)
    chosen += rng.choice(rest, max(0, min(n - len(chosen), len(rest))), replace=False).tolist()
    return np.sort(np.array(chosen, int))


def clean_views(t: StatsTable, tiles: np.ndarray, seed: int, n: int = 300) -> np.ndarray:
    """Row indices of up to n anomaly-free quadrant views from the given tiles."""
    rows = np.flatnonzero(np.isin(t.row_tile, tiles) & (t.row_view != FULL) & ~(t.area > 0).any(1))
    return np.sort(np.random.default_rng(seed).choice(rows, min(n, len(rows)), replace=False))
