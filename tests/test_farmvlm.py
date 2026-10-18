import numpy as np
import pytest

from farmvlm.folds import inner_validation, make_folds, pick_test_sample, rarest_class, tile_presence
from farmvlm.items import caption_refs, extent_of, location_of, make_item, render_question
from farmvlm.sampler import CLEAN_SHARE, Sampler
from farmvlm.scheme import CLASS_INDEX, N_CLASSES, VIEWS
from farmvlm.stats import StatsTable, grid_shares, view_rows

DRY, WEED, STORM = CLASS_INDEX["drydown"], CLASS_INDEX["weed_cluster"], CLASS_INDEX["storm_damage"]


def test_grid_shares_top_left_block():
    m = np.zeros((512, 512), bool)
    m[:170, :170] = True
    s = grid_shares(m)
    assert s[0] == pytest.approx(1.0) and s[1:].sum() == 0


def test_view_rows_quadrants_and_invalid_pixels():
    labels = np.zeros((N_CLASSES, 512, 512), bool)
    labels[DRY, :256, :256] = True
    labels[WEED, :200, :200] = True
    valid = np.ones((512, 512), bool)
    valid[256:, 256:] = False
    rows = {VIEWS[v]: (n, area, cells) for v, n, area, cells in view_rows(labels, valid)}
    assert "q_br" not in rows
    assert rows["full"][1][DRY] == pytest.approx(256 * 256 / (512 * 512 - 256 * 256))
    assert rows["q_tl"][1][DRY] == pytest.approx(1.0)
    assert rows["q_tr"][1].sum() == 0
    assert location_of(rows["full"][2][DRY]) == "top left or throughout"
    assert location_of(rows["full"][2][WEED]) == "top left"
    assert location_of(rows["q_tl"][2][DRY]) == "throughout"


@pytest.mark.parametrize("area,word", [(0.01, "small"), (0.05, "small to moderate"), (0.12, "moderate"),
                                        (0.30, "moderate to large"), (0.5, "large")])
def test_extent_bins_and_borderline(area, word):
    assert extent_of(area, cuts=(0.05, 0.25)) == word


def test_location_ties():
    s = np.zeros(9)
    s[[2, 5]] = [0.48, 0.42]
    assert location_of(s) == "top right or right"
    s = np.zeros(9)
    s[4] = 0.41
    s[0] = 0.2
    assert location_of(s) == "centre or throughout"


def test_items_and_text():
    area = np.zeros(N_CLASSES, np.float32)
    cells = np.zeros((N_CLASSES, 9), np.float32)
    area[DRY], cells[DRY, 4] = 0.6, 1.0
    area[WEED], cells[WEED, 0] = 0.004, 1.0
    assert make_item(0, "presence", WEED, area, cells).answer == "possibly"
    assert make_item(0, "presence", STORM, area, cells).answer == "no"
    ident = make_item(0, "identify", -1, area, cells)
    assert ident.answer == "drydown, possibly weed cluster" and ident.ambiguous
    refs = caption_refs(make_item(0, "caption", -1, area, cells))
    assert refs[0] == "Large drydown in the centre, with possibly small weed cluster at the top left."
    assert len(set(refs)) == 5
    q = make_item(0, "location", WEED, area, cells)
    assert render_question(q, 0) == "Where is the weed cluster?"
    assert "weed" in render_question(q, 1, heldout=True)
    clean = make_item(0, "caption", -1, np.zeros(N_CLASSES), np.zeros((N_CLASSES, 9)))
    assert caption_refs(clean)[0] == "No anomalies visible."
    with pytest.raises(ValueError):
        make_item(0, "extent", STORM, area, cells)


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


def test_folds_group_by_field_and_are_seeded(tmp_path):
    t = synthetic_table()
    t.save(tmp_path / "s.npz")
    t = StatsTable.load(tmp_path / "s.npz")
    a, b, c = make_folds(t, 1), make_folds(t, 1), make_folds(t, 2)
    assert (a == b).all() and (a != c).any() and set(a) == set(range(5))
    for f in np.unique(t.field):
        assert len(set(a[t.field == f])) == 1, "a field straddles folds"
    val = inner_validation(t, a, 0, seed=1)
    assert not (val & (a == 0)).any() and 0.05 < val.sum() / (a != 0).sum() < 0.2
    assert not set(t.field[val]) & set(t.field[(a != 0) & ~val])
    sample = pick_test_sample(t, a, 0, seed=1, n=60, per_class=3)
    assert (a[sample] == 0).all() and len(sample) == len(set(sample)) <= 60
    assert tile_presence(t)[sample].sum(0).min() >= min(3, tile_presence(t)[a == 0].sum(0).min())


def test_rarest_class_label():
    p = np.zeros((3, N_CLASSES), bool)
    p[0, [DRY, STORM]] = True
    p[1, DRY] = True
    p[2, [DRY, WEED]] = True
    assert rarest_class(p).tolist() == [STORM, DRY, WEED]


def test_sampler_balance():
    t = synthetic_table()
    s = Sampler(t, np.arange(len(t.tile)), "served", seed=0)
    items = s.draws(4000)
    clean = np.mean([t.area[i.row].sum() == 0 for i in items])
    assert abs(clean - CLEAN_SHARE) < 0.03
    pres = [i for i in items if i.family == "presence" and t.area[i.row].sum() > 0]
    yes = np.mean([i.answer != "no" for i in pres])
    assert 0.4 < yes < 0.6
    fams = {f: np.mean([i.family == f for i in items]) for f in ("caption", "presence", "identify", "extent", "location")}
    assert abs(fams["caption"] - 0.30) < 0.04
    storm = np.mean([i.cls == STORM for i in items if i.family in ("extent", "location")])
    assert storm > 0.04, "rare class should be upweighted above its 2% frequency"
