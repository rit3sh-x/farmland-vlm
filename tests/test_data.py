"""Mask statistics, QA items and captions, folds, the balanced sampler and the evaluation set."""
import numpy as np
import pytest

from farmvlm.folds import inner_validation, make_folds, pick_test_sample, rarest_class, tile_presence
from farmvlm.items import caption_refs, extent_of, location_of, make_item, render_question
from farmvlm.sampler import CLEAN_SHARE, Sampler
from farmvlm.scheme import N_CLASSES, VIEWS
from farmvlm.stats import StatsTable, grid_shares, view_rows
from tests.synthetic import DRY, STORM, WEED


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
    assert location_of(rows["full"][2][DRY]) == "top left"
    assert location_of(rows["full"][2][WEED]) == "top left"
    assert location_of(rows["q_tl"][2][DRY]) == "throughout"


@pytest.mark.parametrize("area,word", [(0.01, "small"), (0.05, "small to moderate"), (0.12, "moderate"),
                                        (0.27, "moderate to large"), (0.5, "large")])
def test_extent_bins_and_borderline(area, word):
    assert extent_of(area, cuts=(0.05, 0.25)) == word


def test_location_ties():
    s = np.zeros(9)
    s[[2, 5]] = [0.46, 0.43]
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
    assert refs[0] == "Large drydown in the centre, with possibly small weed cluster at the top left"
    assert len(set(refs)) == 5
    q = make_item(0, "location", WEED, area, cells)
    assert render_question(q, 0) == "Where is the weed cluster?"
    assert "weed" in render_question(q, 1, heldout=True)
    clean = make_item(0, "caption", -1, np.zeros(N_CLASSES), np.zeros((N_CLASSES, 9)))
    assert caption_refs(clean)[0] == "No anomalies visible"
    for item in (clean, make_item(0, "caption", -1, area, cells)):
        for ref in caption_refs(item):
            assert ref[-1].isalpha() and ":" not in ref, ref  # BLIP's caption head cannot produce "." or ":"
    with pytest.raises(ValueError):
        make_item(0, "extent", STORM, area, cells)


def test_question_templates_are_many_disjoint_and_keep_the_evaluation_phrasing():
    from farmvlm.items import QUESTION_TEMPLATES
    first = {"presence": "Is there {name} in this tile?", "identify": "What anomalies are visible?",
             "extent": "How much of the tile does the {name} cover?", "location": "Where is the {name}?"}
    for fam, (train, heldout) in QUESTION_TEMPLATES.items():
        assert train[0] == first[fam] and len(train) >= 12 and len(heldout) == 3
        assert not set(train) & set(heldout) and len(set(train)) == len(train)
        assert all(("{name}" in q) == (fam != "identify") for q in train)
        assert all("{syn}" in q for q in heldout) or fam == "identify"


def test_question_family_matches_every_template():
    from farmvlm.items import Item, QUESTION_TEMPLATES
    from farmvlm.text import question_family
    for fam, (train, held) in QUESTION_TEMPLATES.items():
        for k in range(len(train) + len(held)):
            item = Item(0, fam, -1 if fam == "identify" else WEED, "", False)
            q = render_question(item, k, heldout=k >= len(train))
            got = question_family(q)
            assert got[0] == fam, q
            assert got[1] == (-1 if fam == "identify" else WEED), q


def test_folds_group_by_field_and_are_seeded(tmp_path, table):
    t = table
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


def test_sampler_balance(table):
    t = table
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


def test_sampler_location_draws_follow_the_data_unless_a_cell_share_is_set(table):
    t = table
    spread = np.arange(len(t.area)) % 2 == 0  # even rows: the mask is spread over the grid, so the answer is "throughout"
    t.cells[spread] = (t.area[spread] > 0)[:, :, None] / 9
    share = lambda s: np.mean([i.answer == "throughout" for i in s.draws(6000) if i.family == "location"])
    assert abs(share(Sampler(t, np.arange(len(t.tile)), "blip_vqa", 0)) - 0.5) < 0.1
    assert abs(share(Sampler(t, np.arange(len(t.tile)), "blip_vqa", 0, location_cell_share=0.65)) - 0.35) < 0.06


def test_eval_items_cover_every_family(table):
    t = table
    from farmvlm.evalset import eval_items
    from farmvlm.folds import clean_views
    tiles = np.arange(10)
    clean = clean_views(t, tiles, seed=0, n=5)
    items = eval_items(t, tiles, clean)
    full_rows = np.flatnonzero(np.isin(t.row_tile, tiles) & (t.row_view == 0))
    n = lambda r: int((t.area[r] > 0).sum())
    expect = sum(n(r) + min(N_CLASSES - n(r), max(1, n(r))) + 2 + 2 * n(r) for r in full_rows) + len(clean) * 3
    assert len(items) == expect
    assert {i.answer for i in items if i.family == "identify" and i.row in set(clean)} == {"none"}
    said_no = [i.answer == "no" for i in items if i.family == "presence" and i.row not in set(clean)]
    assert 0.4 <= np.mean(said_no) <= 0.6
    assert [(i.row, i.cls) for i in eval_items(t, tiles, clean)] == [(i.row, i.cls) for i in items]


@pytest.mark.parametrize("kind,np_op", [("hflip", np.fliplr), ("rot90", np.rot90)])
def test_geometric_remap_matches_the_pixels(kind, np_op):
    from farmvlm.perturb import remap_loc
    from farmvlm.scheme import CELLS
    for r0, c0 in [(0, 0), (0, 2), (1, 0), (2, 1)]:
        m = np.zeros((300, 300), bool)
        m[r0 * 100:(r0 + 1) * 100, c0 * 100:(c0 + 1) * 100] = True
        before = CELLS[int(np.argmax(grid_shares(m)))]
        after = CELLS[int(np.argmax(grid_shares(np_op(m))))]
        assert remap_loc(before, kind) == after
    assert remap_loc("top left or top", "hflip") == "top right or top"


def test_photometric_variants_keep_size():
    from PIL import Image
    from farmvlm.perturb import apply, variants
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (384, 384, 3), dtype=np.uint8))
    for kind, s in variants():
        assert apply(img, kind, s).size == (384, 384), kind
