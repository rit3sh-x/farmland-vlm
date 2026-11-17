"""Reading model output and scoring it: caption parsing, answer credit, hedges, calibration, the label-shift correction."""
import numpy as np
import pytest

from farmvlm.items import caption_refs, make_item
from farmvlm.scheme import N_CLASSES
from tests.synthetic import DRY, WEED


def test_caption_parse_round_trips_every_phrasing():
    from farmvlm.items import findings_of
    from farmvlm.text import parse_caption
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(400):
        area = np.zeros(N_CLASSES, np.float32)
        cells = rng.dirichlet(np.full(9, rng.choice([0.2, 1, 5])), size=N_CLASSES).astype(np.float32)
        ks = rng.choice(N_CLASSES, rng.integers(0, 4), replace=False)
        area[ks] = rng.choice([0.004, 0.04, 0.05, 0.1, 0.24, 0.3, 0.6], len(ks))
        item = make_item(0, "caption", -1, area, cells, cuts=(0.05, 0.25))
        want = [(f.cls, f.extent, f.loc, f.tiny) for f in findings_of(area, cells, (0.05, 0.25))]
        for ref in caption_refs(item):
            got = [(m.cls, m.extent, m.loc, m.tiny) for m in parse_caption(ref)]
            assert got == want, ref
            checked += 1
    assert checked == 2000


def test_normalise_and_identify_set():
    from farmvlm.text import identify_set, normalise
    assert normalise("The Centre.") == "centre"
    assert normalise("  Weed Clusters!") == "weed cluster"
    assert normalise("a small") == "small"
    assert identify_set("Drydown, possibly weed cluster") == identify_set("possibly weed cluster and drydown")


def test_lenient_answer_reads_sentences_into_the_closed_vocabulary():
    from farmvlm.text import lenient_answer
    assert lenient_answer("presence", "No, there is no endrow visible in this tile. The image shows") == "no"
    assert lenient_answer("presence", "Yes, there is strong evidence of **nutrient deficiency**") == "yes"
    assert lenient_answer("identify", "The image displays weed clusters and a waterway running") == "waterway, weed cluster"
    assert lenient_answer("extent", "the waterway covers approximately a small portion") == "small"
    assert lenient_answer("location", "the green strip in the bottom left corner") == "bottom left"
    assert lenient_answer("location", "it runs diagonally across the whole tile") == "it runs diagonally across the whole tile"


def test_hedge_credit_and_rates():
    from farmvlm.items import Item
    from farmvlm.metrics import vqa_credit, vqa_scores
    ext = Item(0, "extent", DRY, "small to moderate", True)
    assert [vqa_credit(ext, p) for p in ("Small to moderate", "small", "large")] == [1.0, 0.5, 0.0]
    tiny = Item(0, "presence", WEED, "possibly", True)
    assert [vqa_credit(tiny, p) for p in ("possibly", "yes", "no")] == [1.0, 0.5, 0.0]
    clear = Item(0, "location", DRY, "top left", False)
    assert vqa_credit(clear, "top left or top") == 0.0
    ident = Item(0, "identify", -1, "drydown, possibly weed cluster", True)
    assert vqa_credit(ident, "drydown") == 0.5 and vqa_credit(ident, "drydown, weed cluster") == 0.5
    s = vqa_scores([ext, clear], ["small to moderate", "top left or top"])
    assert s["hedge_rate_ambiguous"] == 1.0 and s["hedge_rate_clear"] == 1.0 and s["accuracy"] == 0.5


def test_margin_hedge_forms_the_scheme_hedges_and_scores_pairs_unordered():
    from farmvlm.metrics import margin_hedge, vqa_credit
    assert margin_hedge("presence", "yes", "no", 0.55, 0.45, 0.2) == "possibly"
    assert margin_hedge("presence", "yes", "no", 0.9, 0.1, 0.2) == "yes"
    assert margin_hedge("extent", "moderate", "large", 0.5, 0.4, 0.2) == "moderate to large"
    assert margin_hedge("extent", "small", "large", 0.5, 0.4, 0.2) == "small"
    assert margin_hedge("location", "top left", "throughout", 0.5, 0.45, 0.1) == "top left or throughout"
    item = make_item(0, "location", WEED, *_area_cells_near_tie())
    assert item.answer.count(" or ") == 1 and item.ambiguous
    a, b = item.answer.split(" or ")
    assert vqa_credit(item, f"{b} or {a}") == 1.0 and vqa_credit(item, a) == 0.5


def _area_cells_near_tie():
    area = np.zeros(N_CLASSES); area[WEED] = 0.2
    cells = np.zeros((N_CLASSES, 9)); cells[WEED, 0] = 0.46; cells[WEED, 8] = 0.44; cells[WEED, 4] = 0.10
    return area, cells


def test_caption_facts_and_calibration():
    from farmvlm.metrics import caption_facts, ece, grouped_bootstrap, risk_coverage
    area = np.zeros(N_CLASSES, np.float32); cells = np.zeros((N_CLASSES, 9), np.float32)
    area[DRY], cells[DRY, 4] = 0.6, 1.0
    item = make_item(0, "caption", -1, area, cells)
    f = caption_facts([item, item], ["Large drydown in the centre.", "an aerial view of a farm field"])
    assert f["precision"] == 1.0 and f["recall"] == 0.5 and f["extent_acc"] == 1.0
    conf = np.array([0.9, 0.8, 0.6, 0.3]); ok = np.array([1, 1, 0, 0], float)
    cov, acc, aurc = risk_coverage(conf, ok)
    assert acc[1] == 1.0 and acc[-1] == 0.5 and 0 < aurc < 0.5
    assert ece(np.full(4, 0.5), np.array([1, 0, 1, 0], float)) == pytest.approx(0.0)
    m, lo, hi = grouped_bootstrap(np.array([1, 1, 0, 0.]), np.array(["a", "a", "b", "b"]))
    assert m == 0.5 and lo <= m <= hi


def test_scores_on_empty_inputs():
    from farmvlm.metrics import caption_facts, vqa_scores
    assert vqa_scores([], [])["accuracy"] is None
    assert caption_facts([], [])["f1"] == 0.0


def test_prior_correction_swaps_to_an_undercalled_runner_up():
    from farmvlm.metrics import prior_pick, prior_ratios
    val = [{"family": "identify", "ranked": "none", "truth": "nutrient deficiency"}] * 3 + [
        {"family": "identify", "ranked": "none", "truth": "none"}]
    q = prior_ratios(val)
    assert q["identify"]["nutrient deficiency"] == 1 / 4 and q["identify"]["none"] == 5 / 2
    r = {"family": "identify", "ranked": "none", "ranked_conf": 0.6, "ranked2": "nutrient deficiency", "ranked2_conf": 0.3}
    assert prior_pick(r, q, 0) == "none" and prior_pick(r, q, 1) == "nutrient deficiency"
