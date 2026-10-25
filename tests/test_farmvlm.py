import subprocess

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


def test_eval_items_cover_every_family():
    from farmvlm.evalset import eval_items
    from farmvlm.folds import clean_views
    t = synthetic_table()
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


class _EchoReader:
    """Answers with the truth (or a fixed word), for exercising the evaluation and report plumbing."""
    def __init__(self, items, fixed=None):
        self.truth = {(i.row, i.family, i.cls): i for i in items}
        self.fixed, self.queue = fixed, []

    def answer(self, images, questions):
        return [(self.fixed or "yes", 0.9) for _ in questions]

    def caption(self, images):
        return [("Large drydown in the centre.", 0.8) for _ in images]


def test_report_tables_from_runs(tmp_path):
    from farmvlm.evaluate import predict, summarise, write
    from farmvlm.evalset import eval_items
    from farmvlm.report import latex_fold0, table_ablation, table_constant, table_fold0, table_hedging, table_per_class
    t = synthetic_table()
    items = eval_items(t, np.arange(12), np.array([], int))
    field_of = lambda r: str(t.field[t.row_tile[r]])
    for name, fixed in (("baseline", "no"), ("blip-lora", "yes"), ("served-rgb-eval", "yes")):
        (tmp_path / name).mkdir()
        for f in range(5 if name != "served-rgb-eval" else 1):
            rows_ = predict(_EchoReader(items, fixed), items, lambda r: None, batch=8)
            write(rows_, summarise(rows_), tmp_path / name / f"f{f}", field_of)
    t1 = table_ablation(tmp_path)
    assert [r["model"] for r in t1] == ["Zero-shot BLIP", "BLIP-LoRA"] and t1[1]["presence"][1] == pytest.approx(0.0, abs=1e-12)
    assert table_per_class(tmp_path)["baseline"]["drydown"]["n_present"] >= 0
    t3 = {r["model"]: r for r in table_fold0(tmp_path)}
    assert t3["BLIP-LoRA"]["diff"] is None and t3["Qwen3.5-2B LoRA, RGB"]["diff"][0] == pytest.approx(0.0)
    assert "BLIP-LoRA" in latex_fold0(list(t3.values())) and 0 <= table_constant(tmp_path)["mean"][0] <= 1
    assert {r["model"] for r in table_hedging(tmp_path)} == {"baseline", "blip-lora", "served-rgb-eval"}


def test_finding_confidence_uses_only_its_tokens():
    import math
    from app.backend.engine import _findings
    toks = [("Large", -0.1), (" drydown", -0.1), (" in", -0.1), (" the", -0.1), (" centre", -0.1), (",", -2.0),
            (" with", -2.0), (" possibly", -1.0), (" small", -1.0), (" weed", -1.0), (" cluster", -1.0),
            (" at", -1.0), (" the", -1.0), (" top", -1.0), (" left", -1.0), (".", -3.0)]
    caption = "".join(t for t, _ in toks)
    rows = _findings(caption, toks, 0.5)
    assert [r["cls"] for r in rows] == [DRY, WEED]
    assert rows[0]["conf"] == pytest.approx(math.exp(-0.1), abs=1e-3)
    assert rows[1]["conf"] == pytest.approx(math.exp(-1.0), abs=1e-3) and rows[1]["possibly"] and rows[1]["hedge"]
    assert rows[0]["extent"] == "large" and rows[0]["loc"] == "centre"


def test_scores_on_empty_inputs():
    from farmvlm.metrics import caption_facts, vqa_scores
    assert vqa_scores([], [])["accuracy"] is None
    assert caption_facts([], [])["f1"] == 0.0


def test_hub_name_of_a_pulled_run_ignores_its_symlink(tmp_path, monkeypatch):
    from farmvlm import hub
    monkeypatch.setattr(hub, "RUNS", tmp_path / "runs")
    cached = tmp_path / "runs" / ".hub" / "runs" / "blip-vqa-f0"
    cached.mkdir(parents=True)
    (tmp_path / "runs" / "blip-vqa-f0").symlink_to(cached, target_is_directory=True)
    assert hub.name_of(tmp_path / "runs" / "blip-vqa-f0") == "blip-vqa-f0"
    assert hub.name_of(tmp_path / "runs" / "sweep-vqa" / "r4-lr0.0001-attn_qv") == "sweep-vqa/r4-lr0.0001-attn_qv"


def test_best_and_early_stop_use_only_second_half_checks(tmp_path):
    import torch

    from farmvlm import loop

    class Tiny(torch.nn.Linear):
        def save_pretrained(self, d):
            d.mkdir(parents=True, exist_ok=True)

    model = Tiny(1, 1)
    scores = iter([.3, .5, .55, .6] + [.58] * 8)  # best at 40 of 120 pairs, as blip-vqa-f4
    state = loop.run(model, iter(lambda: torch.ones(1, 1), None), lambda x: model(x).sum(), lambda: next(scores),
                     loop.LoopConfig(out=tmp_path, pairs=120, batch=1, eval_every=10))
    assert state["step"] == 90 and state["best"] == .58 and (tmp_path / "best").exists()


def test_sh_parallel_puts_each_command_on_its_own_gpu(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("pipeline", Path(__file__).parent.parent / "scripts" / "pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    write = "import os, sys; open(sys.argv[1], 'w').write(os.environ.get('CUDA_VISIBLE_DEVICES', '-'))"
    cmds = [[sys.executable, "-c", write, tmp_path / n] for n in ("a", "b")]
    monkeypatch.setattr(pipeline, "visible_gpus", lambda: ["0", "1"])
    pipeline.sh_parallel(cmds)
    assert [(tmp_path / n).read_text() for n in ("a", "b")] == ["0", "1"]
    monkeypatch.setattr(pipeline, "visible_gpus", lambda: ["0"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    pipeline.sh_parallel(cmds)
    assert [(tmp_path / n).read_text() for n in ("a", "b")] == ["-", "-"]
    with pytest.raises(subprocess.CalledProcessError):
        monkeypatch.setattr(pipeline, "visible_gpus", lambda: ["0", "1"])
        pipeline.sh_parallel([[sys.executable, "-c", "pass"], [sys.executable, "-c", "raise SystemExit(3)"]])


def test_write_keeps_a_dotted_variant_name(tmp_path):
    from farmvlm.evaluate import write

    for s in (0.1, 0.25):
        write([], {"s": s}, tmp_path / f"f0-light-{s}", str)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["f0-light-0.1.json", "f0-light-0.25.json"]
