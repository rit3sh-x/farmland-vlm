"""The training loop, the GPU pipeline, run records on the Hub and the report tables built from them."""
import json
import subprocess

import numpy as np
import pytest


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


def test_resume_continues_from_train_state_and_warm_restart_starts_fresh(tmp_path):
    import torch

    from farmvlm import loop

    class Tiny(torch.nn.Linear):
        def save_pretrained(self, d):
            d.mkdir(parents=True, exist_ok=True)

    model, batches, loss = Tiny(1, 1), iter(lambda: torch.ones(1, 1), None), lambda x: model(x).sum()
    scores = iter([.1 * k for k in range(1, 20)])
    first = tmp_path / "c1"; first.mkdir()
    loop.run(model, batches, loss, lambda: next(scores), loop.LoopConfig(out=first, pairs=40, batch=1, eval_every=10))
    assert (first / "last" / "train_state.pt").exists() and (first / "best").exists()
    state = loop.run(model, batches, loss, lambda: next(scores),
                     loop.LoopConfig(out=first, pairs=60, batch=1, eval_every=10, resume=first / "last"))
    assert state["step"] == 60 and len((first / "metrics.jsonl").read_text().splitlines()) == 6  # 4 checks, then 2 more
    second = tmp_path / "c2"; second.mkdir()
    state = loop.run(model, batches, loss, lambda: next(scores),
                     loop.LoopConfig(out=second, pairs=30, batch=1, eval_every=10, resume=first / "best"))
    assert state["step"] == 30 and len((second / "metrics.jsonl").read_text().splitlines()) == 3  # fresh count, as cycle 2


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


def test_hub_name_of_a_pulled_run_ignores_its_symlink(tmp_path, monkeypatch):
    from farmvlm import hub
    monkeypatch.setattr(hub, "RUNS", tmp_path / "runs")
    cached = tmp_path / "runs" / ".hub" / "runs" / "blip-vqa-f0"
    cached.mkdir(parents=True)
    (tmp_path / "runs" / "blip-vqa-f0").symlink_to(cached, target_is_directory=True)
    assert hub.name_of(tmp_path / "runs" / "blip-vqa-f0") == "blip-vqa-f0"
    assert hub.name_of(tmp_path / "runs" / "sweep-vqa" / "r4-lr0.0001-attn_qv") == "sweep-vqa/r4-lr0.0001-attn_qv"


def test_write_keeps_a_dotted_variant_name(tmp_path):
    from farmvlm.evaluate import write

    for s in (0.1, 0.25):
        write([], {"s": s}, tmp_path / f"f0-light-{s}", str)
    assert sorted(p.name for p in tmp_path.glob("*.json")) == ["f0-light-0.1.json", "f0-light-0.25.json"]


def test_gpu_minutes_sums_segments_across_resumes(tmp_path):
    from farmvlm.report import gpu_minutes
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps({"minutes": m}) + "\n" for m in (10.0, 20.0, 5.0, 15.0)))
    assert gpu_minutes(tmp_path) == 35.0


class _EchoReader:
    """Answers with the truth (or a fixed word), for exercising the evaluation and report plumbing."""
    def __init__(self, items, fixed=None):
        self.truth = {(i.row, i.family, i.cls): i for i in items}
        self.fixed, self.queue = fixed, []

    def answer(self, images, questions):
        return [(self.fixed or "yes", 0.9) for _ in questions]

    def caption(self, images):
        return [("Large drydown in the centre.", 0.8) for _ in images]


def test_report_tables_from_runs(tmp_path, table):
    t = table
    from farmvlm.evaluate import predict, summarise, write
    from farmvlm.evalset import eval_items
    from farmvlm.report import latex_fold0, table_ablation, table_constant, table_fold0, table_hedging, table_per_class
    items = eval_items(t, np.arange(12), np.array([], int))
    field_of = lambda r: str(t.field[t.row_tile[r]])
    final = "c5-capc4-blip"
    for name, fixed in (("baseline", "no"), (final, "yes"), ("served-rgb-eval", "yes")):
        (tmp_path / name).mkdir()
        for f in range(5 if name != "served-rgb-eval" else 1):
            rows_ = predict(_EchoReader(items, fixed), items, lambda r: None, batch=8)
            write(rows_, summarise(rows_), tmp_path / name / f"f{f}", field_of)
    t1 = table_ablation(tmp_path, final)
    assert [r["model"] for r in t1] == ["Zero-shot BLIP", "BLIP-LoRA"] and t1[1]["presence"][1] == pytest.approx(0.0, abs=1e-12)
    for earlier in ("archive/v1/blip-lora", "blip-lora"):
        (tmp_path / earlier).mkdir(parents=True)
        for f in range(5):
            write(rows_, summarise(rows_), tmp_path / earlier / f"f{f}", field_of)
    assert [r["model"] for r in table_ablation(tmp_path, final)] == ["Zero-shot BLIP", "BLIP-LoRA v1", "BLIP-LoRA v2", "BLIP-LoRA"]
    assert table_per_class(tmp_path, ("baseline", final))["baseline"]["drydown"]["n_present"] >= 0
    t3 = {r["model"]: r for r in table_fold0(tmp_path, final)}
    assert list(t3) == ["Zero-shot BLIP", "BLIP-LoRA v1", "BLIP-LoRA v2", "BLIP-LoRA", "Qwen3.5-2B LoRA, RGB"]
    assert t3["BLIP-LoRA"]["diff"] is None and t3["Qwen3.5-2B LoRA, RGB"]["diff"][0] == pytest.approx(0.0)
    assert t3["BLIP-LoRA v1"]["diff"][0] == pytest.approx(0.0)
    assert "BLIP-LoRA" in latex_fold0(list(t3.values())) and 0 <= table_constant(tmp_path)["mean"][0] <= 1
    names = ("baseline", final, "served-rgb-eval")
    assert {r["model"] for r in table_hedging(tmp_path, names)} == set(names)
