# Farmland VLM

Captioning and visual question answering about crop anomalies in aerial farmland imagery, for AI 512 (IIT Ropar) Project 3. BLIP is fine-tuned with LoRA on QA pairs and captions generated from [Agriculture-Vision](https://github.com/SHI-Labs/Agriculture-Vision) anomaly masks and compared against zero-shot BLIP. A newer small VLM is served on a laptop CPU behind a FastAPI backend and a React map-and-chat UI.

The design decisions live in [`.scratch/farmland-vlm/map.md`](.scratch/farmland-vlm/map.md), one resolved ticket per decision. Domain terms are defined in [`CONTEXT.md`](CONTEXT.md).

## Data

Agriculture-Vision is for non-commercial research only and may not be redistributed or posted, including in code repositories. This repository holds code only. `.gitignore` and the pre-commit hook in `scripts/hooks/` reject images, data and weight files, notebook outputs and tokens. Enable the hook once per clone:

```sh
git config core.hooksPath scripts/hooks
```

The data is streamed from the official link on every Colab session by `scripts/fetch_data.py`.

## Layout

| Path | Contents |
|---|---|
| `farmvlm/` | The package: QA scheme, mask statistics, QA items and text, field-grouped folds, balanced sampler, BLIP and VLM training and reading, evaluation, heatmaps, report tables |
| `scripts/` | Colab session setup and data fetch, the GPU pipeline, evaluators, bake-off, GGUF export, laptop setup |
| `app/backend/` | FastAPI demo backend: one model queue, NDJSON streams, field context from hosted APIs |
| `app/web/` | React + Vite "Map + chat" UI on MapLibre and OpenFreeMap |
| `tests/` | `pytest` checks on synthetic masks and runs |

## Running

On a fresh Colab VM, this clones the code, installs it, streams the dataset and builds the per-view mask statistics:

```sh
curl -sL https://raw.githubusercontent.com/rit3sh-x/farmland-vlm/main/scripts/session_setup.py | python3 -
```

Then every GPU run of the project, in schedule order, resumable through a private HF Hub repo:

```sh
HF_TOKEN=... python scripts/pipeline.py --stats /content/stats.npz --root /content/av/Agriculture-Vision-2021
```

Stages, tasks and folds can be chosen, which is how a change is checked on one fold before the rest retrain:

```sh
python scripts/pipeline.py ... --only sweep blip --tasks caption --folds 0
```

On Kaggle (two T4s), the notebook cell runs `scripts/kaggle.py`, which starts one pipeline per GPU over the argument groups set in that script; the notebook itself never changes. Finished runs are skipped and interrupted ones resume from their last checkpoint on the Hub, so a stopped session loses at most one checkpoint interval.

The report's tables, figures and headline numbers rebuild from the run records (`runs/` mirrors the Hub repo's json and jsonl files):

```sh
.venv/bin/python -m farmvlm.report --runs runs --out report/tables     # tables/*.tex, numbers.tex, results.json
.venv/bin/python scripts/figures.py --runs runs --out report/figures   # charts only, no tile imagery
(cd report && tectonic -X compile main.tex && tectonic -X compile deck.tex)
```

Tests and the demo, locally:

```sh
uv venv && uv pip install -e ".[dev,app]" && .venv/bin/python -m pytest -q
HF_TOKEN=$(cat ~/.config/farmvlm/hf_token) .venv/bin/python scripts/laptop_setup.py --served lfm-450m
(cd app/web && npm install && npm run build)
.venv/bin/uvicorn app.backend.main:app --host 127.0.0.1 --port 8000   # then open http://127.0.0.1:8000
```

Demo tiles (`data/demo_tiles/`) come from `scripts/make_demo_tiles.py` on Colab and stay private.
