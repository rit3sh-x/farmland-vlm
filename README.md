# Farmland VLM

Captioning and visual question answering about crop anomalies in aerial farmland tiles, for AI 512 (IIT Ropar) Project 3.

We fine-tune BLIP with LoRA on question-answer pairs and captions generated from the [Agriculture-Vision](https://github.com/SHI-Labs/Agriculture-Vision) anomaly masks. On held-out fields it averages 74.8% over the four question types, against 29.6% for zero-shot BLIP. A demo app runs it on a laptop CPU next to zero-shot BLIP and a fine-tuned Qwen3.5-2B, with a map, a chat panel and satellite context for any field you draw.

Design decisions live in [`.scratch/farmland-vlm/map.md`](.scratch/farmland-vlm/map.md), one resolved ticket per decision. [`CONTEXT.md`](CONTEXT.md) defines the domain terms.

## Data

Agriculture-Vision is for non-commercial research only and may not be redistributed, including in code repositories. This repository holds code only. `.gitignore` and the pre-commit hook in `scripts/hooks/` reject images, data, weights, notebook outputs and tokens. Enable the hook once per clone:

```sh
git config core.hooksPath scripts/hooks
```

## Running the app

You need [uv](https://docs.astral.sh/uv/), Node.js 18 or newer, about 8 GB of RAM and 5 GB of disk. No GPU. It runs on Linux, macOS and Windows.

The trained adapters are on the Hugging Face Hub at [DarkSlayer16/farmland-vlm](https://huggingface.co/DarkSlayer16/farmland-vlm), behind a click-through for the dataset's non-commercial terms. Accept them on that page once, then:

```sh
git clone https://github.com/rit3sh-x/farmland-vlm.git && cd farmland-vlm
make install                      # uv sync: Python, PyTorch (CPU build) and llama.cpp through xllamacpp
uv run --extra app hf auth login  # once, so the app can fetch the gated adapters
make web serve                    # build the UI, then serve it on http://127.0.0.1:8000
```

Without make, the three targets are `uv sync --extra app --extra dev`, `pnpm ci && pnpm run build` in `app/frontend`, and `uv run --extra app farmvlm-serve`.

The first start downloads about 4.5 GB of weights into the Hugging Face cache. Later starts take seconds. The model menu shows each model as loading until it is ready.

| Model | What runs | A reply on a laptop CPU |
|---|---|---|
| BLIP-LoRA (default) | BLIP-base with our LoRA, answering by ranking each question type's answers | 1.5 to 3 s |
| Zero-shot BLIP | The same model with the LoRA switched off, the baseline | about 1 s |
| Qwen3.5-2B LoRA | Qwen3.5-2B (4-bit GGUF) with our RGB adapter, run by llama.cpp | 5 to 10 s |
| Qwen3.5-2B LoRA + NDVI | The NDVI adapter, reading an NDVI map as a second image. Demo tiles only, since uploads have no near-infrared band | 10 to 16 s |

Upload any aerial photo or pick a demo tile. Demo tiles are dataset images, so they never enter the repository. Build them from a local copy of the dataset (see Training) with:

```sh
uv run --extra app python scripts/make_demo_tiles.py --stats data/stats.npz --root data/av/Agriculture-Vision-2021 --out data/demo_tiles
```

For live reload while editing the UI, run `make serve` and `make dev` side by side and open the URL Vite prints.

`app/backend/models.py` lists the four models and where their weights come from. Nothing else needs configuring.

## Training and evaluation

Training needs an NVIDIA GPU with 16 GB or more and the dataset (21 GB). `make install-train` installs the training extras with the CUDA build of PyTorch. Then download the dataset from its official link and build the per-view mask statistics:

```sh
make data stats
```

`scripts/pipeline.py` runs every experiment in the report in order: the zero-shot baseline, the LoRA sweep, the five-fold BLIP runs, robustness, the ablation study, the training-length cycles and the served model. It skips finished stages, resumes an interrupted run from its last checkpoint, gives each parallel run its own GPU when there are enough, and writes runs to `./work/runs`. Run `make smoke` first. In a few minutes it runs every training path on a handful of pairs, then zero-shot BLIP scoring and the attention heatmaps, so a broken setup fails before the long runs start.

```sh
make smoke
make pipeline
make pipeline ARGS="--only sweep blip --tasks caption --folds 0"   # one change, checked on one fold first
```

The whole pipeline is several hundred GPU-hours. To train only the final VQA model on fold 0:

```sh
uv run --extra train python -m farmvlm.train_blip --task vqa --stats data/stats.npz --root data/av/Agriculture-Vision-2021 \
    --fold 0 --rank 64 --lr 2e-4 --targets attn_qv --pairs 1920000 --eval-every 320000 \
    --val-tiles 1000 --val-protocol test --select ranked --patience 99 --out work/runs/c6-vqa-f0
```

Score it on the held-out fold with `scripts/eval_blip.py --folds 0 --vqa-adapter work/runs/c6-vqa-f0/best ...`.

With `HF_TOKEN` set, the pipeline also pushes every run to a private runs repo on the Hub (`FARMVLM_HF_REPO`, ours by default) and resumes from it on another machine. Without a token, runs stay local.

## Report

`make report` rebuilds the report's tables, figures and headline numbers from the run records in `runs/`. `make mirror` fills `runs/` from the Hub runs repo, and `make report RUNS=work/runs` reads your own pipeline's runs instead. `make pdf` compiles the report and the deck with [tectonic](https://tectonic-typesetting.github.io/).

## Layout

| Path | Contents |
|---|---|
| `farmvlm/` | The package: QA scheme, mask statistics, QA items and text, field-grouped folds, balanced sampler, BLIP and VLM training and reading, evaluation, heatmaps, report tables |
| `scripts/` | Data download, the GPU smoke test and pipeline, evaluators, bake-off, GGUF export, run comparison, report figures, demo tiles |
| `app/backend/` | FastAPI backend. `models.py` is the model registry, `blip.py` and `qwen.py` run the models, `engine.py` streams their answers, `field.py` fetches satellite context |
| `app/frontend/` | React and TypeScript map-and-chat UI on MapLibre and OpenFreeMap |
| `report/` | LaTeX report and Beamer deck |
| `tests/` | `pytest` checks on synthetic masks and runs |
