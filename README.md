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
| `farmvlm/` | The package: QA scheme vocabulary, mask statistics, QA items and their text, field-grouped folds, the balanced sampler |
| `scripts/` | Colab session setup and data fetch |
| `tests/` | `pytest` checks on synthetic masks |

## Running

On a fresh Colab VM, this clones the code, installs it, streams the dataset and builds the per-view mask statistics:

```sh
curl -sL https://raw.githubusercontent.com/rit3sh-x/farmland-vlm/main/scripts/session_setup.py | python3 -
```

Locally:

```sh
uv venv && uv pip install -e ".[dev]" && .venv/bin/python -m pytest -q
```
