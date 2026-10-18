# Training code layout and data logistics

Type: grilling
Status: resolved
Blocked by: —

## Question

How is the training and evaluation code organised and moved between the laptop and Colab, and how does each session rebuild its data?

- **Code layout.** Python scripts kept on the laptop and pushed with `colab upload` / `colab exec` (as the feasibility checks did), notebooks, or both. The brief's deliverable is "well-documented Python notebooks/scripts" covering preprocessing, dynamic cross-validation, training and inference. Decide which entry points exist (stats pass, pair generator, fold builder, train, eval, export) and how a run is configured and logged.
- **Session rebuild.**
  - Each session re-streams the official tar (~13 min) and writes the per-tile stats file in the same pass.
  - The NDVI variant needs NIR, ~20 GB unpacked train+val on a 71 GB Colab disk.
  - Is the 384 px resize done once per session?
  - What comes home: stats file, adapters, logs, eval outputs.
- **Weights.** The default is runtime adapters (`colab download`). Only if the vision test wins does the merged-GGUF route need a private HF Hub repo and an HF token on the Colab VM: set that up only then, or now as insurance?
- **Run records.** Where results land so the report's tables can be rebuilt: per-run JSONL plus the fold seed.

Inputs: [PEFT configuration and sweep](13-peft-configuration.md) (checkpoint/resume rule, run list) and [Evaluation protocol](06-evaluation-protocol.md) (seeded folds, test samples).

## Answer

Locked with the user on 2026-10-19.

**Repo:** one **public GitHub repo** under `rit3sh-x` for the whole project. Colab `git clone`s it with no token. The project folder is not yet a git repo, so the build starts with `git init`. Layout:

```
farmvlm/        package: stats pass, pair generator, folds, sampler, train, eval, export,
                NDVI rendering, caption fact parser + answer normaliser (also imported by the API)
scripts/        one entry point per step + session_setup.py
notebooks/      01 data exploration · 02 results tables/figures · 03 inference demo (thin, import farmvlm)
app/backend/    FastAPI (see API contract)
app/web/        React + Vite "Map + chat" UI (see UI prototype); npm run dev proxies /api,
                and for the demo FastAPI serves the built files
report/         LaTeX report + Beamer deck
runs/           gitignored: downloaded adapters and run records
CONTEXT.md, project3.md, .scratch/ (planning; prototype tiles and node_modules excluded)
```

**Data guard**
- **Why:** the dataset terms forbid posting the data "in code repositories".
- `.gitignore` covers data folders, the stats file, adapters, checkpoints, `runs/`, `node_modules`, the prototype's tiles, and report figures that show tiles.
- A pre-commit hook:
  - rejects image files outside a UI-icon allowlist, and any file over 1 MB
  - strips notebook outputs, so no rendered tiles get in
  - scans for secrets (`hf_…`, GitHub tokens)

**Weights and checkpoints**
- One **private HF Hub model repo**, `rit3sh-x/farmvlm-runs`. Colab pushes to it:
  - every 5k-pair checkpoint (adapter + optimizer + scheduler + RNG + step)
  - final adapters plus their GGUF LoRA
  - run records and the stats file
- A new session resumes from the latest checkpoint there, and the laptop pulls final adapters with `huggingface_hub` (3.5–5.5 MB/s measured).
- The token is a fine-grained write token scoped to that repo, passed to each Colab VM as `HF_TOKEN` through the environment at session start, and never written into the repo or a notebook.
- `colab download` (0.28 MB/s) is a fallback only.
- **Why:** free Colab VMs can die mid-run. Pushing every checkpoint means nothing is lost, and it keeps the merged-GGUF route practical (~6 min instead of ~2 h per model).
- So the vision-test rule from [PEFT configuration and sweep](13-peft-configuration.md) stands: merged GGUFs if adapting the vision tower wins by ≥ 2 points.
- Weights stay private, which sidesteps any "derivative of the dataset" question.

**Each Colab session** (`scripts/session_setup.py`)
1. Pinned pip installs, plus `pip uninstall -y torchao`.
2. Clone the repo at a recorded commit.
3. Stream the official tar, train+val with NIR (~13 min). The same pass writes the stats file (~30 MB) and a 384 px cache in `/content/cache`: RGB JPEG + NDVI map, ~3–4 GB est.
4. Run the job.
- Base models download anonymously from HF on Colab and on the laptop.
- LoRA → GGUF conversion runs on Colab, so the laptop only fetches small adapter GGUFs.

**Run records:** `runs/<date>-<model>-<fold>-<sha>/{config.json, metrics.jsonl, eval.json, adapter/}`, each carrying the fold seed and commit hash. They are pushed to the HF repo and pulled to the laptop; `notebooks/02` rebuilds every report table from them.

**Checks for build week 1**
- Create the HF token and repo.
- Confirm push and pull throughput from Colab.
