# Schedule, GPU budget and cut-line

Type: grilling
Status: resolved
Blocked by: 14, 15

## Question

In what order does the ~7-week solo build run (from 2026-10-18), how do the ~35–45 GPU-h (est.) of runs fit into free Colab time (Kaggle ~30 h/week as fallback), and what gets cut first if time runs short?

- Week-by-week order, against the brief's plan (baseline by week 2, LoRA config by week 3, ablation by week 5, robustness by week 6, report week 7):
  - data/stats pass
  - zero-shot baseline
  - BLIP sweep
  - bake-off and vision test
  - BLIP 5 folds
  - served model and NDVI variant
  - eval and robustness
  - backend, UI, location panel
  - report and deck
- Which runs overlap with app-building weeks.
- The cut order, in candidate order:
  - DoRA row
  - raw-flight whole-field explorer
  - NDVI variant
  - served model's extra folds
  - location panel extras (season chart, offline cache)
  - UI polish
  - served-model fine-tune itself

  What is never cut: the graded core.
- Checkpoints in the schedule where a slipped step triggers a cut.

Inputs: [PEFT configuration and sweep](13-peft-configuration.md) (run list and GPU-h), [Evaluation protocol](06-evaluation-protocol.md), [API contract](14-api-contract.md) and [Training code layout and data logistics](15-training-code-and-logistics.md) (build size).

## Answer

Locked with the user on 2026-10-19.

**Week plan.** ML first, so the graded parts land on the brief's dates. GPU runs go unattended (`nohup`, checkpoints pushed to HF) while the app is built in parallel.

| Week | ML track (GPU-h, est.) | App and report track | Brief milestone |
|---|---|---|---|
| **W1** Mon 19–Sun 25 Oct | `git init` + data/secret guard; HF repo + token; Kaggle API token + check that `scripts/` runs as a Kaggle notebook; `session_setup.py`; stats pass, pair generator, folds, sampler; set cut points and ambiguity margins from the stats; week-1 checks: per-request LoRA, prompt cache on adapter switch, HF push/pull speed, `colab upload` speed (~1 h) | — | setup + dataset notes |
| **W2** 26 Oct–1 Nov | eval harness (metrics, caption fact parser, normaliser, hedge scoring, field-grouped bootstrap); **zero-shot baseline**, 5 folds (~1); BLIP sweep starts | FastAPI skeleton, `llama-server` management, BLIP on CPU | **baseline** |
| **W3** 2–8 Nov | sweep done → **LoRA config locked** (3.5); bake-off + vision test → served model picked (2.2); BLIP folds 0–1 | React "Map + chat" UI against the real API (base models) | **LoRA config** |
| **W4** 9–15 Nov | BLIP folds 2–4 (≤18 for W3–W4); served model RGB (≤4.5) | field context from the hosted APIs; showcase cache | training |
| **W5** 16–22 Nov | NDVI variant (≤7); DoRA (≤2, optional); **ablation + hedging** eval (~3) | wire in the real adapters, "Compare with", polish | **ablation + hedging** |
| **W6** 23–29 Nov | **robustness** suite (~2) | report draft + Beamer deck; demo hardening (offline, errors) | **robustness + draft** |
| **W7** 30 Nov–6 Dec | no new runs | finish report + deck, rehearse the viva; buffer | **submit** |

GPU need: ~6–12 h/week in W2–W6, ~35–45 GPU-h in total (est.). Colab free is primary; Kaggle T4×2 (30 h/week) is ready from W1 as the fallback.

**Cut order** (first to go → last):
1. DoRA row
2. raw-flight whole-field explorer
3. NDVI variant (training + demo)
4. extra served-model folds
5. field-context extras (season chart, weather strip, soil line, offline cache); keep the Sentinel-2 cut-out + summary
6. UI polish
7. served-model fine-tune itself; the demo then serves **BLIP-LoRA** (fold 0) through the same API and UI

**Never cut:** Baseline, BLIP-LoRA 5-fold ablation, hedging tests, robustness tests, dynamic cross-validation, LaTeX report, Beamer deck.

**Checkpoints** (reviewed each Sunday)

| Date | Check | If it slipped |
|---|---|---|
| **Sun 8 Nov** | BLIP config locked? Served model picked? | Freeze the sweep at the steps done and justify the rest by citation. Without a bake-off, take LFM2.5-VL-450M |
| **Sun 15 Nov** | All 5 BLIP folds trained? Served RGB model trained? | Remaining folds use the 60k cap. If the served model hasn't started, cut items 1 and 3 |
| **Sun 22 Nov** | Ablation + hedging tables exist? Demo works end-to-end on real models? | Cut items 3–5 |
| **Sun 29 Nov** | — | Feature freeze: the report draft is complete; anything unfinished is cut in order |

**Amendment, 2026-10-19.** The user added the three AI extras from the [Evaluation protocol](06-evaluation-protocol.md) amendment and said "AI remains the centre of it all". So ML work wins every trade-off against app work.

**Placement**
- Leakage study: W4, alongside the BLIP folds.
- Risk–coverage: W5, with the hedging eval.
- Heatmaps: W5–W6. BLIP inference only; the report figure first, then the UI toggle.

**Revised cut order** (first to go → last):
1. raw-flight whole-field explorer
2. field-context extras (season chart, weather strip, soil line, offline cache)
3. UI polish
4. DoRA row
5. extra served-model folds
6. heatmap toggle in the UI (the report figure stays)
7. NDVI variant
8. served-model fine-tune; the demo then serves BLIP-LoRA

**Never cut:** the graded core, plus the leakage study and the risk–coverage curve, which are cheap and AI-central.
