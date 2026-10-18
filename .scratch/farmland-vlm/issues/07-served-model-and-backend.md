# Served model and backend stack

Type: grilling
Status: resolved
Blocked by: 02, 03, 10, 12

## Question

Which shortlisted model does the app serve, with which PEFT method — and therefore is the backend Rust end-to-end or fully Python? Also: which BLIP checkpoints make up the graded Baseline / BLIP-LoRA (captioning and VQA, base vs large)?

- Does a C++ binding (llama.cpp or ONNX Runtime called from Rust) count as "Rust"? If yes, InternVL, SmolVLM2 and Moondream2 stay possible; if no, the served model must be one mistral.rs supports, or the backend goes Python.
- Must the demo serve BLIP-LoRA too, or only the served model? With a Rust backend, BLIP VQA needs a hand port in candle (estimated 1–3 days plus a parity test against HF).
- The cost of that choice is now measured: on this laptop, llama.cpp is 1.3–4× faster at prefill than mistral.rs and uses about half the RAM, on the same GGUF files. See [Laptop feasibility check for shortlisted models](10-laptop-feasibility-check.md).
- User leaning, 2026-10-19 (raised while working on [QA-pair and caption generation scheme](05-qa-pair-scheme.md)): start with Qwen3.5-2B, which would also carry the RGB + NDVI two-image comparison row. Weigh it against the measured costs: it is the slowest shortlisted model on the laptop (12.4 s cold caption at 256 image tokens) and in training (8.5 h per 75k-pair pass, ~2× that for the NDVI row), and its fp16 runs overflowed.

## Answer

Locked with the user on 2026-10-19. Glossary term added to [CONTEXT.md](../../../CONTEXT.md): NDVI variant.

**Served model: picked by a bake-off in build week 2–3**
- Candidates: Qwen3.5-2B, LFM2.5-VL-1.6B and LFM2.5-VL-450M.
- Each trains on the same ~10k pairs with the starting setup from [PEFT configuration and sweep](13-peft-configuration.md), at one shared image-token budget.
- All are scored on fold 0's inner validation set (per the [Evaluation protocol](06-evaluation-protocol.md)), for ~2.5 GPU-h (est.).
- Rule: take the fastest model within ~2 points of the best on mean VQA accuracy across the four families, with caption fact F1 as the tiebreak. "Fastest" means cold caption + answer time on the laptop in llama.cpp at that image budget.
- Gemma 4 E2B is dropped: 13 h per pass, batch 1 only, fp16 overflows.
- The RGB + NDVI comparison row (the NDVI variant) uses the winner.
- Why a bake-off: there is no task-quality data yet. After fine-tuning, small models often end up close, so speed may decide, and the report gets a model-selection table.

**PEFT method:** plain 16-bit LoRA (fp16 base, fp32 adapter). QLoRA isn't needed under 4B parameters. The optional DoRA row stays with [PEFT configuration and sweep](13-peft-configuration.md).

**Graded BLIP checkpoints:** `Salesforce/blip-image-captioning-base` + `Salesforce/blip-vqa-base` for both the Baseline and BLIP-LoRA. They were measured at 17–19 samples/s on the T4: ~9 GPU-h for 5 folds × 2 models at ~60k pairs per run (est.).

**Backend: fully Python, no Rust**
- FastAPI. The app starts and manages the prebuilt llama.cpp `llama-server` for the served model.
  - The RGB and NDVI-variant adapters are loaded at startup and switched per request.
  - Per-request adapter switching is untested: check it in build week 1.
- Zero-shot BLIP runs in transformers on CPU (f32) and is loaded on first use.
- Estimated RAM peak: ~6 GB of 7.8 (llama-server ≤3 GB + BLIP base ~2.5 GB).
- The API reuses the training and eval code: NDVI rendering, answer normalising, hedge parsing.
- Why not Rust: the candidates do run in pure Rust (mistral.rs), but measured 1.3–4× slower at prefill with ~2× the RAM. LFM2.5-VL-1.6B took ~28 s. Serving BLIP VQA would also need a hand port to candle.
- The question of whether C++ FFI counts as "Rust" is moot.

**What the demo serves:**
- the served model (RGB)
- its NDVI variant, for dataset tiles (which carry NIR)
- zero-shot BLIP (the Baseline), for a before/after view

BLIP-LoRA is not served. Once zero-shot BLIP is in, adding it is one more weight file.

**Constraint passed to [PEFT configuration and sweep](13-peft-configuration.md):** switching adapters per request needs language-model-only adapters, i.e. runtime adapters. If the vision tower is adapted, each variant becomes its own merged GGUF, swapped in on demand (~3 s load).
