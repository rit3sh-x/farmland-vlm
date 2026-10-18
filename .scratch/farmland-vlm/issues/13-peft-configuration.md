# PEFT configuration and sweep

Type: grilling
Status: resolved
Blocked by: 07

## Question

What is the starting fine-tune setup for each trained model (BLIP caption, BLIP VQA, the served model, and its RGB + NDVI row), and how does the fold-0 sweep justify it, as the brief asks ("choose and justify rank, target modules, and learning rate")?

- Rank, α, dropout, and target modules. This includes whether to adapt the vision tower for aerial imagery: that forces the merged-GGUF route for serving, since runtime adapters reach the language model only (see [Laptop feasibility check for shortlisted models](10-laptop-feasibility-check.md)).
- LR, schedule, epochs, and how many pairs each training run draws, plus the family mix and class weights for the balanced sampler from the [QA-pair and caption generation scheme](05-qa-pair-scheme.md).
- The image-token budget per image. It is shared by the RGB and RGB + NDVI rows and drives laptop latency: Qwen3.5-2B takes 12.4 s at 256 image tokens, 8.3 s at 144.
- fp16 stability levers for the loss-scale overflows seen on Qwen3.5 and Gemma 4 in the [Training feasibility check (Colab T4)](12-kaggle-training-check.md).
- The sweep grid and its budget on fold 0's inner validation set (per the [Evaluation protocol](06-evaluation-protocol.md)), and whether DoRA gets an optional extra row.
- Now unblocked by [Served model and backend stack](07-served-model-and-backend.md). The served model is picked by a bake-off between Qwen3.5-2B, LFM2.5-VL-1.6B and LFM2.5-VL-450M on ~10k equal pairs, so this ticket sets one starting setup the bake-off can use for all three, plus the BLIP-base setup. BLIP checkpoints: `blip-image-captioning-base` + `blip-vqa-base`; plain 16-bit LoRA.
- Constraint: the demo switches the RGB and NDVI-variant adapters per request, which needs language-model-only runtime adapters. Adapting the vision tower instead means two merged GGUFs swapped on demand (~3 s load).

## Answer

Locked with the user on 2026-10-19. Glossary term added to [CONTEXT.md](../../../CONTEXT.md): Inner validation set.

**Input resolution:** every model sees **384 px**: BLIP natively, the VLMs at ~144 image tokens. The NDVI variant gets two 384 px images.
- This keeps the BLIP vs served-model comparison at equal pixels.
- Laptop Qwen3.5-2B caption: 12.4 → 8.3 s.
- Training is est. ~1.5× faster than at 512 px.
- Accepted risk: row-level anomalies lose ~25% of their detail (~13 cm/px).

**BLIP (graded):** `blip-image-captioning-base` + `blip-vqa-base`.
- **Starting point:** r=16, α=2r, dropout 0.05; LoRA on `query`/`value` in the text model + `qkv` in the vision model; LR 1e-4; cosine schedule with 3% warmup; AdamW; batch 16.
- **Sweep:** one setting at a time on fold 0's inner validation set, ~10k pairs per run (~10 min on a T4), for both BLIP models. ~20 runs, ~3.5 GPU-h. The best value of each step is kept before the next:
  1. rank {4, 8, 16, 32, 64}, α kept at 2r
  2. LR {5e-5, 1e-4, 2e-4, 5e-4}
  3. targets {text attention only · text + vision attention · all linear layers}

  It produces one line plot each (score vs rank, LR, targets): the brief's "justify" evidence, table 6 in the [Evaluation protocol](06-evaluation-protocol.md).
- **Final runs:** up to 120k pairs per model per fold.
- **DoRA:** one optional row on BLIP VQA, fold 0, at the swept settings (~1.2× LoRA cost). First to cut.

**Served model:** the [Served model and backend stack](07-served-model-and-backend.md) bake-off candidates (Qwen3.5-2B, LFM2.5-VL-1.6B, LFM2.5-VL-450M).
- **Settings:** r=16, α=32, dropout 0.05, LR 1e-4, the same cosine schedule. LoRA on all linear layers of the **language model only**, giving runtime adapters for per-request switching. The same settings for all candidates; no served-model rank sweep.
- **Effective batch 16** via gradient accumulation:

  | Model | Batch × accumulation |
  |---|---|
  | Qwen3.5-2B | 2 × 8 |
  | LFM2.5-VL-1.6B | 4 × 4 |
  | LFM2.5-VL-450M | 16 × 1 |

- **Bake-off:** 10k pairs each at 384 px, ~1.5 GPU-h (est.).
- **Vision test:** one more short run (10k pairs) of the winner with LoRA also on the vision tower. If it beats the language-model-only run by ≥ ~2 points on inner validation, the served model takes the merged-GGUF route: one merged model per variant, swapped on demand (~3 s). Otherwise it stays on runtime adapters.
- **Final runs:** up to 60k pairs, for the RGB model and the NDVI variant (Qwen3.5-2B worst case: ~4.5 h and ~7 h).

**Every final run**
- The inner validation set is scored every 5k pairs and the best checkpoint is kept. Training stops early after 3 checks without improvement.
- Each check point saves the adapter + optimizer + scheduler + RNG state + step, so a run resumes in the next Colab session. Adapters are 22–34 MB, ~2–3 min via `colab download`.

**Sampler**
- Family mix:

  | Model | Pairs drawn |
  |---|---|
  | Served model | captions 30%, presence 30%, identify 15%, extent 15%, location 10% |
  | BLIP VQA | the same ratios without captions (presence 43%, identify 21%, extent 21%, location 15%) |
  | BLIP caption | captions only |

- Class weights ∝ 1/√frequency; presence yes/no 50/50 per class; clean quadrant views ~12% of draws.
- Loss on answer tokens only.

**fp16 stability**
- Gradient clipping at 1.0.
- `GradScaler(init_scale=2**12)`: the training check's runs fell from 65,536 to 32.
- Normalisation layers kept in fp32.
- Rule: if more than 5% of steps are skipped over 1k steps, halve the LR and resume from the last checkpoint.

**GPU budget (est.):** ~45 GPU-h worst case, ~35 with typical early stops.

| Item | GPU-h (est.) |
|---|---|
| BLIP sweep | 3.5 |
| Bake-off | 1.5 |
| Vision test | ~0.7 |
| BLIP 5 folds × 2 models | ≤ 18 |
| Served RGB model | ≤ 4.5 |
| NDVI variant | ≤ 7 |
| DoRA | ≤ 2 |
| Eval | ~5 |

**Note, 2026-10-19:** weights travel through a private HF Hub repo (see [Training code layout and data logistics](15-training-code-and-logistics.md)), so the merged-GGUF route stays practical and the vision-test rule above stands.
