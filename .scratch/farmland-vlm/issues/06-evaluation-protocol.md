# Evaluation protocol

Type: grilling
Status: resolved
Blocked by: 01, 05, 11, 12

## Question

How do we score the Baseline, BLIP-LoRA and the served model within the free GPU budget?

- What the brief's "dynamic cross-validation" means here: field-grouped k-fold? k, subset size, seeds, how splits are regenerated per run.
- Grouping facts: the official splits leak (480 of 2,116 fields have flights in more than one split), so folds group on the manifest's `field` column over train+val (test is unlabeled). Tiles per flight range from 1 to 387, so equal field counts won't give equal fold sizes.
- Metrics: BLEU / ROUGE (CIDEr?) for captions; VQA accuracy (exact match vs normalised, per question family).
- How hedging is scored on ambiguous cases, now that the [QA-pair and caption generation scheme](05-qa-pair-scheme.md) uses closed hedges (`possibly`, `small to moderate`, `X or Y`): exact match on the hedge only, or partial credit for committing to one side? How over-hedging on clear cases is penalised, and whether the logged token confidence gets a calibration figure.
- Scoring details the scheme leaves here: Identify is set match; captions get multi-reference BLEU/ROUGE (all 4–5 phrasings); storm_damage is reported on its own row; the clean quadrant slice is scored separately.
- Robustness suite: paraphrases are settled (held-out templates and name synonyms, per the scheme). Still open: image perturbations (noise, blur, JPEG, brightness) and their strengths.
- The served model's RGB vs RGB + NDVI comparison row: on which folds, at what cost.
- Optional domain-shift row: test on NAIP chips of the same fields, and possibly the satellite gap (research suggests some Agriculture-Vision 2021 raw fields also have 10 m Sentinel imagery; unconfirmed).
- Which tables and figures the report and deck need from all this.

## Answer

Locked with the user on 2026-10-19. Glossary terms added to [CONTEXT.md](../../../CONTEXT.md): Fold, Test fold, Test sample.

**Dynamic cross-validation**
- 5-fold `StratifiedGroupKFold` over the 75,278 train+val tiles, grouped by the manifest's `field` and stratified on each tile's rarest class.
- Fold sizes are balanced by tile count. Quadrant views stay with their tile.
- Folds are regenerated from a seed recorded at the start of each experiment run, and every model in the run shares them. No split file is saved or reused.
- The dataset's own unlabeled test split is never used.

**Who trains where**

| Row | Folds | Notes |
|---|---|---|
| Baseline | all 5 | inference only |
| BLIP-LoRA | all 5 | |
| Served model, RGB | fold 0 only | more folds if time allows |
| Served model, RGB + NDVI | fold 0 only | more folds if time allows |

Rough total: ~40 GPU-h (est., Qwen3.5-2B as the worst case, ~60k drawn pairs per training run). Draw counts and the budget split belong to the scope and GPU-budget fog.

**Tuning**
- Each training fold holds out ~10% of its fields, grouped, as an inner validation set for early stopping and checkpoint choice.
- A small reduced-budget sweep (rank, target modules, LR) runs once on fold 0's inner validation set, then the chosen settings are fixed for all folds. Its grid belongs to the PEFT configuration ticket.
- Test folds are never seen before final scoring.

**Test sample**
- Per fold, ~2,000 test tiles with at least ~50 tiles of every class where the fold has them.
- It includes their applicable pairs (presence yes/no balanced), every ambiguous pair in those tiles, and ~300 clean quadrant views.
- Every row is scored on the same sample. Per-family accuracy is good to about ±1–2 points.
- A 500-tile robustness subset is drawn from it.

**VQA scoring**
- Primary: the generated answer, strictly normalised, then exact match. Normalising covers case, punctuation, articles, centre/center and plurals, with no synonym map.
- Identify is scored as a set.
- Secondary, for Baseline and BLIP-LoRA only: a ranked score, where the family's fixed answer list is ranked by likelihood and the top answer taken. Those likelihoods double as the confidence numbers.

**Caption scoring:** BLEU-1/4, ROUGE-L and CIDEr (pycocoevalcap, no Java) against all 4–5 reference phrasings. Plus fact precision/recall/F1: class, extent and location mentions are parsed back out of each caption and compared with the masks.

**Hedging**
- On ambiguous pairs: the hedge scores 1, committing to either side of the borderline 0.5, and anything else 0.
- On clear pairs, a hedge counts as wrong.
- Report a 2×2 of hedge rate on ambiguous vs clear pairs, plus one calibration figure: a reliability diagram with ECE, and whether confidence is lower on ambiguous pairs.

**Robustness suite** (500-tile subset), reported as the drop from clean scores:

| Kind | Variants |
|---|---|
| Paraphrases | held-out templates and name synonyms, per the [QA-pair and caption generation scheme](05-qa-pair-scheme.md) |
| Photometric | Gaussian noise σ 5/15/30 (0–255), blur σ 1/2/4 px, JPEG quality 50/20/10, brightness/contrast ±10/25/40% |
| Resolution loss | downsample ×2/×3/×6 and scale back up (≈0.2/0.3/0.6 m), the scored stand-in for NAIP scale |
| Geometric | horizontal flip and 90° rotation, with Location answers remapped |

Served-model eval costs ~15 min per variant (est.).

**NAIP:** no scored row. The manifest has no coordinates, so labelled NAIP chips of the same fields can't be fetched. At most a few unlabelled qualitative examples, if [Role of location imagery in the app](08-location-imagery-role.md) keeps NAIP.

**Uncertainty**
- 5-fold rows: mean ± std across folds.
- Fold-0 rows: 95% bootstrap intervals that resample test fields.
- Served vs BLIP-LoRA, and RGB + NDVI vs RGB: a paired bootstrap of the difference on the same fold-0 sample.

**Results the report and deck need**

Tables:
1. Graded ablation: Baseline vs BLIP-LoRA, VQA per family (generated, with ranked in brackets) and the caption metrics, mean ± std over 5 folds.
2. Per-class presence accuracy and F1 for all 9 classes. storm_damage is marked low-sample, and the clean slice is reported separately.
3. Fold 0: BLIP-LoRA vs served model vs RGB + NDVI, with CIs and paired differences.
4. Hedging: the 2×2 rates and partial-credit accuracy, per row.
5. Robustness: the drop per perturbation and strength, plus the paraphrase drop, per row.
6. LoRA sweep on the inner validation set (rank, targets, LR): the brief's "justify" evidence.
7. Cost: trainable parameters, GPU-h, and laptop latency and RAM per model (from the feasibility checks).

Figures:
- reliability diagram
- robustness curves (score vs strength)
- qualitative grid: tile + mask overlay + NDVI map, with each model's caption and answers, including hedged and clean examples
- training and validation loss curves
- dataset class and extent distributions

**Amendment, 2026-10-19: three extra analyses.** The user asked for them because "the better and unique aspects … the more marks", with "AI remains the centre".

1. **Split-leakage study.**
   - Train one BLIP-LoRA VQA model on the dataset's **official train split**, with the swept settings and the same pair budget.
   - Score it on two halves of the official val split: tiles from fields that also appear in official train (seen-field, 174 + 85 fields straddle train/val) vs tiles from fields that don't (unseen-field). Both use the same stratified sampling as the test sample.
   - The gap measures how much the official split's leakage inflates scores. It anchors a "why field-grouped dynamic CV" section.
   - Cost: ~1–2 GPU-h.
2. **Selective answering (risk–coverage).** From the logged confidence, plot accuracy vs coverage per model (abstain on the least-confident x%), with AURC. Eval-only, no GPU. Reported next to the hedging results.
3. **"Where it looked" heatmaps.**
   - Use BLIP's text-to-image cross-attention, gradient-weighted, for the Baseline vs BLIP-LoRA, on the test sample.
   - Report a **pointing-game** score (does the heat peak fall inside the anomaly's mask?) and heat-in-mask energy, plus a figure: "did fine-tuning teach BLIP where to look?".
   - BLIP only: the served VLM runs in llama.cpp with no attention access.
   - In the demo, the heatmap is a toggle on BLIP readings.

New results:
- Table 8: leakage gap.
- Figures: risk–coverage curves, heatmap grid.
