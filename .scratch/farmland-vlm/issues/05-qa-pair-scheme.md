# QA-pair and caption generation scheme

Type: grilling
Status: resolved
Blocked by: 01

## Question

What exactly do we generate from each tile's anomaly masks?

- Caption templates: which anomalies, how much of the tile, where in the tile.
- Question families (presence yes/no, which anomalies, extent / severity, location) and the answer vocabulary.
- How severity is derived (mask area thresholds?), what counts as an ambiguous case (near a threshold, overlapping anomalies, ...), and what a hedged target answer looks like.
- RGB only, or use NIR too (false-colour NIR-R-G, NDVI composite, or a second image at double the image-token cost)? BLIP and every shortlisted model take RGB input only.
- Pairs per tile and class balancing. Classes are heavily skewed (drydown in ~31% of tiles, storm_damage in ~0.5%), and 6.2% of val tiles carry 2+ classes.
- Every tile has at least one anomaly, so "no" answers come only from the classes a tile lacks. Is that enough, or do we need anomaly-free crops?
- What to do with storm_damage: the challenge never scores it, and it's very rare.

## Answer

Locked with the user on 2026-10-19. Glossary terms added to [CONTEXT.md](../../../CONTEXT.md): Extent, NDVI map, Quadrant view, Ambiguous case.

**Input image**
- Every model reads the RGB tile.
- The served model gets one extra comparison row whose input is RGB + NDVI map as two images. Its pairs and per-image image-token budget are the same as the RGB row's. It costs ~2× image tokens (Qwen3.5-2B est.: ~1.3 samples/s, ~16 h per 75k-pair pass, ~20 s laptop caption), so it is the first row to cut. BLIP takes one image, so the graded BLIP rows stay RGB.
- The NDVI map uses one fixed colour map and range. The UI shows the same map next to the RGB tile.
- Why: annotators drew the masks while looking at RGB, NIR and NDVI views (paper §3.3). NIR channels added 0.4–4.4 test mIoU for the paper's CNN (Table 5: ResNet-50 RGB 38.3 → NRG 41.3 / NRGB 41.8; ResNet-101 39.6 → 40.1 / 43.7). No published VLM comparison was found.
- NDVI from these JPEGs is relative: uncalibrated cameras, different rigs per year.

**Classes**
- All nine, per the brief. storm_damage is reported on its own low-sample row.
- Display names: double plant, drydown, endrow, nutrient deficiency, planter skip, storm damage, standing water (dataset `water`), waterway, weed cluster.

**Area basis:** a class's pixels inside field boundary ∧ valid mask, as a share of in-field pixels. Each class is counted independently, so overlapping labels count for both.

**Question families**, all with short closed-vocabulary answers scored by exact match after normalising:

| Family | Asked about | Answer |
|---|---|---|
| Presence | every class | yes / no |
| Identify | the tile or view | class names, largest first; `none` for a clean quadrant view. Scored as a set (order ignored) |
| Extent | present classes | small / moderate / large. One global pair of cut points for all classes, set in week 1 near the tertiles of the measured area distribution, rounded, then fixed |
| Location | present classes | the 3×3 grid cell with the most mask pixels (top left … bottom right, centre), or `throughout` when no cell holds ≥ ~40% of the mask. Positions are relative to the image, not compass directions |

Counting questions are excluded: the masks don't separate instances.

**Captions**
- Every anomaly present, largest first, each with its extent and location. Example: "Large drydown throughout the tile, with a small weed cluster at the top left."
- Each pattern has 4–5 template phrasings. Training samples one; BLEU/ROUGE use all of them as references.
- A clean view is captioned "No anomalies visible."

**Ambiguous cases**, flagged from the masks; exact margins are set in week 1 from the measured distributions:
- extent within ~±25% (relative) of a cut point
- presence where the class covers under ~1% of the in-field area
- location where the top cell barely beats the runner-up, or sits near the 40% `throughout` cut

Overlapping labels are not a flag on their own.

**Hedged targets:** flagged pairs train on a hedge from a fixed set:

| Case | Hedge |
|---|---|
| tiny-mask presence | `possibly` |
| near-threshold extent | `small to moderate` / `moderate to large` |
| location near-tie | `X or Y` |

Identify and captions reuse the same wording, e.g. "drydown, possibly weed cluster". Per-token confidence is logged for analysis. There is no run-time confidence rule.

**Clean examples**
- Each tile's four 256×256 quadrants are enlarged to 512 and relabelled from that quadrant's masks.
- Clean quadrants give `none` / all-`no` / "No anomalies visible." pairs. Quadrants that contain anomalies stop zoom level from becoming a shortcut for "clean".
- They make up ~10–15% of training pairs, and the test set gets a small clean slice.

**Volume and balance**
- Generate the full pool of every applicable pair for every tile and quadrant view.
- Each epoch draws a fixed number of pairs with a balanced sampler:
  - presence yes/no 50/50 per class
  - rare classes upweighted (e.g. weight ∝ 1/√freq)
  - a fixed mix of families
- The GPU budget sets the draw count. Test sets are fixed and stratified.

**Wording:** ~8 hand-written templates per family. About 5 are used for training; about 3 are held out, with name synonyms ("weeds", "standing water"), for the paraphrase robustness test.

**Output shape**
- One pass while the tar streams writes a per-tile and per-view stats file: area shares, grid-cell shares, flags (~30 MB).
- A separate generator turns stats into pairs, so retuning cut points or margins in week 1 doesn't need another re-stream.
- The stats file stays private: Colab or the laptop only.

Side inputs recorded on other tickets:
- Qwen3.5-2B leaning → [Served model and backend stack](07-served-model-and-backend.md)
- RGB and NDVI side by side → [Role of location imagery in the app](08-location-imagery-role.md)
