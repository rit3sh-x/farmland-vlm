# Farmland VLM

Captioning and question answering about crop anomalies in aerial farmland imagery, for AI 512 Project 3.

## Language

### Data

**Tile**:
One Agriculture-Vision image: a 512×512 aerial view of part of a field, RGB plus near-infrared, at 10–15 cm per pixel.
_Avoid_: image, patch, crop (clashes with the plant sense)

**Quadrant view**:
One 256×256 quarter of a tile, enlarged to tile size, with its own labels taken from that quarter of the masks. It is the only source of anomaly-free images.
_Avoid_: crop, patch

**Field**:
The farm parcel a tile was cut from; many tiles share one field. Identified by the dataset manifest, not the tile's filename.
_Avoid_: farm, parcel

**Flight**:
One aerial capture of a field on one date. A tile's filename starts with its flight code; a field can have several flights.
_Avoid_: field ID (for the filename prefix)

**Anomaly**:
One of the dataset's annotated field patterns (e.g. drydown, nutrient deficiency, weed cluster, water), marked by a per-tile mask.
_Avoid_: stress, defect, class

**Extent**:
How much of a tile's in-field area one anomaly covers, binned as small, moderate or large. The masks mark where an anomaly is, not how bad it is, so extent is the only severity proxy they support.
_Avoid_: severity, mild/severe

**NDVI map**:
A per-pixel vegetation index computed from a tile's NIR and red bands, coloured so healthy crop is green and stressed or bare ground is yellow to red. It is relative, since the cameras are uncalibrated.
_Avoid_: spectral image, false colour (NIR shown as red, a different picture)

**QA pair**:
A question about a tile and its answer, generated from the tile's anomaly masks.
_Avoid_: sample, example

**Caption**:
A generated sentence describing which anomalies a tile shows.

**Location imagery**:
Imagery fetched from an outside source for a place the user picks, as opposed to a tile.
It is shown as context next to the model's answers, never fed to the model.
_Avoid_: satellite image (one source, NASA GIBS's MODIS/HLS or Sentinel-2, among several)

**Field box**:
The rectangle a user drags on the map to stand in for a real field's boundary. Location imagery and its numbers are cut to this box.
_Avoid_: field (that is the dataset's parcel)

### Models

**Baseline**:
BLIP used zero-shot, with no fine-tuning.

**BLIP-LoRA**:
BLIP fine-tuned with LoRA on QA pairs and captions; the graded comparison against the Baseline.

**Candidate model**:
A newer model considered for serving.

**Served model**:
The one fine-tuned model the app runs.
_Avoid_: best model, production model

**NDVI variant**:
The served model fine-tuned with a second adapter on RGB + NDVI map image pairs: the same QA pairs, an extra image. Used for comparison, and in the demo for tiles that carry NIR.
_Avoid_: NDVI model

**Adapter**:
The small set of LoRA weights a fine-tune produces, applied on top of an unchanged base model; merging folds it into the base weights instead.
_Avoid_: checkpoint (for the adapter alone)

### Evaluation

**Fold**:
One of five field-grouped partitions of the labelled tiles, regenerated from a seed for each experiment run. No field appears in two folds.
_Avoid_: split (for our partitions)

**Test fold**:
The fold held out from training in one cross-validation round.
_Avoid_: test split, test set (the dataset's own unlabelled test split, which is never used)

**Inner validation set**:
The ~10% of a training fold's fields held back for early stopping, checkpoint choice and the hyperparameter sweep, so the test fold stays untouched.
_Avoid_: dev set, validation split (the dataset's own "val" folder)

**Test sample**:
The fixed, class-stratified subset of a test fold that every model is scored on.

### Answers

**Ambiguous case**:
A QA pair or caption whose masks put it on a borderline: extent near a bin edge, an anomaly covering a tiny area, or a near-tie between locations. Flagged from the masks, not by judgement.
_Avoid_: hard case, edge case

**Hedged answer**:
An answer that states its uncertainty instead of committing, expected on ambiguous cases.
