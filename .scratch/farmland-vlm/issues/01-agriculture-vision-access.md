# Agriculture-Vision access and label structure

Type: research
Status: resolved
Blocked by: —

## Question

How do we get Agriculture-Vision into a Kaggle/Colab notebook, and what exactly is in it?

- Which release matches the brief's 94,986 images (2020 challenge set vs the 2021 full set), the download route (S3 / AWS CLI / registration), total size, and license or terms of use.
- File layout: RGB and NIR files, per-anomaly binary masks, boundary / valid-pixel masks, the exact anomaly type names and count, and which splits have public labels.
- Do filenames carry a field ID, so tiles can be grouped by field for leakage-free splits?
- Ground resolution per tile, per-anomaly tile counts and class imbalance, how often anomalies overlap in one tile.
- Is there a usable mirror on Kaggle Datasets or the HF Hub, so a notebook can avoid a huge download? Does a subset fit Kaggle's disk limits?

## Answer

The brief's 94,986 tiles are the full 2021 "supervised" release: one 21.0 GB `Agriculture-Vision-2021.tar.gz`, anonymous over S3/HTTPS or the official ungated HF repo `shi-labs/Agriculture-Vision`. It has 9 classes. Splits are 56,944/18,334/19,708, and test has no labels. IntelinAir's terms allow non-commercial research only, ban redistribution, and allow downloads only via the official links. That rules out the many Kaggle/HF mirrors.
Each tile has rgb/nir JPGs, a boundary PNG, a valid-pixel mask PNG and one binary PNG per class.
The filename `<ID>` is a flight code, not a field. The official `data_manifest.csv` maps 3,432 flights to 2,116 fields, and 480 of those fields straddle official splits. Group folds on the manifest's `field`.
Resolution is 10 cm/px (15 cm/px for 2017). Classes run from drydown at ~31% of tiles down to storm_damage at ~0.5%. On val, 6.2% of tiles have 2 or more classes.
Plan: stream the tar once and cache a small RGB subset (train+val ≈9 GB without NIR).
[findings](../research/agriculture-vision-access.md)
