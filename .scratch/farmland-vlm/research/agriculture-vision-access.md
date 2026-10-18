# Agriculture-Vision: access, layout and label structure

Research for ticket [01-agriculture-vision-access](../issues/01-agriculture-vision-access.md). As of 2026-10-18. Bracketed numbers like [3] point to Sources at the end. **UNVERIFIED** marks anything not confirmed in a primary source. **est.** marks my own estimates. **derived** marks numbers I computed from official metadata files (scripts and method under "How the numbers were derived").

## Summary

1. **Release.** The brief's 94,986 images are the **full CVPR 2020 paper dataset** [1], published as the **2021 challenge "supervised" set**: one file, `Agriculture-Vision-2021.tar.gz`, **21,033,702,578 bytes (21.0 GB)** [4][5]. It is *not* the 2020 challenge subset (21,061 images, 6 classes including cloud shadow, ~4.4 GB, access by request [3][6]). Download is anonymous, with no registration: `aws s3 cp s3://intelinair-data-releases/agriculture-vision/cvpr_challenge_2021/supervised supervised --no-sign-request --recursive` [5], plain HTTPS from the same bucket, or the official, ungated HF repo `shi-labs/Agriculture-Vision`, which holds a byte-identical file [3][7]. **License:** IntelinAir's Terms of Use. It allows non-commercial academic research only. It bans redistribution, and §1 says the dataset may be obtained **only through the officially provided link** [8].
2. **Layout.** `Agriculture-Vision-2021/{train,val,test}/` contains `images/rgb/*.jpg`, `images/nir/*.jpg`, `boundaries/*.png` (field boundary), `masks/*.png` (valid pixels) and, for **train and val only**, `labels/<class>/*.png`. That is one binary mask per class per tile [3][7][derived]. There are **9 classes**: `double_plant, drydown, endrow, nutrient_deficiency, planter_skip, storm_damage, water, waterway, weed_cluster`. The challenge scores 8 classes plus background and ignores `storm_damage` [1][9]. Splits are **56,944 / 18,334 / 19,708** train/val/test [1]. Test labels are not public [3].
3. **Field ID.** Filenames are `<ID>_<x1>-<y1>-<x2>-<y2>.{jpg,png}` [3][5]. The README calls `<ID>` the field id, but it is really a **flight code**: one aerial capture of a field on one date. The official manifest maps the 3,432 codes to **2,116 physical fields** [10][derived]. **480 of those fields have flights in more than one official split**, covering 1,474 of 3,432 flight codes. So the official splits leak across revisits of the same field. To get leakage-free splits, group on the manifest's `field` column, not the filename prefix.
4. **Resolution and statistics.** Ground resolution is 15 cm/px for 2017, 10 cm/px for 2018 (its blue channel was 20 cm/px, upscaled) and 10 cm/px for 2019 [1]. A later paper describes the tiles as "pre-processed and scaled", 10–20 cm/px [10]. The tile's year comes from the manifest. Per-class tile counts are heavily skewed: drydown ≈29.5k down to storm damage ≈0.8k (read off the paper's chart) [1]. Classes overlap, even within one pixel [9]. On the val split, **6.2% of tiles carry 2 or more classes** [derived]. The commonest pair is double_plant + endrow.
5. **Mirrors.** Kaggle has several full re-uploads (~21.35 GB) and per-split ones (train 13.0 GB, val 4.2 GB, test 4.3 GB). HF has third-party tar and parquet copies [11][12]. **All of them conflict with Terms §1 and §3**, and some are relabeled MIT/CC0/Apache [8][11]. The HF parquet copy also drops the filenames, which makes field grouping impossible. The terms-compliant path is to stream the official tar into the notebook and extract only what you need. Train+val unpacks to ≈17 GB with NIR and ≈9 GB RGB-only (est.). For comparison, Kaggle's saved `/kaggle/working` holds 20 GB and its scratch size is unpublished (see [free-gpu-limits](free-gpu-limits.md)). So extract once, then cache a small preprocessed subset as notebook output.

## 1. Which release matches 94,986

| Release | Images | Classes | Size | Access | Source |
|---|---|---|---|---|---|
| CVPR 2020 paper dataset ("Agriculture-Vision") | **94,986** (56,944 / 18,334 / 19,708) from 3,432 "farmland images" (flights), 2017–2019, mostly corn/soy, Illinois/Iowa | 9 | — | the paper describes it; released later as the 2021 set | [1] |
| 2020 challenge set | 21,061 (12,901 / 4,431 / 3,729) | 6: cloud_shadow, double_plant, planter_skip, standing_water, waterway, weed_cluster | ~4.4 GB | "Please contact us to get access" | [3][6][9] |
| **2021 challenge supervised set** (`Agriculture-Vision-2021.tar.gz`) | 94,986 (derived, see below) | 9 (8 evaluated + storm_damage) | **21,033,702,578 B** | anonymous S3 / official HF | [4][5][7] |
| 2021 challenge raw set | 261 full-field flights from 54 fields, 2017–2020, 10 cm/px, 4 × GeoTIFF bands + boundary | none | 299 GB | anonymous S3 | [5][derived] |
| Extended AV / TMLR 2023 benchmarks | 3,600 extra unlabeled full-field flights | none | ~5.26 TB | anonymous S3 | [10][derived] |

The 2021 tar is the brief's dataset. The evidence:
- The README's download section points only to the 2021 file ("around 20 GB") [3].
- The 2021 challenge page uses the 9 class IDs `0 background, 1 double_plant, 2 drydown, 3 endrow, 4 nutrient_deficiency, 5 planter_skip, 6 water, 7 waterway, 8 weed_cluster` and says `storm_damage` "will not be evaluated", "following our paper" [9].
- The file count matches the paper exactly. HF's dataset viewer indexed a byte-for-byte re-upload of this tar as 1,057,446 files [12]. That equals 94,986 tiles × 4 files (rgb, nir, boundary, mask) + 75,278 train+val tiles × 9 label masks = 379,944 + 677,502 [derived].
- Every flight-code prefix I sampled from the archive's train/val/test folders sits in the same split in the paper's official split files [10][derived].

Note the brief's wording: it says "standing water", which is the 2020 class name. The 2021 class is `water`.

**Download routes.** All three are verified with HEAD or list requests only.
- **S3:** bucket `intelinair-data-releases`, key `agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz`. It is public-read with unsigned listing, supports byte ranges (`Accept-Ranges: bytes`) and was last modified 2023-03-02 [4]. You don't need the AWS CLI: `https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz` works with `curl` or `wget` [4]. A second copy sits under `cvpr_challenge_2021_full/supervised/` with the same size [4].
- **Official HF:** `shi-labs/Agriculture-Vision` is `gated: false`, has one file `Agriculture-Vision-2021.tar.gz` (`X-Linked-Size: 21033702578`) and no dataset viewer or parquet export [3][7]. `huggingface_hub.hf_hub_download(repo_id="shi-labs/Agriculture-Vision", filename="Agriculture-Vision-2021.tar.gz", repo_type="dataset")` should work without a token (est.; ungated status verified [7]).
- **Registration:** none needed for the download. The Codalab registration on the 2021 page was only for leaderboard submissions [9].

**Terms of use** [8]. The same 3-page PDF sits in every S3 prefix. The README links a Drive copy and says "by downloading or using the dataset, user signifies its agreement" [3].
- §2: "perpetual, non-exclusive, … revocable license to download and use the Dataset for the sole purpose of developing non-commercial research applications."
- §1: "The User may only acquire the Dataset through the officially provided link. Obtaining the Dataset or its derivatives in any other format, from any other source, or by any other means is expressly prohibited."
- §3: no commercial use; no redistribution "(except to research employees, assistants and colleagues who also agree to be bound)"; (e) do not "redistribute or post the Dataset in any manner including but not limited to code repositories, blogs, websites, or social media"; you must cite the paper.
- §6: IntelinAir may collect the user's contact information. Anonymous S3 collects none in practice.
- What this means for us: course use is fine. Don't post the data or generated tiles anywhere public, and keep the demo local, as map.md already plans. Whether a *private* Kaggle dataset of our preprocessed subset counts as "posting" is not stated: UNVERIFIED (see Open questions).

## 2. File layout and label semantics

```
Agriculture-Vision-2021/
  train/  (56,944 tiles)   val/  (18,334)   test/  (19,708, no labels/)
    images/rgb/<ID>_<x1>-<y1>-<x2>-<y2>.jpg     512x512 RGB
    images/nir/<ID>_<x1>-<y1>-<x2>-<y2>.jpg     512x512 single-channel NIR
    boundaries/<...>.png                        farmland boundary map
    masks/<...>.png                             valid-pixel mask
    labels/<class>/<...>.png                    one binary mask per class (train/val only)
```
Sources: the folder names come from the archive's tar headers and the HF viewer keys [4][7][12]. The file extensions and per-folder counts come from a file listing (derived; see §6). The 512×512 size, the RGB+NIR channels, and the boundary/mask roles are in [1][3]. JPG for rgb/nir is stated in [1] and confirmed by the listing.

- **Boundary vs mask.** "The boundary map indicates the region of the farmland, and the mask indicates valid pixels in the image. Regions outside of either the boundary map or the mask are not evaluated" [3]. For QA generation, count an anomaly pixel only where `label & boundary & mask` [3]. Exact pixel values (0/255 vs 0/1) are UNVERIFIED until a tile is opened. Invalid regions are blacked out in the imagery [1].
- **Labels are not mutually exclusive.** Patterns "are stored separately as binary masks due to potential overlaps" [3]. "A pixel can contain more than one pattern" [9]. Every train/val tile has all 9 label files, and the empty ones are blank [derived]. So "class present" means a non-empty mask, not that a file exists.
- **Every tile is cropped around at least one annotation.** Small patterns are centered in a 512 window. Large ones are covered by non-overlapping sliding windows, and windows with only background are dropped. Tiles with more than 90% annotated pixels are dropped. Of two tiles with IoU > 30%, the one with fewer annotated pixels is dropped [1]. Two consequences for VQA: (a) "Is there any anomaly?" is almost always *yes*, so negatives for a class must come from tiles of other classes; (b) a field's tiles may still overlap a little (IoU ≤ 30%), and the `x1-y1-x2-y2` coordinates let you detect that.
- **storm_damage**: present in train/val, very rare, and not evaluated in the paper or challenge [1][9]. Whether to keep it as a QA class is a design choice for the QA-scheme ticket.

## 3. Field IDs and leakage-free grouping

- The README and website: "Each field id uniquely identifies the farmland that the image is cropped from, and (x1, y1, x2, y2) is a 4-tuple indicating the position in which the image is cropped" [3][5].
- The paper: "We first randomly split the 3,432 farmland images with a 6/2/2 train/val/test ratio … no cropped images from the same farmland will appear in multiple splits" [1]. The 3,432 "farmland images" are **flights**. The official `data20{17,18,19}_splits.json` files list exactly 405+135+135, 860+286+286 and 795+265+265 = 3,432 codes [13][derived].
- The official `data_manifest.csv` (`neurips_benchmarks_2021/`, the same file sits under `TMLR_benchmarks_2023/`) has columns `flightcode, field, date, split, year, dataset` [10][14]. The Extended AV paper says "the original dataset can be recreated from this new dataset by extracting the tiles at the appropriate pixel coordinates provided in the data manifest" and notes "data revisiting: capturing images from the same locations multiple times" [10]. All 54 field folders in the official raw release (`raw/<field>/`) appear as `field` values in the manifest [5][derived], so `field` is the physical-field key.

What the manifest shows (derived, supervised rows only):

| Quantity | Value |
|---|---|
| flight codes (= filename `<ID>`) | 3,432, all in manifest, splits agree 3,432/3,432 with the split JSONs |
| physical fields (`field`) | 2,116 |
| flights per field | 1: 1,391 fields · 2: 394 · 3: 183 · 4: 82 · 5: 39 · 6–8: 26 · 14: 1 |
| fields whose flights span >1 official split | **480** (train+val 174, train+test 165, all three 85, val+test 56) |
| flight codes inside those fields | 1,474 of 3,432 |
| fields flown in more than one year | 289 |
| fields per official split if grouped | train 1,488 · val 596 · test 597 (they sum past 2,116 because of the overlap) |

**Recommendation:** key = `manifest.field[ID]`. Run `GroupKFold` or `StratifiedGroupKFold` on train+val tiles (75,278) with that key, then rebuild splits each run. That also meets the brief's "dynamic cross-validation; static splits discarded" [2]. Ignore the official train/val boundary. Don't use test for supervised evaluation, because it has no labels. Note that `flightcode` alone is not enough as a key.

## 4. Resolution, class balance, overlap

**Ground resolution** (paper Table 2 [1]):

| Year | Channels | GSD | Camera |
|---|---|---|---|
| 2017 | N, R, G, B | 15 cm/px | 2 × Canon SLR |
| 2018 | N, R, G | 10 cm/px | 2 × Nikon D850 |
| 2018 | B | 20 cm/px, upscaled to align | Nikon D800E |
| 2019 | N, R, G, B | 10 cm/px | WAMS |

The Extended AV paper calls the tiles "pre-processed and scaled", 10–20 cm/px [10]. Whether 2017 tiles were resampled to 10 cm/px is UNVERIFIED. A 512 px tile spans about 51 m (10 cm/px) to 77 m (15 cm/px) on a side (est.). Val tiles by year (via manifest, derived): 2018 12,507 (68%) · 2019 3,779 (21%) · 2017 2,048 (11%). So about 89% of val tiles are 10 cm/px captures.

**Per-class tile counts, whole dataset** (read from paper Fig. 6, "Number of images containing each annotation class"; ±~300 reading error) [1]:

| Class | ≈ tiles | ≈ % of 94,986 |
|---|---|---|
| drydown | 29,500 | 31% |
| nutrient_deficiency | 21,200 | 22% |
| weed_cluster | 16,800 | 18% |
| double_plant | 10,700 | 11% |
| endrow | 7,800 | 8% |
| waterway | 5,800 | 6% |
| planter_skip | 4,700 | 5% |
| water | 4,000 | 4% |
| storm_damage | 800 | 1% |

By pixel area (paper Fig. 5) the skew is stronger: drydown ≈3.3×10⁹ px, nutrient deficiency ≈1.8×10⁹, weed cluster ≈1.3×10⁹, and every other class ≤0.3×10⁹ [1]. Paper Fig. 7 shows the share of each tile that is annotated: 0–20%: ~44.7k tiles · 20–40%: ~21k · 40–60%: ~14k · 60–80%: ~9.4k · 80–100% (capped at 90%): ~5.8k [1]. That is useful for "small vs. large patch" caption wording.

**Overlap.** The per-class counts above add up to about 101k for 94,986 tiles, and every tile has at least one label, so the dataset has at least ~6k extra tile-class pairs (est.). The paper gives no exact co-occurrence rate. My measurement on the val split (derived, method in §6). It covers 18,334 tiles and 238,342 files, which is 13 files per tile as expected. 0 tiles came out with no class, which matches the paper's cropping rule and supports the method:

| Class (val) | tiles with non-empty mask | % of val |
|---|---|---|
| drydown | 5,800 | 31.6% |
| nutrient_deficiency | 3,883 | 21.2% |
| weed_cluster | 2,834 | 15.5% |
| double_plant | 2,322 | 12.7% |
| endrow | 1,755 | 9.6% |
| planter_skip | 1,197 | 6.5% |
| water | 987 | 5.4% |
| waterway | 696 | 3.8% |
| storm_damage | 89 | 0.5% |

| Classes per tile (val) | tiles |
|---|---|
| 1 | 17,189 (93.8%) |
| 2 | 1,062 |
| 3 | 82 |
| 4 | 1 |
| **≥2** | **1,145 (6.2%)** |

The most common co-occurring pairs are double_plant+endrow (511), double_plant+weed_cluster (184), endrow+weed_cluster (171), endrow+planter_skip (128) and double_plant+planter_skip (121). That fits planting artifacts clustering at field edges. Val has 656 flights from 572 physical fields; tiles per flight are min 1, median 12, max 387. Folds therefore need tile-count balancing, not just field counts. Other per-tile facts (est., from constant-size files): 99% of val `masks/` are constant, i.e. fully valid, and 62% of `boundaries/` are constant, i.e. the tile sits entirely inside the field.

These per-class shares match the paper's whole-dataset chart within ±3 points, so the val figures are a fair stand-in until week 1 recomputes them from the masks.

## 5. Mirrors and fitting Kaggle/Colab disk

| Mirror | What | Size | Notes | Source |
|---|---|---|---|---|
| HF `shi-labs/Agriculture-Vision` | **official**, tar.gz | 21.03 GB | ungated, no viewer | [3][7] |
| HF `Myo48Kyal/…`, `asfdfdg/Agriculture-Vision` | re-uploaded tar (same byte size as official) | 21.03 GB | third-party | [12] |
| HF `Vadim21221/Agriculture_Vision`, `Babu0281/…`, `paarivendhan/…` | re-packed tar | 21.35–22.28 GB | third-party | [12] |
| HF `Vadim21221/agri-vision-2021-segmentation` | parquet, train 56,944 + val 18,334 | 16.81 GB | **no filename, boundary or mask columns**, so field grouping and valid-pixel masking are impossible | [12] |
| Kaggle `nauxqouh/agriculture-vision-2021-{train,val,test}` | extracted, one dataset per split | 12.96 / 4.20 / 4.29 GB | labeled CC0 (wrong) | [11] |
| Kaggle `dwchen/…`, `a10samhouri/…`, `nilanjanat/agri-vision-entire`, `anoushkayadav15/…` | extracted full set | 21.35 GB each | labeled MIT / Unknown / Apache (wrong) | [11] |
| Kaggle `rgislikassi/agriculture-vision-lite`, `nilanjanat/agri-balanced` | third-party subsets | 1.93 GB / 0.35 GB | unknown sampling | [11] |

**None of the third-party mirrors is allowed under Terms §1 and §3** [8]. They are handy evidence of size and structure, but the project shouldn't depend on them. Using one would also be awkward to defend in a graded report.

**Fitting the notebook** (Kaggle: 20 GB saved `/kaggle/working` plus unsaved scratch of unpublished size; Colab: disk unpublished, see [free-gpu-limits](free-gpu-limits.md)):
- Extracted sizes (val derived, 4.20 GB in total): `images/rgb` 1.94 GB · `images/nir` 2.01 GB · `labels/` 0.20 GB · `boundaries/` 0.03 GB · `masks/` 0.03 GB. Train is about 3.1× that (12.96 GB on Kaggle [11]). So **train+val is ≈17.2 GB with NIR and ≈9.0 GB without** (est.). Kaggle's scratch disk size is unpublished (check with `df -h` in the first session). The NIR-included version would nearly fill the 20 GB `/kaggle/working` by itself. BLIP-style models take RGB only, so NIR can be skipped unless the second model uses it.
- Streaming extract, so the 21 GB archive is never stored:
  `curl -sL <S3 URL> | tar -xz -C /kaggle/tmp --wildcards 'Agriculture-Vision-2021/train/*' 'Agriculture-Vision-2021/val/*' --exclude='*/images/nir/*'`
  gzip can't seek, so all 21 GB still go over the network each time. Throughput from Kaggle/Colab to S3 is UNVERIFIED; time it once.
- Better: run the stream once, write a compact derived cache (e.g. RGB resized to the model's 384 px input as JPEG, plus a per-tile JSON of class presence, area fraction and bounding boxes from `label & boundary & mask`). Save it as notebook output (≤20 GB) and attach that output to later runs. Training then never touches the 21 GB tar. The cache size is est. ≈2–4 GB for all 75k train+val tiles at 384 px. Whether a private Kaggle dataset or notebook output of derived tiles counts as "posting" under §3(e) is UNVERIFIED.
- For quick iteration, HTTP range requests can't pull single files out of a .tar.gz, but `tar` can stop early. A subset is just a time limit on the stream, which is biased by archive order (val comes first [4]).

## 6. How the numbers were derived

- **S3 listing:** unsigned `ListObjectsV2` on `intelinair-data-releases` (prefixes and sizes only) [4].
- **Tar structure:** one 256 KB HTTP range read from the start of the official tar.gz. Only tar headers were decompressed and parsed in memory, and nothing was extracted or executed. That showed the folder layout and that `val/masks/` comes first [4].
- **Official metadata files** (14–448 KB, read as data): `data20{17,18,19}_splits.json` and `data_manifest.csv` [13][14]. They gave the flight → field mapping and the leakage counts in §3.
- **Per-class counts and overlap (val):** I paged through the public Kaggle file listing of `nauxqouh/agriculture-vision-2021-val` (names and byte sizes only, 200 per call, no file contents) [11]. A constant 512×512 PNG always has the same byte size. In every label folder the modal size is 1,102 B, for 68–99% of files, so I take that as the empty mask. Any other size counts as a non-empty mask. Boundary/mask files have a separate constant size of 1,358 B, which also shows up for `val/masks/` in the official tar header [4]. Caveat: this is a heuristic on a third-party copy. The flight codes match the official manifest, but recompute the real numbers from the masks in week 1.
- **File-count check:** the HF dataset viewer's row count (1,057,446) for a re-upload of the tar [12] checked against the arithmetic in §1.
- Scripts are kept outside the repo, in the session scratchpad: `kaggle_listing.py`, `analyze_listing.py`.

## Open questions

- Does a private Kaggle dataset, or attached notebook output, holding our derived 384 px tiles and masks violate Terms §3(e) "post the Dataset … websites"? Ask the TA, or email support@intelinair.com [8].
- Showing example tiles in the report, Beamer deck and local demo: the terms allow research use and require citation, but don't address figures. It is probably fine with a citation; UNVERIFIED.
- Mask pixel encoding (0/255 vs 0/1) and whether boundary/mask are 0 outside the valid region: check on the first tile.
- Exact per-class counts and co-occurrence for train (the val figures above are derived); compute from the masks in week 1.
- S3 → Kaggle/Colab streaming throughput (how long a 21 GB pass takes).
- Whether 2017 (15 cm/px) tiles were resampled. This matters for any "size of patch in m²" caption wording.

## Sources

1. Chiu et al., "Agriculture-Vision: A Large Aerial Image Database for Agricultural Pattern Analysis", CVPR 2020, arXiv:2001.01306 (v2). §3.1 Table 2, §3.2–3.5, §4, Figs. 4–7: https://arxiv.org/abs/2001.01306 (HTML render used: https://ar5iv.labs.arxiv.org/html/2001.01306)
2. Brief: [project3.md](../../../project3.md)
3. SHI-Labs/Agriculture-Vision README (master, last push 2024-07-27; no license file): https://github.com/SHI-Labs/Agriculture-Vision
4. S3 bucket listing and HEAD, `intelinair-data-releases/agriculture-vision/` (unsigned): https://intelinair-data-releases.s3.amazonaws.com/?list-type=2&prefix=agriculture-vision/
5. Agriculture-Vision 2021 dataset page (download commands, filename format, raw set): https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021
6. Agriculture-Vision 2020 dataset page (21,061 images, 6 classes, 4.4 GB, "contact us"): https://www.agriculture-vision.com/agriculture-vision-2020/dataset
7. Official HF dataset repo, API and resolve HEAD: https://huggingface.co/datasets/shi-labs/Agriculture-Vision · https://huggingface.co/api/datasets/shi-labs/Agriculture-Vision
8. "Agriculture-Vision Dataset Terms of Use" (IntelinAir), PDF in the official bucket: https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/Agriculture-Vision%20Dataset%20Terms%20of%20Use.pdf
9. 2nd Agriculture-Vision Prize Challenge 2021 page (supervised/semi-supervised tracks, class IDs 0–8, storm_damage not evaluated, overlap-aware mIoU): https://www.agriculture-vision.com/agriculture-vision-2021/prize-challenge-2021 · 1st challenge paper, arXiv:2004.09754 (§2.1 "labels … not mutually exclusive"): https://arxiv.org/abs/2004.09754
10. Wu et al., "Extended Agriculture-Vision", TMLR 2023, arXiv:2303.02460 (§3.1 "pre-processed and scaled", "data manifest", data revisiting): https://arxiv.org/abs/2303.02460
11. Kaggle public API, dataset search and file listings (`/api/v1/datasets/list?search=…`, `/api/v1/datasets/list/<owner>/<slug>`), e.g. https://www.kaggle.com/datasets/nauxqouh/agriculture-vision-2021-val
12. HF Hub search and dataset-viewer API for third-party copies (`/api/datasets?search=agriculture-vision`, `datasets-server.huggingface.co/size|first-rows|rows`), e.g. https://huggingface.co/datasets/Vadim21221/agri-vision-2021-segmentation
13. Official split files: https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_paper_2020/Dataset/data2019_splits.json (also data2017_, data2018_)
14. Official manifest: https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/neurips_benchmarks_2021/data_manifest.csv
15. Sibling note on Kaggle/Colab limits: [free-gpu-limits.md](free-gpu-limits.md)
