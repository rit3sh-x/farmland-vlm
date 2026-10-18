# NASA imagery sources and fit with Agriculture-Vision

Research for ticket [04-nasa-imagery-sources](../issues/04-nasa-imagery-sources.md). As of 2026-10-18. Bracketed numbers like [3] point to Sources at the end. **UNVERIFIED** marks anything not confirmed in a primary source. **est.** marks my own estimates. **live** marks a request I ran on 2026-10-18 (see Live checks).

## Summary

- **Which API the TA most likely meant:** NASA **GIBS** (Global Imagery Browse Services), seen through **Worldview**. The old api.nasa.gov "Earth" imagery endpoint (pan-sharpened Landsat 8 served through Google Earth Engine) has been archived. api.nasa.gov now carries a banner saying it "has been archived and replaced with Earthdata GIBS API" [1][2][3]. In my live checks the old endpoint timed out with no bytes returned. GIBS serves imagery that is already rendered (corrected reflectance, colour maps), which fits "image data with all the effects/filters applied" [5][10]. Worldview's HLS layers add true colour, false colour, NDVI, EVI, NDMI and other index views of 30 m Harmonized Landsat Sentinel-2 (HLS) data. The index views are rendered on the fly by NASA VEDA's `titiler-cmr` [7][8]. This is still my inference; ask the TA to confirm.
- **Best NASA source for the app:** GIBS WMTS tiles or Worldview Snapshots for HLS true colour, and `titiler-cmr` for HLS NDVI by bbox or point. All three need no key, return a PNG or JPEG for a lat/lon box, and worked live [6][8][9]. They are good for a **context panel** (recent field view, NDVI, NDVI over time).
- **Best source if a picked location must go into the model:** **USDA NAIP** aerial imagery (not NASA). It is 0.3 m (2023 flights for IA, IL, OH, MN, KS) or 0.6 m elsewhere, with RGB+NIR, the same four bands as Agriculture-Vision [21][22]. Planetary Computer renders any bbox to PNG with no key (live) [22].
- **Verdict:** NASA satellite imagery **cannot feed a model trained on Agriculture-Vision tiles.** One tile covers 51–102 m (512 px at 10–20 cm) [27]. At HLS's 30 m that is only 2×2 to 3×3 pixels. In the live sample those pixels were near-uniform green and carried no anomaly texture. Upsampling to 512 px yields blocks that are far out of distribution. Even Sentinel-2's 10 m gives only 5–10 pixels per side. NAIP is the only thing here that bridges the gap. It is still a domain shift (different sensor, a 2–3-year revisit, a single date), and row-level anomalies (planter skip, double plant) are borderline at 0.3 m and invisible at 0.6 m.

## Comparison table

Scale reference: an Agriculture-Vision tile is 512×512 px at 10–20 cm per pixel, so it covers 51–102 m (0.26–1.05 ha), with RGB+NIR [27][28]. In the "px per tile" notes, "tile" means that footprint.

| Source | Ground resolution | Bands (RGB / NIR / NDVI) | US Midwest coverage | Dates / revisit | Auth / rate limits | lat/lon or bbox request | Cost | Rust access | Python access |
|---|---|---|---|---|---|---|---|---|---|
| **api.nasa.gov Earth imagery** (`/planetary/earth/imagery`) | Landsat 8 pan-sharpened [3]; 15 m is UNVERIFIED | RGB PNG only [3] | Was global [3] | Nearest date to the one requested [3] | api.nasa.gov key | Yes: `lat`, `lon`, `dim` in degrees, `date` [3] | Free | n/a | n/a |
| ↳ status | **Archived**: banner says it was replaced by GIBS [1]; removed from the API catalog [2]; outages reported since 2022 [4]; **live: 60 s and 90 s timeouts, 0 bytes** | | | | | | | | |
| **GIBS WMTS / WMS: HLS true colour** (`HLS_S30_…`, `HLS_L30_Nadir_BRDF_Adjusted_Reflectance`) | Deepest zoom 0.000275°/px, about 30.6 m N–S and 22.7 m E–W at 42°N (**live**, GetCapabilities) [6]. Native HLS is 30 m [12] | True colour only (pre-rendered PNG) [6][7] | Global, Midwest included (**live** tile over Ames, IA) [6] | Daily layer per overpass: S30 from 2015-11-28, L30 from 2013 [6]. **No compositing**: dates without an overpass come back transparent and cloudy dates show clouds (**live**) | None. No key needed (**live**). Rate limits not documented: UNVERIFIED | WMTS REST `…/{Layer}/default/{Time}/31.25m/{z}/{row}/{col}.png`; WMS `GetMap` takes a bbox [5][6] | Free [10] | `reqwest` + `image` (plain PNG/JPEG); no geo crate needed | `requests` or `owslib` [38] |
| **Worldview Snapshots** (`wvs.earthdata.nasa.gov/api/v1/snapshot`) | Same as the GIBS layer it renders | Any GIBS layer, composited [9] | Global | Per layer | None (**live**) | **Yes: one GET with bbox, size, layers and date returns JPEG/PNG** [9] (**live**: HLS S30, 512×256) | Free | `reqwest` | `requests` |
| **Worldview HLS index layers** (NDVI, EVI, SAVI, NDMI, NDWI, false colour; marked *BETA*), via VEDA `titiler-cmr` | 30 m HLS, rendered at the requested size [7][8] | **NIR false colour, NDVI, EVI, NDMI and more**, as colour-mapped PNG or raw values; any band math expression [7][8] | Global | Any HLS overpass; CMR filters such as `temporal` and `cloud_cover` [8] | None (**live**). No documented SLA or limits: UNVERIFIED | **Yes:** `/rasterio/bbox/{minx},{miny},{maxx},{maxy}/{w}x{h}.png` and `/rasterio/point/{lon},{lat}`, plus a timeseries endpoint [8] (**live**: NDVI PNG in 1.7 s; point NDVI 0.887) | Free | `reqwest` | `requests` |
| **GIBS coarse NDVI** (MODIS, VIIRS) | 250 m (MODIS 8- and 16-day), 500 m (VIIRS), 1 km (monthly) [6] | NDVI colour map | Global | 8-day, 16-day or monthly composites [6] | None | Same as GIBS | Free | as GIBS | as GIBS |
| **HLS raw data** (HLSS30, HLSL30, HLS-VI v2.0; LP DAAC) | 30 m [11][12][13] | S30: 13 bands incl. NIR B08/B8A; L30: 11 bands incl. NIR B05 [12][13]; **HLS-VI: NDVI, EVI etc. as 30 m COGs** (since Feb 2025) [11][14] | Global; over Ames, IA: 47 S30 + 30 L30 granules from 2026-06-01 to 10-11 (**live**) [15] | One look every ~1.6 days with both satellites combined [11]; **live**: one per 1.7 days, before cloud filtering | CMR search open (**live**); downloads need Earthdata Login (EDL), and an unauthenticated GET gets 302 to URS (**live**) [15]. EDL user tokens last 60 days [16] | CMR `point=` or `bounding_box=`, then per-band COG over HTTPS [15] | Free | CMR JSON via `reqwest` + `serde`; `Authorization: Bearer <EDL token>`; COG via `async-tiff` / `tiff` or `gdal` crate [39] | `earthaccess` [18] + `rasterio` [38] |
| ↳ HLS mirror on Planetary Computer (`hls2-s30`) | 30 m [24] | Same bands [24] | Global | From 2020-01-01 [24] (**live**: latest item 2026-10-10) | No EDL; anonymous SAS token [23] | STAC search + COG, or the PC data API | Free | as NAIP via PC | `pystac-client` + `planetary-computer` [38] |
| **AppEEARS** | Product-native: HLS 30 m in AppEEARS (**live** product list) [19] | Any layer of 192 products, incl. HLSS30/HLSL30 and their `_VI` NDVI products [19] | Global | Product-native | EDL → Bearer token, valid about 48 h [19]. HTTP 429 exists; numeric limits not documented: UNVERIFIED | bbox/polygon "area" or lat/lon "point" tasks [19] | Free | REST via `reqwest` | `requests` |
| ↳ latency | **Asynchronous**: submit task → poll `/status` → download `/bundle` [19]. Typically minutes ("usually ~10 minutes" per a forum user), and days during backlogs [20]. **Not usable for a live UI request** | | | | | | | | |
| **NAIP via Planetary Computer** (USDA) | **0.3 m** (2023: IA, IL, OH, MN, KS), 0.6 m (2018–2022 for most states), 1 m (older) (**live** STAC) [21][22] | **R, G, B, NIR** in one 4-band COG [21]; NDVI must be computed | CONUS. Latest per state (**live**): IA/IL/OH/MN/KS 2023 at 0.3 m; ND 2023 at 0.6 m; IN/WI/MO/NE/SD/MI 2022 at 0.6 m [22] | One date per state about every 2–3 years ("at least once every three years") [21]. PC holds 2010–2023 only [21] | STAC search open. Blob reads need a free anonymous SAS token (**live**). A subscription key gives "less restricted rate limiting" [23] | **Yes:** STAC `intersects`, then the data API `/item/bbox/{minx},{miny},{maxx},{maxy}/{w}x{h}.png`; `asset_bidx=image\|4,1,2` gives NIR false colour (**live**) [22] | Free. STAC metadata says licence `proprietary` [21]; GEE says FSA data are public domain [26] | `reqwest` for the PNG; or `stac` / `stac-client` + `async-tiff` [39] | `pystac-client` + `planetary-computer` + `rasterio` / `odc-stac` [38] |
| **NAIP via AWS Open Data** | 0.3–1 m [25] | `naip-analytic`: 4-band MRF/COG; `naip-visualization`: RGB only [25] | CONUS | Same as above; years beyond 2023 not checked: UNVERIFIED | AWS account; **all three buckets are Requester Pays** in us-west-2 [25] | No bbox API; read the file tiles yourself | **Requester pays egress** (amount not checked: UNVERIFIED) | `object_store` / AWS SDK + COG reader (not checked: UNVERIFIED) | `boto3` + `rasterio` |
| **NAIP via Earth Engine** (`USDA/NAIP/DOQQ`) | Catalog lists 0.6 m [26] even though 0.3 m flights exist [22] | R, G, B, N [26] | CONUS | 2002-06-15 to 2023-11-17 [26] | Register a Cloud project. Free for research, education and nonprofit use [26] | Yes (EE geometry; `getThumbURL` / export: UNVERIFIED) | Free (non-commercial) [26] | REST API only; no official Rust client (UNVERIFIED) | `earthengine-api` [38] |
| **Sentinel Hub Process API on Copernicus Data Space (CDSE)** (ESA, not NASA) | **Sentinel-2 L2A: 10 m** for B02/B03/B04/B08; 20 m for red edge and B8A [33] | Anything an **evalscript** computes: true colour, false colour, NDVI PNG [30] | Global | Sentinel-2 revisit about 5 days [11] | OAuth2 client credentials from `identity.dataspace.copernicus.eu` [32]; endpoint live (unauthenticated POST gets 401). **Free tier: 10,000 requests/month, 300/min, 10,000 PU/month, 300 PU/min** [29]. 1 PU = 512×512 px × 3 bands [31] | Yes: POST with a bbox or GeoJSON and a time range [30] | Free tier [29] | `reqwest` (OAuth + JSON POST) | `sentinelhub` [38] |
| **GHISA** (GHISACONUS v001) | n/a: not imagery | A **CSV spectral library** from EO-1 Hyperion, per crop and growth stage [34] | CONUS agro-ecological zones [34] | 2008–2015, one granule, "COMPLETE" [34] | LP DAAC / CMR | No location imagery | Free | n/a | n/a |
| **Farmonaut API** (commercial) | UNVERIFIED | UNVERIFIED | UNVERIFIED | UNVERIFIED | Account; pricing page blocked our fetch (403) [36] | UNVERIFIED | Paid. Its ToS mentions a "limited free tier"; a third-party review says there is no permanent free plan [36] | — | — |

Pixels per Agriculture-Vision tile footprint (51–102 m), **est.** from the resolutions above:

| Resolution | Pixels per side | What a tile becomes |
|---|---|---|
| 10–20 cm (Agriculture-Vision) | 512 | Rows, gaps, weed patches visible |
| NAIP 0.3 m | 170–340 | Close: rows marginal, patches clear (**live** crop showed wheel tracks and within-field variation) |
| NAIP 0.6 m | 85–170 | Zones and patches only |
| Sentinel-2 10 m | 5–10 | A few blobs |
| HLS 30 m | 2–3 | **live**: 4×3 px nearly identical greens. No pattern at all |
| MODIS 250 m+ | under 1 | The tile is part of one pixel |

## Anomalies visible at each resolution

Agriculture-Vision 2021 annotates: double plant, drydown, endrow, nutrient deficiency, planter skip, water, waterway, weed cluster [28]. The physical sizes below are my estimates unless cited. Midwest corn is mostly in about 30-inch (0.76 m) rows [37], so row-level anomalies need well under 0.76 m per pixel.

| Anomaly (typical scale, **est.**) | Agriculture-Vision 10–20 cm | NAIP 0.3 m | NAIP 0.6 m | Sentinel-2 10 m | HLS 30 m | MODIS/VIIRS 250 m–1 km |
|---|---|---|---|---|---|---|
| Planter skip (missing row segment, under 1 m wide) | Yes | Marginal: about 2.5 px per row | No | No | No | No |
| Double plant (overlapping rows) | Yes | Marginal to no | No | No | No | No |
| Weed cluster (patches, metres to tens of metres) | Yes | Yes | Mostly | Only very large patches | No | No |
| Endrow (headland strip, about 10–20 m wide) | Yes | Yes | Yes | Marginal | No | No |
| Waterway (grassed channel, about 10–30 m wide) | Yes | Yes | Yes | Marginal (wide ones) | Marginal | No |
| Standing water | Yes | Yes | Yes | Ponds over ~20–30 m | Ponds over ~1 ha. OPERA DSWx-HLS gives a 30 m water mask in GIBS [6] | No |
| Nutrient deficiency (zones, tens to hundreds of m) | Yes (colour and texture) | Yes | Yes | Only as a low-NDVI zone, not diagnosable | Only as a low-NDVI zone of several pixels | Regional stress only |
| Drydown (field or zone senescence) | Yes | Yes | Yes | Yes, best as a time series | Yes, field-level time series | Regional only |

Satellite NDVI can flag "this part of the field is lower" but cannot say why: nutrient, drought, disease or water. This is a natural hedging case for the app (see the brief's hedging requirement).

## Extra leads checked (from a pasted AI chat)

1. **GHISA.** It is a hyperspectral **spectral library** (CSV reflectance spectra) of five major crops by growth stage and agro-ecological zone. It was built from EO-1 Hyperion scenes from 2008–2015, using the USDA Cropland Data Layer as reference, and is one granule on LP DAAC [34]. It holds **no images of a location**, so it **cannot feed a captioning or VQA model**. The claim that it is queryable "through the NASA Open Data Portal API" did not check out: a live data.nasa.gov CKAN search for "GHISA" returned 0 results [35]. It is discoverable through LP DAAC / CMR (concept ID C2763264663-LPCLOUD) [34].
2. **earthaccess.** It is a real NASA DAAC + community Python library that searches CMR and then downloads or "streams" files through `fsspec` [18]. **Direct S3 access works only inside AWS us-west-2.** The library checks the EC2 instance metadata for `us-west-2`, and anywhere else it uses HTTPS links [18]. LP DAAC's S3 credentials are "same-region, read-only" and valid for 1 hour [17]. From the laptop it therefore streams over HTTPS (range reads) with an EDL login [18]. The **Rust equivalent** is CMR search (`granules.umm_json`, no auth, **live**) [15] plus HTTPS GET with `Authorization: Bearer <EDL user token>` (60-day tokens, at most 2 per user) [16], then a COG reader (`async-tiff` / `tiff` / `gdal` crate) [39]. I could not test bearer-token downloads from LP DAAC (no account): UNVERIFIED.
3. **AppEEARS.** Point and area (bbox/polygon) extraction is a REST API, but it is **asynchronous**: submit a task, poll `/status`, then fetch `/bundle` files [19]. Turnaround is usually minutes, but LP DAAC staff have attributed multi-day queues to backlogs [20]. It can only serve **precomputed** data, such as demo fields fetched ahead of time, not a live click in the UI.
4. **Sentinel Hub Process API (CDSE).** It is free through a Copernicus Data Space account, with evalscripts returning colour-mapped NDVI or true-colour PNGs for a bbox at 10 m [29][30][33]. Free-tier quotas are 10,000 requests/month, 300 requests/min, 10,000 PU/month and 300 PU/min [29]. One 512×512 3-band request costs about 1 PU, and the minimum is 0.005 PU [31]. Auth is an OAuth2 client-credentials token [32]. "Effects/filters applied" also fits this API, but it is ESA/Copernicus, not NASA. It is the best single API for "styled image of a bbox", and it is 3× sharper than HLS. It still cannot feed the model (5–10 px per tile). Correction to the paste: Sentinel-2 and Landsat are **multispectral** (13 and 11 bands) [12][13][33], not hyperspectral.
5. **Farmonaut.** It is a commercial crop-health platform. Its pricing page returned 403 to our fetch. Its Terms of Service mention a "limited free tier" for subscriptions, while a third-party review says there is no permanent free plan and lists "Contact us" pricing [36]. I treat it as paid or unclear for a student and did not evaluate it further. The paste's resolution and API claims about it are UNVERIFIED.

## Live checks (2026-10-18)

Downloads are in `/tmp/claude-1000/-home-rit3sh-project-ai512/7b5c81d6-ca8f-42f6-bc78-a5cbbf7b1cb8/scratchpad/nasa-check/` (untrusted data; never executed).

- `api.nasa.gov/planetary/earth/imagery` and `/assets` with DEMO_KEY: 60 s and 90 s timeouts, 0 bytes. The api.nasa.gov header banner text confirms archival [1].
- GIBS WMTS GetCapabilities (EPSG:4326 and 3857): HTTP 200, 1330 layers. HLS layers use the `31.25m` matrix set, deepest level 11 [6].
- GIBS tile `HLS_S30…/2026-08-11/31.25m/11/341/613.png` over Ames, IA: clear, 100 % opaque. The same tile on non-overpass dates was fully transparent, and L30 on 2026-08-04 was clouds.
- Worldview Snapshots, HLS S30, bbox 41.99,-93.72,42.01,-93.68: 512×256 JPEG in 2.3 s.
- VEDA `titiler-cmr` NDVI bbox PNG: 1.7 s. Point NDVI at (-93.70, 42.00) on 2026-08-11 was 0.887.
- CMR granule search for HLSS30/HLSL30 at that point (June to October 2026): 47 + 30 granules, 14 + 9 under 20 % cloud. An unauthenticated asset GET got 302 to `urs.earthdata.nasa.gov`.
- Planetary Computer NAIP STAC at that point: 2023 at 0.3 m back to 2011 at 1 m. The SAS token endpoint returned 200. A data API bbox crop of a 77 m footprint returned 512×512 PNG in RGB and CIR in about 1.3 s.
- CDSE Process API, unauthenticated POST: 401 (endpoint live).

## Open questions (for the map)

- **Confirm with the TA** that they meant GIBS / Worldview, and whether they expect the location imagery to go *into* the model or to sit beside it. This feeds ticket 08.
- **Role split for ticket 08:** NASA HLS (true colour plus NDVI and an NDVI time series via `titiler-cmr`) as a context panel, and NAIP 0.3/0.6 m as the only candidate model input. Should the app label NAIP answers as "out of training domain", or run them only through the hedging path?
- **Domain-shift evaluation:** is it worth a small robustness row testing the model on NAIP chips resampled to tile scale? The Agriculture-Vision 2021 *raw* release has 10 cm full-field flights for 54 fields, and some fields come with 10 m Sentinel imagery [28]. That pairing could quantify the satellite gap honestly.
- **Freshness vs resolution:** NAIP's latest Midwest flight is 2022–2023 on Planetary Computer. Is a 2–3-year-old sub-metre image acceptable in the demo next to a current 30 m HLS view? Whether 2024–2025 NAIP exists anywhere free with a bbox API is UNVERIFIED.
- **Rust-backend cost:** PNG endpoints (GIBS, Snapshots, `titiler-cmr`, PC data API) need only `reqwest`. Raw 4-band NAIP or HLS COGs need a COG reader (`async-tiff` or GDAL bindings). Decide whether the app ever needs raw bands, for example to compute NDVI from NAIP NIR itself.
- **Offline demo:** the laptop demo should cache fetched imagery for a few showcase locations, since none of the free services publish SLAs or rate limits (UNVERIFIED).

## Sources

1. api.nasa.gov header banner (live 2026-10-18): https://api.nasa.gov/assets/html/header.html. Banner commits: https://github.com/nasa/api-docs/commit/cce30085e6 (2025-06-25, "will be archived") and https://github.com/nasa/api-docs/commit/b5b9556081 (2026-09-10, "has been archived and replaced with Earthdata GIBS API").
2. api.nasa.gov API catalog, current: https://api.nasa.gov/assets/json/apis.json (16 APIs, no "Earth", GIBS present).
3. api.nasa.gov Earth API docs as of 2025-06-25: https://raw.githubusercontent.com/nasa/api-docs/cce30085e6/assets/json/apis.json
4. nasa/api-docs issues on Earth API failures: https://github.com/nasa/api-docs/issues/169, https://github.com/nasa/api-docs/issues/223, https://github.com/nasa/api-docs/issues/237
5. GIBS Access Basics (WMTS, WMS, TWMS endpoints and templates): https://nasa-gibs.github.io/gibs-api-docs/access-basics/
6. GIBS WMTS GetCapabilities (live): https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities&VERSION=1.0.0 and https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/wmts.cgi?SERVICE=WMTS&REQUEST=GetCapabilities&VERSION=1.0.0
7. Worldview layer configuration (HLS NDVI, EVI and false-colour titiler layers): https://worldview.earthdata.nasa.gov/config/wv.json
8. VEDA titiler-cmr OpenAPI (live): https://openveda.cloud/api/titiler-cmr/api. Docs: https://developmentseed.org/titiler-cmr/
9. Worldview Snapshots API (Earthdata blog, 2019): https://www.earthdata.nasa.gov/news/blog/replacement-worldviews-image-download-service
10. Earthdata GIBS API page ("free, open"): https://www.earthdata.nasa.gov/engage/open-data-services-software/earthdata-developer-portal/gibs-api. GIBS intro: https://nasa-gibs.github.io/gibs-api-docs/
11. HLS project page (1.6-day revisit, Sentinel-2 5-day, HLS-VI Feb 2025): https://www.earthdata.nasa.gov/data/projects/hls
12. HLSS30 v2.0 (30 m, 13 bands, from 2015-11-28): https://lpdaac.usgs.gov/products/hlss30v002/
13. HLSL30 v2.0 (30 m, 11 bands): https://lpdaac.usgs.gov/products/hlsl30v002/
14. HLS Sentinel-2 Vegetation Indices v2.0 (NDVI, EVI; 30 m): https://lpdaac.usgs.gov/products/hlss30_viv002/
15. CMR granule search (live, no auth): https://cmr.earthdata.nasa.gov/search/granules.umm_json?short_name=HLSS30&version=2.0&point=-93.7,42.0&temporal=2026-06-01T00:00:00Z,2026-10-11T23:59:59Z&page_size=200. API docs: https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html
16. Earthdata Login user tokens (60-day validity, Bearer): https://urs.earthdata.nasa.gov/documentation/for_users/user_token
17. LP DAAC in-region S3 credentials README (same-region, 1 hour): https://data.lpdaac.earthdatacloud.nasa.gov/s3credentialsREADME
18. earthaccess docs: https://earthaccess.readthedocs.io/en/latest/. Source showing the us-west-2 check and HTTPS fallback: https://github.com/earthaccess-dev/earthaccess/blob/main/earthaccess/store.py and https://github.com/earthaccess-dev/earthaccess/blob/main/earthaccess/api.py
19. AppEEARS API (tasks, status, bundle, 48 h token, area/point parameters, geotiff/netcdf4): https://appeears.earthdatacloud.nasa.gov/api/. Product list (live): https://appeears.earthdatacloud.nasa.gov/api/product
20. Earthdata Forum on AppEEARS turnaround: https://forum.earthdata.nasa.gov/viewtopic.php?p=25062 and https://forum.earthdata.nasa.gov/viewtopic.php?p=25161
21. Planetary Computer NAIP collection (live): https://planetarycomputer.microsoft.com/api/stac/v1/collections/naip. Dataset page: https://planetarycomputer.microsoft.com/dataset/naip
22. Planetary Computer STAC search and data API (live): https://planetarycomputer.microsoft.com/api/stac/v1/search?collections=naip and https://planetarycomputer.microsoft.com/api/data/v1/item/bbox/-93.7200,42.0000,-93.71907,42.0007/512x512.png?collection=naip&item=ia_m_4209359_sw_15_030_20230902_20240104&assets=image&asset_bidx=image%7C1,2,3
23. Planetary Computer SAS token endpoint (live): https://planetarycomputer.microsoft.com/api/sas/v1/token/naip. SDK README (subscription key is optional and gives less restricted rate limiting): https://github.com/microsoft/planetary-computer-sdk-for-python
24. Planetary Computer HLS S30 collection (live): https://planetarycomputer.microsoft.com/api/stac/v1/collections/hls2-s30
25. NAIP on AWS registry: https://registry.opendata.aws/naip/. Registry YAML (RequesterPays: True for all three buckets): https://github.com/awslabs/open-data-registry/blob/main/datasets/naip.yaml
26. Earth Engine NAIP catalog: https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ
27. Agriculture-Vision paper (94,986 images, 3,432 farmlands, RGB+NIR, up to 10 cm/px): https://arxiv.org/abs/2001.01306
28. Agriculture-Vision 2021 dataset page (anomaly classes; raw 10 cm flights 2017–2020 with some 10 m Sentinel imagery): https://www.agriculture-vision.com/agriculture-vision-2021/dataset-2021
29. CDSE quotas (Sentinel Hub: 10,000 requests/month, 300/min, 10,000 PU/month, 300 PU/min): https://documentation.dataspace.copernicus.eu/Quotas.html
30. CDSE Sentinel Hub Process API: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html. Overview: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub.html
31. CDSE Processing Unit definition: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/ProcessingUnit.html
32. CDSE Sentinel Hub authentication (OAuth2 client credentials): https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview/Authentication.html
33. CDSE Sentinel-2 L2A bands and resolutions: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html
34. GHISACONUS v001 (LP DAAC): https://lpdaac.usgs.gov/products/ghisaconusv001/
35. data.nasa.gov CKAN search for GHISA (live, 0 results): https://data.nasa.gov/api/3/action/package_search?q=GHISA
36. Farmonaut: pricing page (403 to our fetch) https://farmonaut.com/api-pricing; Terms of Service https://farmonaut.com/?p=19; Toolradar pricing review https://toolradar.com/tools/farmonaut/pricing
37. UNL Extension G2216, corn row spacing (Nebraska USDA survey: ~30-inch rows most common); read via a search excerpt, not in full: https://extensionpubs.unl.edu/publication/g2216/2013/html/view
38. PyPI (live versions): https://pypi.org/project/earthaccess/, https://pypi.org/project/pystac-client/, https://pypi.org/project/planetary-computer/, https://pypi.org/project/rasterio/, https://pypi.org/project/OWSLib/, https://pypi.org/project/sentinelhub/, https://pypi.org/project/earthengine-api/, https://pypi.org/project/odc-stac/
39. crates.io (live versions): https://crates.io/crates/stac, https://crates.io/crates/stac-client, https://crates.io/crates/async-tiff, https://crates.io/crates/tiff, https://crates.io/crates/gdal, https://crates.io/crates/reqwest, https://crates.io/crates/image
