# Role of location imagery in the app

Type: grilling
Status: resolved
Blocked by: 04

## Question

Given what the NASA imagery research found, how does location imagery fit the app: a core input (pick a spot on a map → fetch imagery → model answers about it), a context panel next to the model's output (e.g. a colour-mapped NDVI field map, or NDVI over time), or dropped? What does the user see and do after picking a location?

Also: should the app show an NDVI ("deep green") view computed from the tile's own NIR band? That needs no external API.

Another option: Agriculture-Vision's 2021 raw set has 261 full-field flights at 10 cm (4 GeoTIFF bands, 299 GB in all). A few of them could serve as in-domain "pick a spot in this field" demo imagery. Whether they carry real coordinates is unverified.

If NAIP aerial imagery goes into the model:
- Do those answers carry an "outside training data" label, or go only through the hedging path?
- Is 2–3-year-old NAIP acceptable next to a current satellite view?

And either way: rendered PNGs only, or raw bands too (the backend then needs a GeoTIFF reader)? Cache imagery for a few showcase locations so the demo works offline? None of the free services publish uptime or rate limits.

User input, 2026-10-19 (raised while working on [QA-pair and caption generation scheme](05-qa-pair-scheme.md)): show RGB and NDVI side by side, even when the model only uses RGB. So the tile-NDVI view is wanted. It uses the same fixed colour map that the RGB + NDVI comparison row's model sees.

## Answer

Locked with the user on 2026-10-19. The user's own lookup confirmed the API is NASA GIBS, matching [NASA imagery sources and fit with Agriculture-Vision](04-nasa-imagery-sources.md). Whether the TA expects the imagery to go *into* the model is still unconfirmed.

**Role: context panel only.** The model answers only about dataset tiles and uploads, never about satellite pixels.

**Layers**
1. **NASA GIBS**, loaded by the browser directly as WMTS tiles:
   - `MODIS_Terra_L3_NDVI_16Day`: regional, 2000 onward
   - `MODIS_Terra_NDVI_8Day`: the most recent view, from 2025-02 only
   - HLS true colour (`HLS_S30/L30_Nadir_BRDF_Adjusted_Reflectance`): 30 m

   Checked live on 2026-10-19: MODIS NDVI is served only up to zoom 9 (≈230 m/px at 42°N), about 3×3 px per field. HLS goes to zoom 12 (≈28 m/px).
2. **Sentinel-2 field NDVI**
   - Source: Element 84 Earth Search (`https://earth-search.aws.element84.com/v1`, collection `sentinel-2-l2a`, assets `red`/`nir`, 10 m), no key.
   - The backend picks the latest scene where the field box is ≤20% cloudy and reads only that window from the public COGs with `rasterio`.
   - It is coloured with the same map as the tile NDVI view.
   - Checked live: 11 scenes with <20% cloud over Ames, IA between 2026-08-01 and 10-18; COGs support range reads.
3. **NDVI over the season:** the field's mean NDVI scene by scene, over the last 6 months.
   - Plan A: a titiler-cmr HLS time-series call. Box statistics are UNVERIFIED; check in build week 1.
   - Fallback: field means from the Sentinel-2 scenes, cached.

**Summary text:** one or two rule-based sentences built from those numbers, e.g. "Field NDVI 0.58 on 17 Oct, 0.21 below its 4 Aug peak; the south-west third is lowest. Satellite data shows where, not why." No model involved.

**Field area:** a pin with a resizable square box, ~800 m by default.

**Pairing with the model**
- Each analysis has an image (dataset tile or upload) and an optional pin.
- An upload can be pinned where it was taken, and the panel then shows that field's context next to the model's answers.
- Dataset tiles carry no coordinates, so for them the panel is a separate explorer.

**Offline**
- ~5 showcase fields are pre-cached, all layers in a few MB: 3–4 Midwest fields and one near IIT Ropar.
- Offline, a showcase field shows its cached layer images and chart in place of the live map.
- Other spots are fetched live. On failure the panel says "imagery unavailable" and the rest of the app keeps working.

**Tile NDVI view:** yes. RGB and the tile's NDVI map sit side by side, as the user decided while working on the [QA-pair and caption generation scheme](05-qa-pair-scheme.md).

**Dropped from the user's pasted sources**

| Source | Why dropped |
|---|---|
| NAIP | out as both model input and base layer |
| USGS M2M | needs an account and approval, adds nothing |
| HLS COG downloads via LP DAAC STAC | needs Earthdata login; titiler-cmr renders the same data keyless |
| USDA's own NAIP ArcGIS server | reset the connection from this laptop |

**Stretch, first to cut:** a whole-field explorer from one Agriculture-Vision 2021 raw flight at 10 cm. Process it on Colab into a downsampled RGB + NIR mosaic of ~100–300 MB on the laptop; a click cuts a 512×512 window for the model. No masks are available, and the raw flights' coordinates are unverified.

**Left to [UI prototype](09-ui-prototype.md):** the base map's look, place search, and the panel layout.

**Amendment, 2026-10-19:** the user replaced the pin + resizable box with a drag-to-select rectangle on a MapLibre map (OpenFreeMap `liberty` style). Recorded in [UI prototype](09-ui-prototype.md).

**Amendment, 2026-10-19 (from [Free hosted APIs for field-context data](17-free-hosted-field-apis.md)).** The user's rule: hosted APIs for all external data, only inference on our side.
- The Sentinel-2 cut-out and latest-clear-scene lookup now come from the **Planetary Computer data API**: STAC search, an NDVI render with a fixed expression and colour map, box statistics. This replaces Earth Search + rasterio window reads.
- The season series comes from **titiler-cmr** `/rasterio/timeseries/statistics`. Box statistics are now verified, so this is plan A with no fallback needed.
- **New:** a season weather strip (rain, max temperature) and a root-zone soil-wetness line on the season chart, from **NASA POWER**. The summary gains a clause such as "rain since May X% below POWER's long-term normal".
