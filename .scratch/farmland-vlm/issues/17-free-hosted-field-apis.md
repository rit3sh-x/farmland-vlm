# Free hosted APIs for field-context data

Type: research
Status: resolved
Blocked by: —

## Question

Raised by the user on 2026-10-19 ("for data you can also maybe look at some free hosted apis"). Which free hosted APIs, keyless where possible, could serve the Field context tab's data directly instead of the backend computing it from raw COGs? And which could add useful context?

Field-context data:
- the Sentinel-2 NDVI cut-out for a box
- per-scene mean NDVI for the season series
- latest clear-scene lookup

Possible additions:
- weather or soil moisture over the season, which could explain *why* NDVI fell
- the crop type grown in the box

For each candidate, check live: keyless or account, limits, latency, coverage (US vs global, which includes Punjab), and how it fits the locked `/api/field` stream in the [API contract](14-api-contract.md).

## Answer

All checked services work keyless, except Sentinel Hub (OAuth) and Agromonitoring (key), which were skipped.
- **Planetary Computer data API:** Sentinel-2 scene search, NDVI PNG for a box, and box statistics; 0.6–1.2 s each, CORS `*`.
- **NASA VEDA titiler-cmr:** the whole season's HLS NDVI series for a box in **one** call (7.5 s for 3.5 months). This verifies plan A from the location-imagery decision.
- **NASA POWER:** daily rain, temperature, root-zone soil wetness and monthly normals, global, 0.4–1.5 s.
- **Photon:** sends CORS `*`, so the browser can call it directly.
- **Open-Meteo and USDA CropScape (US only)** also work.

**User decision, 2026-10-19:** "we want hosted api for the data only inference on our party". So all external field data comes from hosted APIs:
- `scene` ← Planetary Computer
- `season` ← titiler-cmr
- weather and soil wetness ← NASA POWER

Our backend only orchestrates, caches and runs the models. There is no rasterio/GDAL and no COG reading. Extras adopted: a season weather strip and a soil-wetness line, both from NASA POWER. Not adopted: crop type.

[findings](../research/free-hosted-field-apis.md)
