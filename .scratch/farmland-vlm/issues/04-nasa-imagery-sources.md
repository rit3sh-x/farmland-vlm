# NASA imagery sources and fit with Agriculture-Vision

Type: research
Status: resolved
Blocked by: —

## Question

Which NASA imagery API did the TA most likely mean ("image data with all the effects/filters applied"), and can its imagery feed a model trained on Agriculture-Vision tiles?

- Cover NASA GIBS / Worldview (WMTS layers: true colour, false colour, NDVI), Earthdata HLS, the api.nasa.gov Earth imagery endpoint (is it still live?), AppEEARS, and the higher-resolution non-NASA alternative USDA NAIP (via Planetary Computer, AWS or Earth Engine).
- For each: ground resolution, bands (RGB / NIR / NDVI), US Midwest coverage, dates / revisit, auth and rate limits, request by lat/lon or bbox, cost, how to call it from Rust and from Python.
- Compare against Agriculture-Vision's tile scale (10–20 cm per pixel, 512 px) and state which anomalies are visible at each resolution.
- Also check leads from a pasted AI chat: GHISA, earthaccess (direct S3 vs laptop), AppEEARS async turnaround, Sentinel Hub Process API on Copernicus Data Space, and the Farmonaut API.

## Answer

The TA most likely meant NASA GIBS / Worldview. The old api.nasa.gov Earth imagery endpoint is archived and points to GIBS, and Worldview's HLS layers come pre-rendered: true colour, plus false colour and NDVI rendered on the fly by VEDA titiler-cmr.
GIBS tiles, Worldview Snapshots and titiler-cmr all return a PNG or JPEG for a bbox with no key, which makes them good for a context panel (field view, NDVI, NDVI over time).
None of the NASA satellite imagery can feed the Agriculture-Vision model: at 30 m a 51–102 m tile is 2–3 pixels wide, and Sentinel Hub's 10 m is only 5–10.
The only bridge is USDA NAIP (0.3–0.6 m, RGB+NIR) via Planetary Computer's no-key bbox PNG. It still carries a domain shift, data are 2–3 years old, and planter skips and double plants are borderline at 0.3 m.
GHISA is a CSV spectral library, not imagery. AppEEARS is async (minutes to days), so it suits precomputed data only.

[findings](../research/nasa-imagery-sources.md)
