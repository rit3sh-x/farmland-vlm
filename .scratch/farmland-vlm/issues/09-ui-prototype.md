# UI prototype

Type: prototype
Status: resolved
Blocked by: 07, 08

## Question

How should the app look and behave? Build a rough, throwaway Vite prototype of the main flow — image in (upload and/or location pick), caption, Q&A, how hedged answers and detected anomalies are shown — and react to it until it feels intuitive, not plain. Settle the framework (React / Svelte / vanilla) and visual direction.

Inputs from [Served model and backend stack](07-served-model-and-backend.md):
- The backend is FastAPI.
- The demo serves the served model (RGB), its NDVI variant for dataset tiles, and zero-shot BLIP for a before/after view. The UI needs a way to show or switch these.
- Cold latency on the laptop is ~2–12 s per caption depending on the bake-off winner, so the UI needs a good waiting state. Follow-up questions on the same tile are cheap thanks to the prompt cache.

Inputs from [Role of location imagery in the app](08-location-imagery-role.md):
- The map panel is context only: NASA GIBS layers (MODIS NDVI, HLS true colour), a Sentinel-2 10 m field NDVI cut-out, an NDVI-over-the-season chart and a rule-based summary sentence.
- Field = pin + resizable ~800 m box. Each analysis carries an optional pin, so an upload can sit next to its field's context.
- ~5 cached showcase fields have an offline view.
- Also show the tile's RGB and NDVI map side by side.
- Still to decide here: the base map's look and place search.

## Answer

Locked with the user on 2026-10-19 after two rounds. Prototype (primary source, throwaway): [`prototype-ui/`](../prototype-ui/). Run `npm install && npm run dev` there, then open `http://localhost:5173/?variant=A…E`. It holds fake model output and four local Agriculture-Vision tiles, so it is never published.

**How we got here**
- Round 1 had three variants: A Workbench, B Map-first, C Chat-first. The user's verdict: "A is good but is not very clean. B looks good but no chat option, can we do better? C is worst but I like the small square grids."
- Round 2 added D (Map + chat) and E (Focus, a cleaner A). **D won.**

**Direction: Variant D "Map + chat"**
- **Canvas:** a full-screen MapLibre GL JS map with OpenFreeMap's `liberty` style (`https://tiles.openfreemap.org/styles/liberty`).
  - NASA GIBS layers go in as raster sources below the labels. GIBS and OpenFreeMap both send CORS `*` (checked).
  - **Field box:** a drag-to-select tool ("Select area" bottom-left, then drag a rectangle; a size readout like "240 × 238 m · 6 ha"). This replaces the pin + resizable box from [Role of location imagery in the app](08-location-imagery-role.md). Drawing a box switches the panel to Field context.
- **Top-left bar:** brand, a keyless OpenStreetMap place-search box (Photon or Nominatim, fair use), and a "Showcase fields" picker.
- **Under the bar:** a tile strip (dataset tiles from the test sample, plus upload).
- **Right panel**, frosted and full height, with two tabs:
  - **Ask the tile**
    - The tile, with a 3×3 findings overlay where hedged findings are dashed.
    - An image bar: RGB / NDVI / Both, a **Ground truth** toggle for dataset tiles (taken from A), and a model picker (Fine-tuned · Fine-tuned + NDVI · Zero-shot BLIP). The NDVI model is disabled for uploads, which have no NIR band.
    - Below the image, a chat. The model's reading is the first message: a waiting state ("Looking at the tile… 3.2 s", plus a note that the first look is slow and follow-ups are quick), then the streaming caption, then one **finding row** per anomaly.
    - **"Compare with <model>"** under a reading adds that model's reading as another message. This replaces A's "Compare all".
    - Your questions and the answers follow. A horizontally scrolling row of suggested questions and the input sit at the bottom.
  - **Field context**
    - B's summary sentence in large type.
    - The NDVI-over-the-season chart.
    - Layer toggles: Sentinel-2 field NDVI · NASA HLS true colour · NASA MODIS NDVI.
    - The NDVI legend and the "satellite pixels are 10–250 m, so the model never reads them" note.
- **The square-grid motif** (the user asked to keep it):
  - **Finding rows:** each has its own small 3×3 grid in the class colour, with hedged cells striped. Then name, extent and location, extent bars, and a confidence bar.
  - **Location answers:** shown as that grid, e.g. a striped "right or bottom right".
  - **Extent answers:** shown with the bars.
- **Hedged answers:** amber bubble with a left rule, a "not sure" tag and a one-line reason, e.g. "Two parts of the tile hold almost the same share of it."

**Framework and look**
- React + Vite.
- The prototype's look:
  - warm sand background, deep field green, Inter for UI, Fraunces serif for captions
  - amber reserved for hedges
  - one red → yellow → green NDVI ramp everywhere

**Found while building, for the build**
- The HLS true-colour GIBS layer is daily with no compositing, so a hard-coded date is blank for most places. Look up each field's latest clear HLS overpass first.
- MapLibre's CSS sets `.maplibregl-map { position: relative }`, so the map container needs a more specific rule to fill its parent.
- Scroll the chat container itself, not `scrollIntoView`, which also scrolls the page sideways.
- Give flex children `min-width: 0` so that long suggestion chips scroll instead of widening the panel.
