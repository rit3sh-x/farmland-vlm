# API contract

Type: grilling
Status: resolved
Blocked by: —

## Question

What are the FastAPI endpoints and payloads between the React "Map + chat" UI and the backend?

- **Analysis:** caption + findings for a tile or upload with a chosen model (served, NDVI variant, zero-shot BLIP). Streaming (SSE vs chunked fetch) for the waiting state and streaming caption. The findings shape: class, extent, location, hedge, confidence.
- **Questions:** one answer per call, carrying `hedged`, `why` and `conf`. How a follow-up reuses the tile's prompt cache, and how "Compare with <model>" fans out to another model: queue behind the current request, or run in parallel, given one laptop CPU.
- **Tiles:** the dataset-tile gallery from the test sample, ground-truth findings for the toggle, and NDVI-map rendering for tiles (server-side).
- **Field context for a box:**
  - the Sentinel-2 NDVI cut-out (PNG + bounds)
  - the season series
  - the summary sentence
  - the latest clear HLS date for the GIBS layer
  - the showcase-field cache and the offline fallback

  Also: which calls the browser makes directly (GIBS tiles, OpenFreeMap, the geocoder) and which go through the backend.
- Errors and timeouts: model busy, satellite service down, no clear scene.

Inputs:
- [Served model and backend stack](07-served-model-and-backend.md) (FastAPI + `llama-server` + transformers)
- [Role of location imagery in the app](08-location-imagery-role.md)
- [UI prototype](09-ui-prototype.md), with its throwaway `prototype-ui/` showing every field the UI reads

## Answer

Locked with the user on 2026-10-19.

**General rules**
- **Same origin:** Vite proxies `/api` to FastAPI on `127.0.0.1:8000` in dev; in the demo, FastAPI serves the built UI. No auth, local only.
- **One global queue:** exactly one model call runs at a time. llama-server runs with one slot and BLIP sits behind the same lock. Waiting calls report `queued {ahead}`.
- **Every model call answers as an NDJSON stream** (`StreamingResponse`, one JSON object per line), read in the browser with `fetch` + a line reader.

**Endpoints**

| Call | Returns |
|---|---|
| `GET /api/health` | llama-server up, models loaded, offline flag |
| `GET /api/tiles` | `[{id, label, year, has_nir}]`: ~24 tiles from **fold 0's test sample**, so the served model never shows its own training tiles. Stored locally, never published (dataset terms) |
| `GET /api/tiles/{id}/rgb.jpg` · `/ndvi.png` | NDVI rendered on the server from NIR + red with the one fixed red → yellow → green ramp |
| `GET /api/tiles/{id}/truth` | ground-truth findings, precomputed from the stats file |
| `POST /api/images` (multipart) | `{image_id, has_nir: false}`. Resized to 384 px, kept in an in-memory LRU of 16 |
| `POST /api/read` `{image: {tile_id} \| {image_id}, model: "ft" \| "ndvi" \| "blip"}` | stream: `queued {ahead}` → `phase {encoding}` → `token {text}`… → `done {caption, findings: [{cls, extent, loc, hedge?, conf}], timings: {queue_ms, encode_ms, total_ms}, cached}` |
| `POST /api/ask` `{image, model, question}` | stream: `queued` → `done {answer, family, cls?, hedged, why?, conf, timings}` |
| `POST /api/field` `{bbox: [w, s, e, n]}` | stream: `scene {s2_png, bounds, date, cloud_pct, hls_date}` · `season {series: [{date, ndvi}], peak}` · `summary {text}` · `unavailable {part, reason}` · `done`. Cached per box; showcase boxes come from disk |
| `GET /api/field/s2/{key}.png` | the Sentinel-2 cut-out |
| `GET /api/showcase` | `[{name, note, bbox}]` |

**`/api/ask` is single-turn:** the model sees only the tile and this question, exactly as trained and scored. The UI's suggested questions always name the anomaly, e.g. offering "Where is the drydown?" after a yes.

**How answers are built**
- **Findings** are parsed from the caption by the evaluation's caption fact parser, so rows can never disagree with the caption. Zero-shot BLIP's free-form captions usually parse to none.
- **Confidence** is the geometric mean of the token probabilities over each finding's words, or over the answer's tokens.
- **"why" line:** one fixed template per hedge type:

  | Hedge | Template |
  |---|---|
  | `possibly` | covers only a sliver |
  | `X to Y` | close to the cut-off |
  | `A or B` | two parts hold nearly the same share |

- **Normalising:** answers pass through the evaluation's normaliser.

**Errors**
- Codes: `needs_nir` (the `ndvi` model asked about an upload), `bad_image`, `timeout`, `model_unavailable`.
- External satellite calls time out after 20 s. Offline, a matching showcase box is served from the cache; otherwise the stream sends `unavailable {reason: "offline"}`.
- A bbox must be 50 m – 5 km per side.

**Calls the browser makes directly**
- OpenFreeMap style and tiles.
- GIBS WMTS tiles; the HLS layer is dated by `scene.hls_date`.
- The Photon geocoder, with a `/api/geocode?q=` proxy fallback if its CORS fails.

**Model plumbing**
- llama-server runs with `-np 1` and `cache_prompt: true`. The prompt order is system → image → question, so follow-ups reuse the cached tile prefix.
- BLIP runs in transformers on CPU behind the same lock.

**Checks for build week 1**
- Whether per-request LoRA selection works in llama-server.
- Whether switching between the `ft` and `ndvi` adapters clears the prompt cache.
- Whether Photon allows CORS.

**Amendment, 2026-10-19 (from [Free hosted APIs for field-context data](17-free-hosted-field-apis.md))**
- `/api/field` gains a `weather` event: `{daily: [{date, rain_mm, tmax_c, soil_wetness}], normals: {month: rain_mm_per_day}}`, from NASA POWER. `season` keeps NDVI only; the UI overlays soil wetness from `weather`.
- The backend fills `scene` from Planetary Computer, `season` from titiler-cmr and `weather` from NASA POWER. It fetches and caches the NDVI PNG and serves it at `/api/field/s2/{key}.png`, so showcase boxes work offline.
- Photon's CORS is verified, so the browser calls it directly. The `/api/geocode` fallback is not needed.
