# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary: the instructor, evaluators and classmates at the AI 512 (IIT Ropar) Project 3 presentation. They try the demo for a few minutes and judge the model and the idea. They are not agronomists using it day to day. Taken from the user's answer; no real farmer audience is confirmed.

## Product Purpose

A demo of a vision-language model that captions and answers questions about crop anomalies in aerial farmland tiles. BLIP is fine-tuned with LoRA on question-answer pairs and captions generated from Agriculture-Vision anomaly masks. On held-out fields it averages 74.8% over the four question types, against 29.6% for zero-shot BLIP. The app lets a viewer see that model run, compare it with baselines, and see satellite context for a field they draw.

## Positioning

The model is trained on 10 cm aerial RGB+NIR tiles and answers only about what is visible in them. The app keeps that apart from the satellite Field context, which is measured at 10 m and answers where and how much, never why.

## Operating Context

Runs locally: FastAPI backend plus a React/Vite/MapLibre frontend served on 127.0.0.1:8000, on a laptop CPU (about 8 GB RAM, no GPU). Model replies take 1 to 16 s depending on model. Two independent tools share one screen:
- Ask the tile: pick a sample aerial tile or upload a photo, read it or ask a question, compare four models, optionally overlay ground truth. Not linked to the map. Samples are Agriculture-Vision dataset tiles with no coordinates.
- Field context: search a place or draw a box on the map (50 m to 5 km per side), then get a Sentinel-2 NDVI cut-out, a season NDVI trend (HLS), and rainfall (NASA POWER), plus optional MODIS and HLS map layers.

## Capabilities and Constraints

- Models: BLIP-LoRA (default), zero-shot BLIP, Qwen3.5-2B LoRA, Qwen3.5-2B LoRA + NDVI (demo tiles only, needs NIR).
- Chat streams NDJSON events: queued, phase (loading, encoding, writing), token, done with timings, error.
- Question families: presence, identify, extent, location. Hedged answers and confidence are returned.
- Map: OpenFreeMap Liberty style, NASA GIBS layers, Photon geocoding; all need internet in the browser.
- Backend API is fixed; the redesign is frontend only.
- Stack stays React 18, Vite, TypeScript, MapLibre GL (existing codebase answers this).
- The drawn box must never be presented as the input to the model chat.
- Dataset imagery is non-commercial and must not be redistributed; no dataset images may be added to the repo.

## Evidence on Hand

Report and deck in `report/`; demo tiles built locally into `data/demo_tiles` (not committed). Domain terms in `CONTEXT.md`; design decisions in `.scratch/farmland-vlm/map.md`. No user testimonials or deployment claims exist; none may be invented.

## Product Principles

1. Say what each tool reads. A viewer must never mistake a sample aerial tile for the area on the map.
2. Uncertainty is part of the answer. Hedges, confidence and "satellite shows where, not why" stay visible.
3. The model comparison is the point. Switching models and seeing differences must be quick and obvious.
4. Honest about the machine. Show loading, queueing and CPU latency instead of hiding them.
5. Keep what works: it is a working demo, and the redesign fixes problems rather than starting from nothing.

## Accessibility & Inclusion

No product-specific requirement established. Treat as WCAG AA for contrast, keyboard use and reduced motion.
