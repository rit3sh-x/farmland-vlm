"""FastAPI app for the laptop demo (issues/14-api-contract.md). Run from the repo root:

    .venv/bin/uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
"""
import hashlib
import io
import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from farmvlm.images import SIZE, colourise, ndvi
from farmvlm.scheme import CLASSES
from farmvlm.text import question_family

from . import config, field
from .engine import MODEL_IDS, Engine

engine = Engine()


@asynccontextmanager
async def lifespan(app):
    if config.LLAMA_SERVER.exists() and config.SERVED_GGUF.exists():
        engine.llama.start()
        await engine.llama.ready()
    yield
    engine.llama.stop()


app = FastAPI(title="Farmland VLM", lifespan=lifespan)
uploads: OrderedDict[str, Image.Image] = OrderedDict()


class ImageRef(BaseModel):
    tile_id: str | None = None
    image_id: str | None = None


class ReadRequest(BaseModel):
    image: ImageRef
    model: str


class AskRequest(ReadRequest):
    question: str


class FieldRequest(BaseModel):
    bbox: tuple[float, float, float, float]


def _tiles() -> list[dict]:
    p = config.DEMO_TILES / "tiles.json"
    return json.loads(p.read_text()) if p.exists() else []


def _tile_paths(tile_id: str) -> tuple[Path, Path]:
    if tile_id not in {t["id"] for t in _tiles()}:
        raise HTTPException(404, f"unknown tile {tile_id}")
    return config.DEMO_TILES / f"{tile_id}.jpg", config.DEMO_TILES / f"{tile_id}_nir.jpg"


def _ndvi_map(rgb_path: Path, nir_path: Path, size: int) -> Image.Image:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    nir = np.asarray(Image.open(nir_path).convert("L"))
    return Image.fromarray(colourise(ndvi(rgb, nir))).resize((size, size), Image.BILINEAR)


def _model_images(ref: ImageRef, model: str) -> list[Image.Image]:
    if model not in MODEL_IDS:
        raise HTTPException(422, {"code": "bad_model", "message": f"model must be one of {MODEL_IDS}"})
    if ref.tile_id:
        rgb_p, nir_p = _tile_paths(ref.tile_id)
        rgb = Image.open(rgb_p).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        return [rgb, _ndvi_map(rgb_p, nir_p, SIZE)] if model == "ndvi" else [rgb]
    if ref.image_id in uploads:
        if model == "ndvi":
            raise HTTPException(422, {"code": "needs_nir", "message": "the NDVI variant needs a NIR band; uploads have none"})
        return [uploads[ref.image_id]]
    raise HTTPException(404, {"code": "bad_image", "message": "unknown image"})


def _ndjson(events):
    async def gen():
        try:
            async for ev in events:
                yield json.dumps(ev) + "\n"
        except Exception as e:  # stream the failure instead of cutting the connection
            yield json.dumps({"type": "error", "code": "model_unavailable", "message": str(e)}) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/health")
async def health():
    up = engine.llama.proc is not None and engine.llama.proc.poll() is None
    return {"llama_server": up, "served": config.SERVED_KEY, "adapters": engine.llama.adapters,
            "blip_loaded": engine.blip.reader is not None}


@app.get("/api/tiles")
async def tiles():
    return [{k: t[k] for k in ("id", "label", "year", "has_nir")} for t in _tiles()]


@app.get("/api/tiles/{tile_id}/rgb.jpg")
async def tile_rgb(tile_id: str):
    return FileResponse(_tile_paths(tile_id)[0], media_type="image/jpeg")


@app.get("/api/tiles/{tile_id}/ndvi.png")
async def tile_ndvi(tile_id: str):
    buf = io.BytesIO()
    _ndvi_map(*_tile_paths(tile_id), 512).save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.get("/api/tiles/{tile_id}/truth")
async def tile_truth(tile_id: str):
    t = next((t for t in _tiles() if t["id"] == tile_id), None)
    if t is None:
        raise HTTPException(404, f"unknown tile {tile_id}")
    return t["truth"]


@app.post("/api/images")
async def upload(file: UploadFile):
    data = await file.read()
    try:
        img = ImageOps.fit(Image.open(io.BytesIO(data)).convert("RGB"), (SIZE, SIZE), Image.BICUBIC)
    except Exception:
        raise HTTPException(422, {"code": "bad_image", "message": "not an image"})
    image_id = hashlib.sha1(data).hexdigest()[:16]
    uploads[image_id] = img
    uploads.move_to_end(image_id)
    while len(uploads) > config.UPLOADS_KEPT:
        uploads.popitem(last=False)
    return {"image_id": image_id, "has_nir": False}


@app.post("/api/read")
async def read(req: ReadRequest):
    return _ndjson(engine.read(_model_images(req.image, req.model), req.model))


@app.post("/api/ask")
async def ask(req: AskRequest):
    images = _model_images(req.image, req.model)
    fam, cls = question_family(req.question)

    async def events():
        async for ev in engine.ask(images, req.model, req.question):
            if ev["type"] == "done":
                ev.update(family=fam, cls=CLASSES[cls].key if cls >= 0 else None)
            yield ev
    return _ndjson(events())


@app.post("/api/field")
async def field_context(req: FieldRequest):
    if not field.valid_bbox(req.bbox):
        raise HTTPException(422, {"code": "bad_bbox", "message": "a box must be 50 m to 5 km per side"})
    return _ndjson(field.field_events(list(req.bbox)))


@app.get("/api/field/s2/{key}.png")
async def field_png(key: str):
    p = config.FIELD_CACHE / f"{key}.png"
    if not key.isalnum() or not p.exists():
        raise HTTPException(404, "no such cut-out")
    return FileResponse(p, media_type="image/png")


@app.get("/api/showcase")
async def showcase():
    return JSONResponse(json.loads(config.SHOWCASE.read_text()))


_web = config.REPO / "app" / "web" / "dist"
if _web.exists():
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
