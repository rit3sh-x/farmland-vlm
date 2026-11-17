import hashlib
import io
import json
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from farmvlm.images import SIZE, colourise, ndvi
from farmvlm.scheme import CLASSES
from farmvlm.text import question_family

from . import field
from .engine import Engine
from .models import MODELS

REPO = Path(__file__).resolve().parents[2]
TILES = REPO / "data" / "demo_tiles"
WEB = REPO / "app" / "frontend" / "dist"
UPLOADS_KEPT = 16

engine = Engine()


@asynccontextmanager
async def lifespan(app):
    engine.start()
    yield


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
    p = TILES / "tiles.json"
    return json.loads(p.read_text()) if p.exists() else []


def _tile_paths(tile_id: str) -> tuple[Path, Path]:
    if tile_id not in {t["id"] for t in _tiles()}:
        raise HTTPException(404, f"unknown tile {tile_id}")
    return TILES / f"{tile_id}.jpg", TILES / f"{tile_id}_nir.jpg"


def _ndvi_map(rgb_path: Path, nir_path: Path, size: int) -> Image.Image:
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
    nir = np.asarray(Image.open(nir_path).convert("L"))
    return Image.fromarray(colourise(ndvi(rgb, nir))).resize((size, size), Image.BILINEAR)


def _model_images(ref: ImageRef, model_id: str) -> list[Image.Image]:
    model = MODELS.get(model_id)
    if model is None:
        raise HTTPException(422, {"code": "bad_model", "message": f"model must be one of {list(MODELS)}"})
    if ref.tile_id:
        rgb_p, nir_p = _tile_paths(ref.tile_id)
        rgb = Image.open(rgb_p).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        return [rgb, _ndvi_map(rgb_p, nir_p, SIZE)] if model.ndvi else [rgb]
    if ref.image_id in uploads:
        if model.ndvi:
            raise HTTPException(422, {"code": "needs_nir", "message": f"{model.name} needs a NIR band; uploads have none"})
        return [uploads[ref.image_id]]
    raise HTTPException(404, {"code": "bad_image", "message": "unknown image"})


def _ndjson(events):
    async def gen():
        try:
            async for ev in events:
                yield json.dumps(ev) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "code": "model_unavailable", "message": str(e)}) + "\n"
    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/models")
async def models():
    return [{"id": m.id, "name": m.name, "about": m.about, "tuned": m.tuned, "ndvi": m.ndvi, **engine.status(m.id)}
            for m in MODELS.values()]


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
    while len(uploads) > UPLOADS_KEPT:
        uploads.popitem(last=False)
    return {"image_id": image_id, "has_nir": False}


@app.post("/api/read")
async def read(req: ReadRequest):
    return _ndjson(engine.read(_model_images(req.image, req.model), req.model))


@app.post("/api/ask")
async def ask(req: AskRequest):
    images = _model_images(req.image, req.model)
    family, cls = question_family(req.question)

    async def events():
        async for ev in engine.ask(images, req.model, req.question):
            if ev["type"] == "done":
                ev.update(family=family, cls=CLASSES[cls].key if cls >= 0 else None)
            yield ev
    return _ndjson(events())


@app.post("/api/field")
async def field_context(req: FieldRequest):
    if not field.valid_bbox(req.bbox):
        raise HTTPException(422, {"code": "bad_bbox", "message": "a box must be 50 m to 5 km per side"})
    return _ndjson(field.field_events(list(req.bbox)))


@app.get("/api/field/s2/{key}.png")
async def field_png(key: str):
    p = field.CACHE / f"{key}.png"
    if not key.isalnum() or not p.exists():
        raise HTTPException(404, "no such cut-out")
    return FileResponse(p, media_type="image/png")


if WEB.exists():
    app.mount("/", StaticFiles(directory=WEB, html=True), name="web")
