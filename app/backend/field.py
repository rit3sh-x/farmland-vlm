"""Field context for a dragged box, from hosted APIs only (issues/17-free-hosted-field-apis.md).

scene   <- Planetary Computer (latest clear Sentinel-2 scene, NDVI PNG, NDVI grid) + NASA CMR (latest clear HLS date)
season  <- NASA VEDA titiler-cmr (HLS NDVI two-week means for the box)
weather <- NASA POWER (daily rain, max temperature, root-zone soil wetness; monthly normals)
Results are cached on disk per box and day; a showcase box falls back to its cache when offline.
"""
import asyncio
import datetime as dt
import hashlib
import io
import json

import httpx
import numpy as np

from . import config

PC = "https://planetarycomputer.microsoft.com/api"
TITILER_CMR = "https://openveda.cloud/api/titiler-cmr"
CMR = "https://cmr.earthdata.nasa.gov/search/granules.json"
POWER = "https://power.larc.nasa.gov/api/temporal"
HLS_S30 = "C2021957295-LPCLOUD"
EXPR = "(B08-B04)/(B08+B04)"
SAT_NDVI_RANGE = (-0.2, 0.9)  # calibrated surface reflectance, unlike the tiles' relative NDVI
PARTS = ("north-west", "north", "north-east", "west", "centre", "east", "south-west", "south", "south-east")
MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def valid_bbox(b) -> bool:
    w, s, e, n = b
    lat = (s + n) / 2
    wm, hm = (e - w) * 111320 * np.cos(np.radians(lat)), (n - s) * 110540
    return 50 <= wm <= 5000 and 50 <= hm <= 5000


def key_of(bbox, day: str) -> str:
    return hashlib.sha1(json.dumps([round(x, 5) for x in bbox] + [day]).encode()).hexdigest()[:16]


def _polygon(b):
    w, s, e, n = b
    return {"type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}}


async def scene(c: httpx.AsyncClient, bbox, key: str) -> dict:
    today = dt.date.today()
    search = {"collections": ["sentinel-2-l2a"], "bbox": bbox, "limit": 1,
              "datetime": f"{today - dt.timedelta(days=60)}/{today}", "query": {"eo:cloud_cover": {"lt": 20}},
              "sortby": [{"field": "datetime", "direction": "desc"}]}
    feats = (await c.post(f"{PC}/stac/v1/search", json=search)).raise_for_status().json()["features"]
    if not feats:
        raise LookupError("no Sentinel-2 scene under 20% cloud in the last 60 days")
    item = feats[0]
    box = ",".join(map(str, bbox))
    q = {"collection": "sentinel-2-l2a", "item": item["id"], "expression": EXPR, "asset_as_band": "true"}
    png = (await c.get(f"{PC}/data/v1/item/bbox/{box}/256x256.png",
                       params={**q, "rescale": f"{SAT_NDVI_RANGE[0]},{SAT_NDVI_RANGE[1]}", "colormap_name": "rdylgn"})).raise_for_status()
    (config.FIELD_CACHE / f"{key}.png").write_bytes(png.content)
    grid = np.load(io.BytesIO((await c.get(f"{PC}/data/v1/item/bbox/{box}/60x60.npy", params=q)).raise_for_status().content))
    ndvi, valid = grid[0], grid[1] > 0
    cells = [ndvi[r * 20:(r + 1) * 20, k * 20:(k + 1) * 20][valid[r * 20:(r + 1) * 20, k * 20:(k + 1) * 20]]
             for r in range(3) for k in range(3)]
    means = [float(x.mean()) if x.size else np.inf for x in cells]
    hls = (await c.get(CMR, params={"collection_concept_id": HLS_S30, "bounding_box": box, "sort_key": "-start_date",
                                    "cloud_cover": "0,20", "page_size": 1})).raise_for_status().json()["feed"]["entry"]
    return {"type": "scene", "s2_png": f"/api/field/s2/{key}.png", "bounds": bbox, "date": item["properties"]["datetime"][:10],
            "cloud_pct": round(item["properties"]["eo:cloud_cover"], 1), "ndvi_mean": round(float(ndvi[valid].mean()), 3),
            "lowest_part": PARTS[int(np.argmin(means))], "hls_date": hls[0]["time_start"][:10] if hls else None}


async def season(c: httpx.AsyncClient, bbox) -> dict:
    today = dt.date.today()
    params = [("collection_concept_id", HLS_S30), ("temporal", f"{today - dt.timedelta(days=183)}T00:00:00Z/{today}T23:59:59Z"),
              ("step", "P2W"), ("temporal_mode", "interval"), ("assets_regex", "B[0-9][0-9A]"),
              ("assets", "B04"), ("assets", "B08"),
              ("expression", EXPR), ("asset_as_band", "true"), ("cloud_cover", "0,20")]
    r = (await c.post(f"{TITILER_CMR}/rasterio/timeseries/statistics", params=params, json=_polygon(bbox))).raise_for_status()
    stats = r.json()["properties"]["statistics"]
    series = []
    for interval, bands in sorted(stats.items()):
        st = next(iter(bands.values()))
        if st.get("mean") is not None and np.isfinite(st["mean"]):
            series.append({"date": interval.split("/")[0][:10], "ndvi": round(float(st["mean"]), 3)})
    if not series:
        raise LookupError("no clear HLS scenes in the last six months")
    peak = max(series, key=lambda p: p["ndvi"])
    return {"type": "season", "series": series, "peak": peak}


async def weather(c: httpx.AsyncClient, bbox) -> dict:
    lon, lat = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    end = dt.date.today() - dt.timedelta(days=3)
    start = end - dt.timedelta(days=183)
    common = {"community": "AG", "longitude": round(lon, 4), "latitude": round(lat, 4), "format": "JSON"}
    daily, clim = await asyncio.gather(
        c.get(f"{POWER}/daily/point", params={**common, "parameters": "PRECTOTCORR,T2M_MAX,GWETROOT",
                                                "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d")}),
        c.get(f"{POWER}/climatology/point", params={**common, "parameters": "PRECTOTCORR"}))
    p = daily.raise_for_status().json()["properties"]["parameter"]
    normals = clim.raise_for_status().json()["properties"]["parameter"]["PRECTOTCORR"]
    days = []
    for d in sorted(p["PRECTOTCORR"]):
        rain, tmax, wet = p["PRECTOTCORR"][d], p["T2M_MAX"][d], p["GWETROOT"][d]
        if rain > -900:
            days.append({"date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "rain_mm": rain, "tmax_c": tmax if tmax > -900 else None,
                         "soil_wetness": wet if wet > -900 else None})
    expected = sum(normals[MONTHS[int(x["date"][5:7]) - 1]] for x in days)
    rain = sum(x["rain_mm"] for x in days)
    return {"type": "weather", "daily": days, "normals": {m: normals[m] for m in MONTHS},
            "rain_mm": round(rain), "rain_vs_normal": round(rain / expected - 1, 3) if expected else None}


def summary(sc: dict | None, se: dict | None, we: dict | None) -> str:
    out = []
    if sc and se:
        drop = se["peak"]["ndvi"] - sc["ndvi_mean"]
        peak_day = dt.date.fromisoformat(se["peak"]["date"]).strftime("%-d %b")
        out.append(f"Field NDVI {sc['ndvi_mean']:.2f} on {dt.date.fromisoformat(sc['date']).strftime('%-d %b')}, "
                   f"{drop:.2f} {'below' if drop >= 0 else 'above'} its {peak_day} peak; the {sc['lowest_part']} is lowest.")
    elif sc:
        out.append(f"Field NDVI {sc['ndvi_mean']:.2f} on {sc['date']}; the {sc['lowest_part']} is lowest.")
    if we and we["rain_vs_normal"] is not None:
        pct = round(abs(we["rain_vs_normal"]) * 100)
        out.append(f"Rain over the season is {pct}% {'below' if we['rain_vs_normal'] < 0 else 'above'} NASA POWER's long-term normal.")
    out.append("Satellite data shows where, not why.")
    return " ".join(out)


async def field_events(bbox):
    """NDJSON events for /api/field: scene, season, weather, summary, unavailable, done."""
    config.FIELD_CACHE.mkdir(parents=True, exist_ok=True)
    key = key_of(bbox, dt.date.today().isoformat())
    cached = config.FIELD_CACHE / f"{key}.json"
    showcase = [s for s in json.loads(config.SHOWCASE.read_text()) if [round(x, 4) for x in s["bbox"]] == [round(x, 4) for x in bbox]]
    if not cached.exists() and showcase:
        older = sorted(config.FIELD_CACHE.glob(f"showcase-{key_of(bbox, '')}-*.json"))
        cached = older[-1] if older else cached
    if cached.exists():
        for ev in json.loads(cached.read_text()):
            yield ev
        yield {"type": "done", "cached": True}
        return
    got = {}
    async with httpx.AsyncClient(timeout=config.EXTERNAL_TIMEOUT, headers={"User-Agent": config.USER_AGENT}) as c:
        tasks = {"scene": asyncio.create_task(scene(c, bbox, key)), "season": asyncio.create_task(season(c, bbox)),
                 "weather": asyncio.create_task(weather(c, bbox))}
        for name, task in tasks.items():
            try:
                got[name] = await task
                yield got[name]
            except (httpx.HTTPError, LookupError, KeyError, ValueError) as e:
                yield {"type": "unavailable", "part": name, "reason": str(e) or type(e).__name__}
    text = {"type": "summary", "text": summary(got.get("scene"), got.get("season"), got.get("weather"))}
    yield text
    events = list(got.values()) + [text]
    if len(got) == 3:
        cached.write_text(json.dumps(events))
        if showcase:
            (config.FIELD_CACHE / f"showcase-{key_of(bbox, '')}-{dt.date.today()}.json").write_text(json.dumps(events))
    yield {"type": "done"}
