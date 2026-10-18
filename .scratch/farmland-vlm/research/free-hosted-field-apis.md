# Free hosted APIs for field-context data

Checked live on 2026-10-19 for [Free hosted APIs for field-context data](../issues/17-free-hosted-field-apis.md).
- **Test box:** `[-93.705, 41.996, -93.695, 42.004]`, about 830 × 890 m near Ames, IA. Weather calls used the point (−93.70, 42.00).
- **User agent:** `farmland-vlm-research/0.1`.
- **Times:** one call each from this laptop.

## Results

| API | Used for | Key | Live result | CORS | Coverage |
|---|---|---|---|---|---|
| Planetary Computer STAC (`/api/stac/v1/search`, `sentinel-2-l2a`, `eo:cloud_cover < 20`, sorted by date desc) | latest clear scene | none | 11 items since 1 Aug; newest `S2C_MSIL2A_20261017…T15TVG…`, 0.0% cloud; 1.2 s | — | global |
| Planetary Computer data API `GET /api/data/v1/item/bbox/{w},{s},{e},{n}/{W}x{H}.png?collection=sentinel-2-l2a&item=…&expression=(B08-B04)/(B08+B04)&asset_as_band=true&rescale=-0.2,0.9&colormap_name=rdylgn` | NDVI cut-out PNG for the box | none | 256×256 PNG, 18.7 kB, 0.7 s | `*` | global |
| Planetary Computer data API `POST /api/data/v1/item/statistics?collection=…&item=…&expression=…&asset_as_band=true` (GeoJSON body) | box mean NDVI for one scene | none | mean 0.20, median 0.13, 7,220 px (17 Oct, post-harvest); 0.6 s | — | global |
| NASA VEDA titiler-cmr `POST /rasterio/timeseries/statistics?collection_concept_id=C2021957295-LPCLOUD&temporal=2026-07-01T00:00:00Z/2026-10-15T23:59:59Z&step=P2W&temporal_mode=interval&assets=B04&assets=B08&expression=(B08-B04)/(B08+B04)&asset_as_band=true&cloud_cover=0,20` (GeoJSON body) | **season NDVI series in one call** (HLS S30) | none | 7 two-week means: 0.831, 0.834, **0.864** (29 Jul–11 Aug), 0.846, 0.831, 0.383, 0.344; 7.5 s | — | global |
| NASA POWER `GET /api/temporal/daily/point?parameters=PRECTOTCORR,T2M_MAX,GWETROOT&community=AG&longitude=…&latitude=…&start=…&end=…&format=JSON` | daily rain, max temperature, root-zone soil wetness | none | 168 days to 2026-10-15; 734 mm rain since 1 May; GWETROOT 0.69 on 15 Oct; 1.5 s (Punjab point: 0.4 s) | none (backend call) | global, ~0.5° grid |
| NASA POWER `GET /api/temporal/climatology/point?parameters=PRECTOTCORR&community=AG&…` | monthly normals for "vs normal" | none | mm/day per month, e.g. JUN 4.26, JUL 3.01; 0.9 s | none | global |
| Open-Meteo archive `archive-api.open-meteo.com/v1/archive` | alternative weather source | none (non-commercial terms) | 168 days, 845 mm since 1 May; 1.8 s | none | global, ~9–25 km |
| USDA CropScape `GetCDLValue?year=2024&x=…&y=…` (Albers EPSG:5070) | crop type at a point | none | `Grass/Pasture` at a test point; 2.6 s | — | US only |
| Photon `photon.komoot.io/api/?q=Rupnagar` | place search | none | Rupnagar, Punjab, India; 1.5 s | `*` | global |

Not checked live because they need an account or key: Copernicus Sentinel Hub Statistical API (OAuth client, free-tier quotas), and OpenWeather Agromonitoring (API key).

## Notes

- **Two NDVI sources, two dates.** Sentinel-2 (PC) gave 0.20 on 17 Oct; HLS (titiler-cmr) gave 0.34 for 7–15 Oct. Both fit harvest in progress. The summary sentence should name its source and date.
- **Cloud filter is per scene, not per box.** Sentinel-2's `eo:cloud_cover` describes the whole ~110 km tile, so the box itself can still be cloudy. A second stats call on the `SCL` band (classes 3, 8, 9, 10: shadow and cloud) could reject cloudy boxes. Untested.
- **POWER's grid is coarse** (~50 km), so it is regional weather, not the field's own. Its climatology period is UNVERIFIED; say "POWER's long-term normal".
- **Unknowns:** no published SLA or rate limits for any of these. The showcase cache in [Role of location imagery in the app](../issues/08-location-imagery-role.md) covers demo-day failures.
