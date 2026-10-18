import { useEffect, useRef, useState } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { CELLS, CLASSES, ndviColor } from './scheme.js'

const cellsOf = (loc) => (!loc ? [] : loc === 'throughout' ? CELLS : loc.split(' or ').filter((c) => c !== 'throughout'))
const striped = (c) => `repeating-linear-gradient(45deg, ${c} 0 3px, ${c}55 3px 6px)`

export function Conf({ v }) {
  if (v == null) return null
  return <span className="conf" title={`model confidence ${(v * 100).toFixed(0)}%`}><span style={{ width: `${v * 100}%` }} /></span>
}

export function Grid({ items, size = 18 }) {
  const style = (c) => {
    const f = items.find((x) => cellsOf(x.loc).includes(c))
    if (!f) return null
    const col = CLASSES[f.cls].color
    return { background: f.hedge ? striped(col) : col }
  }
  return <span className="grid9" style={{ width: size, height: size }}>{CELLS.map((c) => <i key={c} style={style(c)} />)}</span>
}

export function Bars({ extent, color }) {
  const n = { small: 1, moderate: 2, large: 3 }[(extent || '').split(' to ').pop()] || 0
  return <span className="bars">{[1, 2, 3].map((i) => <i key={i} style={i <= n ? { background: color } : null} />)}</span>
}

export function FindingRow({ f }) {
  const c = CLASSES[f.cls]
  return (
    <div className={'frow' + (f.hedge ? ' hedged' : '')}>
      <Grid items={[f]} size={30} />
      <span className="frow-name">{f.possibly && <em>possibly </em>}{c.name}</span>
      <span className="frow-meta">{[f.extent, f.loc].filter(Boolean).join(' · ') || 'no extent or location given'}</span>
      <Bars extent={f.extent} color={c.color} />
      <Conf v={f.conf} />
    </div>
  )
}

export function TileView({ image, mode, findings = [], truth }) {
  if (!image) return <div className="tile empty">Pick a tile or upload a photo</div>
  const items = truth || findings
  const overlay = (
    <div className="overlay">
      {CELLS.map((c) => {
        const f = items.find((x) => cellsOf(x.loc).includes(c))
        if (!f) return <i key={c} />
        const col = CLASSES[f.cls].color
        return <i key={c} style={{ background: col + '44', outline: `2px ${f.hedge || truth ? 'dashed' : 'solid'} ${col}` }} />
      })}
    </div>
  )
  const one = (src, label) => (
    <div className="tile-img"><img src={src} alt={label} />{overlay}<span className="tile-label">{label}</span></div>
  )
  return (
    <div className={'tile ' + mode}>
      {mode !== 'ndvi' && one(image.rgb, 'RGB')}
      {mode !== 'rgb' && (image.ndvi ? one(image.ndvi, 'NDVI') : <div className="tile-img no-nir">NDVI needs a NIR band. Uploads don’t have one.</div>)}
    </div>
  )
}

export function NdviLegend() {
  return <span className="legend"><span>stressed / bare</span><span className="ramp" /><span>healthy</span></span>
}

export function SeasonChart({ season, weather, w = 380, h = 150 }) {
  if (!season) return null
  const pts = season.series
  const t0 = Date.parse(pts[0].date), t1 = Date.parse(pts[pts.length - 1].date) || t0 + 1
  const X = (d) => 24 + ((Date.parse(d) - t0) / (t1 - t0 || 1)) * (w - 36)
  const Y = (v) => 10 + (1 - v) * (h - 46)
  const wet = (weather?.daily || []).filter((d) => d.soil_wetness != null && Date.parse(d.date) >= t0)
  const rain = (weather?.daily || []).filter((d) => Date.parse(d.date) >= t0)
  const maxRain = Math.max(10, ...rain.map((d) => d.rain_mm))
  const months = []
  for (let d = new Date(t0); d <= new Date(t1); d.setMonth(d.getMonth() + 1, 1)) months.push(new Date(d))
  return (
    <svg className="season" viewBox={`0 0 ${w} ${h}`} width="100%">
      {[0.25, 0.5, 0.75].map((v) => <line key={v} x1="24" x2={w - 12} y1={Y(v)} y2={Y(v)} className="grid" />)}
      {rain.map((d) => <rect key={d.date} x={X(d.date)} y={h - 36 - (d.rain_mm / maxRain) * 18} width="1.6" height={(d.rain_mm / maxRain) * 18} className="rain" />)}
      {wet.length > 1 && <polyline className="wet" points={wet.map((d) => `${X(d.date)},${Y(d.soil_wetness)}`).join(' ')} />}
      <polyline className="line" points={pts.map((p) => `${X(p.date)},${Y(p.ndvi)}`).join(' ')} />
      {pts.map((p) => { const [r, g, b] = ndviColor(p.ndvi); return <circle key={p.date} cx={X(p.date)} cy={Y(p.ndvi)} r="3.2" fill={`rgb(${r},${g},${b})`} /> })}
      <text x={X(season.peak.date)} y={Y(season.peak.ndvi) - 7} textAnchor="middle" className="peak">peak {season.peak.ndvi.toFixed(2)}</text>
      {months.map((m) => <text key={+m} x={X(m.toISOString())} y={h - 8} textAnchor="middle" className="month">{m.toLocaleString('en', { month: 'short' })}</text>)}
      <text x={w - 12} y={14} textAnchor="end" className="month">NDVI · {wet.length > 1 ? 'soil wetness (dashed) · ' : ''}rain bars</text>
    </svg>
  )
}

const GIBS = (layer, time, tms) => `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${layer}/default/${time}/${tms}/{z}/{y}/{x}.png`
const EMPTY = { type: 'FeatureCollection', features: [] }
const boxGeo = (b) => ({ type: 'Feature', geometry: { type: 'Polygon', coordinates: [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]] } })

export function boxSize(b) {
  const lat = (b[1] + b[3]) / 2
  const wm = (b[2] - b[0]) * 111320 * Math.cos((lat * Math.PI) / 180), hm = (b[3] - b[1]) * 110540
  return { wm, hm, ha: (wm * hm) / 1e4 }
}

export function FieldMap({ box, onBox, layers, scene, flyTo, className = '' }) {
  const ref = useRef(), mapRef = useRef()
  const [ready, setReady] = useState(false)
  const [tool, setTool] = useState(false)

  useEffect(() => {
    const map = new maplibregl.Map({ container: ref.current, style: 'https://tiles.openfreemap.org/styles/liberty', center: [-92, 40], zoom: 4, attributionControl: { compact: true } })
    mapRef.current = map
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    map.on('load', () => {
      const before = map.getStyle().layers.find((l) => l.type === 'symbol')?.id
      map.addSource('modis', { type: 'raster', tiles: [GIBS('MODIS_Terra_L3_NDVI_16Day', 'default', 'GoogleMapsCompatible_Level9')], tileSize: 256, maxzoom: 9, attribution: 'NASA GIBS' })
      map.addLayer({ id: 'modis', type: 'raster', source: 'modis', layout: { visibility: 'none' }, paint: { 'raster-opacity': 0.75 } }, before)
      map.addSource('box', { type: 'geojson', data: EMPTY })
      map.addLayer({ id: 'box-fill', type: 'fill', source: 'box', paint: { 'fill-color': '#1a9850', 'fill-opacity': 0.06 } })
      map.addLayer({ id: 'box-line', type: 'line', source: 'box', paint: { 'line-color': '#0b5d34', 'line-width': 2.5, 'line-dasharray': [2, 1] } })
      setReady(true)
    })
    return () => map.remove()
  }, [])

  useEffect(() => {
    const map = mapRef.current
    if (!ready) return
    map.setLayoutProperty('modis', 'visibility', layers.modis ? 'visible' : 'none')
    for (const id of ['hls', 's2']) if (map.getLayer(id)) { map.removeLayer(id); map.removeSource(id) }
    if (layers.hls && scene?.hls_date) {
      map.addSource('hls', { type: 'raster', tiles: [GIBS('HLS_S30_Nadir_BRDF_Adjusted_Reflectance', scene.hls_date, 'GoogleMapsCompatible_Level12')], tileSize: 256, maxzoom: 12, attribution: 'NASA GIBS HLS' })
      map.addLayer({ id: 'hls', type: 'raster', source: 'hls' }, 'box-fill')
    }
    if (layers.s2 && scene?.s2_png) {
      const [w, s, e, n] = scene.bounds
      map.addSource('s2', { type: 'image', url: scene.s2_png, coordinates: [[w, n], [e, n], [e, s], [w, s]] })
      map.addLayer({ id: 's2', type: 'raster', source: 's2', paint: { 'raster-opacity': 0.85 } }, 'box-fill')
    }
  }, [ready, layers.modis, layers.hls, layers.s2, scene])

  useEffect(() => {
    if (!ready) return
    mapRef.current.getSource('box').setData(box ? boxGeo(box) : EMPTY)
  }, [ready, box])

  useEffect(() => {
    if (!ready || !flyTo) return
    if (flyTo.bbox) mapRef.current.fitBounds([[flyTo.bbox[0], flyTo.bbox[1]], [flyTo.bbox[2], flyTo.bbox[3]]], { padding: 90, duration: 900 })
    else mapRef.current.flyTo({ center: [flyTo.lon, flyTo.lat], zoom: 14, duration: 1200 })
  }, [ready, flyTo])

  useEffect(() => {
    const map = mapRef.current
    if (!ready || !tool) return
    map.dragPan.disable(); map.getCanvas().style.cursor = 'crosshair'
    let start = null
    const mk = (a, b) => [Math.min(a.lng, b.lng), Math.min(a.lat, b.lat), Math.max(a.lng, b.lng), Math.max(a.lat, b.lat)]
    const down = (e) => { start = e.lngLat }
    const move = (e) => { if (start) map.getSource('box').setData(boxGeo(mk(start, e.lngLat))) }
    const up = (e) => { if (!start) return; const b = mk(start, e.lngLat); start = null; setTool(false); onBox(b) }
    map.on('mousedown', down); map.on('mousemove', move); map.on('mouseup', up)
    return () => { map.off('mousedown', down); map.off('mousemove', move); map.off('mouseup', up); map.dragPan.enable(); map.getCanvas().style.cursor = '' }
  }, [ready, tool])

  const size = box && boxSize(box)
  const ok = size && size.wm >= 50 && size.hm >= 50 && size.wm <= 5000 && size.hm <= 5000
  return (
    <div className={'fieldmap ' + className}>
      <div ref={ref} className="fieldmap-canvas" />
      <div className="map-tools">
        <button className={'tool' + (tool ? ' active' : '')} onClick={() => setTool(!tool)} title="Drag on the map to select the field">
          ⬚ {tool ? 'Drag to draw…' : box ? 'Redraw area' : 'Select area'}
        </button>
        {size && <span className={'box-size' + (ok ? '' : ' bad')}>{Math.round(size.wm)} × {Math.round(size.hm)} m · {size.ha.toFixed(0)} ha{ok ? '' : ' · keep it 50 m–5 km'}</span>}
      </div>
    </div>
  )
}
