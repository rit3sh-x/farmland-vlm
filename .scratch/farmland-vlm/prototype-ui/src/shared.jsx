// PROTOTYPE — wipe me. Small pieces the three variants share (not layout).
import { useEffect, useRef, useState, useCallback } from 'react'
import maplibregl from 'maplibre-gl'
import 'maplibre-gl/dist/maplibre-gl.css'
import { CLASSES, MODELS, caption, findings, answer, ndviColor, fakeS2Ndvi, boxSize } from './fake.js'

// ---------- fake model runs ----------

const seen = new Set() // tile+model pairs already "encoded" (prompt cache)

export function useCaption(tile, model) {
  const [s, set] = useState({ phase: 'idle' })
  useEffect(() => {
    if (!tile) return
    const key = tile.id + model, cold = !seen.has(key)
    const m = MODELS.find((x) => x.id === model)
    const total = cold ? m.cold : 700, t0 = performance.now(), full = caption(tile, model)
    let iv
    const tick = () => {
      const el = performance.now() - t0
      const enc = total * 0.7
      if (el < enc) set({ phase: 'encoding', elapsed: el, cold })
      else if (el < total) set({ phase: 'writing', elapsed: el, cold, text: full.slice(0, Math.ceil(full.length * (el - enc) / (total - enc))) })
      else { seen.add(key); set({ phase: 'done', elapsed: total, cold, text: full, findings: findings(tile, model) }); clearInterval(iv) }
    }
    iv = setInterval(tick, 50)
    return () => clearInterval(iv)
  }, [tile?.id, model])
  return s
}

export function useQA(tile, model) {
  const [thread, setThread] = useState([])
  useEffect(() => setThread([]), [tile?.id])
  const ask = useCallback((q, m = model) => {
    const id = Math.random()
    setThread((t) => [...t, { id, q, model: m, pending: true }])
    setTimeout(() => setThread((t) => t.map((x) => (x.id === id ? { ...x, pending: false, a: answer(q, tile, m) } : x))), 650 + Math.random() * 500)
  }, [tile, model])
  return { thread, ask }
}

// ---------- small display pieces ----------

export function Waiting({ s, compact }) {
  const secs = ((s.elapsed || 0) / 1000).toFixed(1)
  return (
    <div className={'waiting' + (compact ? ' compact' : '')}>
      <span className="pulse" />
      <span>{s.phase === 'encoding' ? (s.cold ? 'Reading the image…' : 'Using cached image…') : 'Writing caption…'}</span>
      <span className="secs">{secs}s</span>
      {!compact && s.cold && <span className="hint">first look takes a few seconds on the laptop; follow-ups are fast</span>}
    </div>
  )
}

export function Conf({ v }) {
  return <span className="conf" title={`model confidence ${(v * 100).toFixed(0)}%`}><span style={{ width: `${v * 100}%` }} /></span>
}

export function Answer({ a }) {
  if (!a) return null
  return (
    <span className={'answer' + (a.hedged ? ' hedged' : '')}>
      {a.hedged && <span className="hedge-tag">not sure</span>}
      <b>{a.text}</b>
      <Conf v={a.conf} />
      {a.why && <span className="why">{a.why}</span>}
    </span>
  )
}

const CELLS = ['top left', 'top', 'top right', 'left', 'centre', 'right', 'bottom left', 'bottom', 'bottom right']

export function MiniGrid({ loc, color }) {
  const on = loc === 'throughout' ? CELLS : loc.split(' or ')
  return (
    <span className="minigrid">
      {CELLS.map((c) => <i key={c} style={on.includes(c) ? { background: color } : null} />)}
    </span>
  )
}

export function Chip({ a }) {
  const c = CLASSES[a.cls]
  const ext = { small: 1, moderate: 2, large: 3 }[a.extent]
  return (
    <span className={'chip' + (a.hedge ? ' hedged' : '')} title={a.hedge ? `borderline: ${a.hedge}` : ''}>
      <span className="dot" style={{ background: c.color }} />
      <span className="chip-name">{a.hedge === 'possibly' ? 'possibly ' : ''}{c.name}</span>
      <span className="ext">{[1, 2, 3].map((i) => <i key={i} className={i <= ext ? 'on' : ''} style={i <= ext ? { background: c.color } : null} />)}</span>
      <span className="chip-ext">{a.hedge && a.hedge.includes(' to ') ? a.hedge : a.extent}</span>
      <MiniGrid loc={a.loc} color={c.color} />
      <Conf v={a.conf} />
    </span>
  )
}

// ---------- tile view: RGB / NDVI / side by side, with a 3×3 findings overlay ----------

function useFakeNdvi(src) {
  const [url, setUrl] = useState(null)
  useEffect(() => {
    if (!src) return
    const img = new Image(); img.crossOrigin = 'anonymous'; img.src = src
    img.onload = () => {
      const c = document.createElement('canvas'); c.width = c.height = 256
      const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0, 256, 256)
      const d = ctx.getImageData(0, 0, 256, 256)
      for (let i = 0; i < d.data.length; i += 4) {
        const r = d.data[i], g = d.data[i + 1]
        const [R, G, B] = ndviColor(0.5 + 2.2 * (g - r) / (g + r + 1)) // PROTOTYPE: fake NDVI from RGB (no NIR copied)
        d.data[i] = R; d.data[i + 1] = G; d.data[i + 2] = B
      }
      ctx.putImageData(d, 0, 0); setUrl(c.toDataURL())
    }
  }, [src])
  return url
}

function Overlay({ items, dashed }) {
  return (
    <div className="overlay">
      {CELLS.map((c) => {
        const hit = items.find((a) => a.loc === 'throughout' || a.loc.split(' or ').includes(c))
        return <i key={c} style={hit ? { background: CLASSES[hit.cls].color + '44', outline: `2px ${hit.hedge || dashed ? 'dashed' : 'solid'} ${CLASSES[hit.cls].color}` } : null} />
      })}
    </div>
  )
}

export function TileView({ tile, mode = 'rgb', items = [], truth = false, round }) {
  const ndvi = useFakeNdvi(tile?.nir ? tile.src : null)
  if (!tile) return <div className="tile empty">Pick a tile or upload a photo</div>
  const one = (src, label) => (
    <div className="tile-img">
      <img src={src} alt={label} />
      <Overlay items={truth ? tile.truth || [] : items} dashed={truth} />
      <span className="tile-label">{label}</span>
    </div>
  )
  const nd = ndvi || tile.src
  return (
    <div className={'tile ' + mode + (round ? ' round' : '')}>
      {mode !== 'ndvi' && one(tile.src, 'RGB')}
      {mode !== 'rgb' && (tile.nir ? one(nd, 'NDVI') : <div className="tile-img no-nir">NDVI needs a NIR band — uploads don’t have one</div>)}
    </div>
  )
}

export function NdviLegend() {
  return <span className="legend"><span>stressed / bare</span><span className="ramp" /><span>healthy</span></span>
}

// ---------- map: OpenFreeMap liberty + GIBS rasters + drag-to-select field box ----------

const GIBS = (layer, date, tms, ext = 'png') =>
  `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/${layer}/default/${date}/${tms}/{z}/{y}/{x}.${ext}`
const EMPTY = { type: 'FeatureCollection', features: [] }
const boxGeo = (b) => ({ type: 'Feature', geometry: { type: 'Polygon', coordinates: [[[b.w, b.s], [b.e, b.s], [b.e, b.n], [b.w, b.n], [b.w, b.s]]] } })
const corners = (b) => [[b.w, b.n], [b.e, b.n], [b.e, b.s], [b.w, b.s]]

export function FieldMap({ box, onBox, layers = {}, center = [-93.7, 42.0], zoom = 13, className = '', toolbar = true }) {
  const ref = useRef(), mapRef = useRef()
  const [ready, setReady] = useState(false)
  const [tool, setTool] = useState(false)

  useEffect(() => {
    const map = new maplibregl.Map({ container: ref.current, style: 'https://tiles.openfreemap.org/styles/liberty', center, zoom, attributionControl: { compact: true } })
    mapRef.current = map
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    map.on('load', () => {
      const before = map.getStyle().layers.find((l) => l.type === 'symbol')?.id
      map.addSource('modis', { type: 'raster', tiles: [GIBS('MODIS_Terra_L3_NDVI_16Day', '2026-09-30', 'GoogleMapsCompatible_Level9')], tileSize: 256, maxzoom: 9, attribution: 'NASA GIBS' })
      map.addLayer({ id: 'modis', type: 'raster', source: 'modis', layout: { visibility: 'none' }, paint: { 'raster-opacity': 0.75 } }, before)
      map.addSource('hls', { type: 'raster', tiles: [GIBS('HLS_S30_Nadir_BRDF_Adjusted_Reflectance', '2026-10-11', 'GoogleMapsCompatible_Level12')], tileSize: 256, maxzoom: 12, attribution: 'NASA GIBS HLS' })
      map.addLayer({ id: 'hls', type: 'raster', source: 'hls', layout: { visibility: 'none' } }, before)
      map.addSource('box', { type: 'geojson', data: EMPTY })
      map.addLayer({ id: 'box-fill', type: 'fill', source: 'box', paint: { 'fill-color': '#1a9850', 'fill-opacity': 0.08 } })
      map.addLayer({ id: 'box-line', type: 'line', source: 'box', paint: { 'line-color': '#0b5d34', 'line-width': 2.5, 'line-dasharray': [2, 1] } })
      setReady(true)
    })
    return () => map.remove()
  }, [])

  // layer toggles + fake Sentinel-2 cut-out
  useEffect(() => {
    const map = mapRef.current
    if (!ready) return
    map.setLayoutProperty('modis', 'visibility', layers.modis ? 'visible' : 'none')
    map.setLayoutProperty('hls', 'visibility', layers.hls ? 'visible' : 'none')
    if (map.getLayer('s2')) { map.removeLayer('s2'); map.removeSource('s2') }
    if (layers.s2 && box) {
      map.addSource('s2', { type: 'image', url: fakeS2Ndvi(box), coordinates: corners(box) })
      map.addLayer({ id: 's2', type: 'raster', source: 's2', paint: { 'raster-opacity': 0.85, 'raster-resampling': 'nearest' } }, 'box-fill')
    }
  }, [ready, layers.modis, layers.hls, layers.s2, box])

  // show the box, fly to it when it changes from outside
  useEffect(() => {
    const map = mapRef.current
    if (!ready) return
    map.getSource('box').setData(box ? boxGeo(box) : EMPTY)
    if (box && box.fly) map.fitBounds([[box.w, box.s], [box.e, box.n]], { padding: 80, duration: 900 })
  }, [ready, box])

  // drag-to-select tool
  useEffect(() => {
    const map = mapRef.current
    if (!ready || !tool) return
    map.dragPan.disable(); map.getCanvas().style.cursor = 'crosshair'
    let start = null
    const mk = (a, b) => ({ w: Math.min(a.lng, b.lng), e: Math.max(a.lng, b.lng), s: Math.min(a.lat, b.lat), n: Math.max(a.lat, b.lat) })
    const down = (e) => { start = e.lngLat }
    const move = (e) => { if (start) map.getSource('box').setData(boxGeo(mk(start, e.lngLat))) }
    const up = (e) => { if (!start) return; const b = mk(start, e.lngLat); start = null; setTool(false); if (b.e - b.w > 1e-5) onBox?.(b) }
    map.on('mousedown', down); map.on('mousemove', move); map.on('mouseup', up)
    return () => { map.off('mousedown', down); map.off('mousemove', move); map.off('mouseup', up); map.dragPan.enable(); map.getCanvas().style.cursor = '' }
  }, [ready, tool])

  const size = box && boxSize(box)
  return (
    <div className={'fieldmap ' + className}>
      <div ref={ref} className="fieldmap-canvas" />
      {toolbar && (
        <div className="map-tools">
          <button className={'tool' + (tool ? ' active' : '')} onClick={() => setTool(!tool)} title="Drag on the map to select the field">
            ⬚ {tool ? 'Drag to draw…' : box ? 'Redraw area' : 'Select area'}
          </button>
          {size && <span className="box-size">{Math.round(size.wm)} × {Math.round(size.hm)} m · {size.ha.toFixed(0)} ha</span>}
        </div>
      )}
    </div>
  )
}

export function SeasonChart({ ctx, w = 300, h = 110 }) {
  const px = (p) => [16 + p.x * (w - 28), 10 + (1 - p.v) * (h - 32)]
  const pts = ctx.series.map(px)
  const [pkx, pky] = px(ctx.peak)
  return (
    <svg className="season" viewBox={`0 0 ${w} ${h}`} width="100%">
      {[0.25, 0.5, 0.75].map((v) => <line key={v} x1="16" x2={w - 12} y1={10 + (1 - v) * (h - 32)} y2={10 + (1 - v) * (h - 32)} className="grid" />)}
      <polyline points={pts.map((p) => p.join(',')).join(' ')} className="line" />
      {pts.map(([x, y], i) => { const [R, G, B] = ndviColor(ctx.series[i].v); return <circle key={i} cx={x} cy={y} r="3" fill={`rgb(${R},${G},${B})`} /> })}
      <text x={pkx} y={pky - 6} textAnchor="middle" className="peak">peak</text>
      {ctx.months.map((m, i) => <text key={m} x={16 + (i / (ctx.months.length - 1)) * (w - 28)} y={h - 6} textAnchor="middle" className="month">{m}</text>)}
    </svg>
  )
}

// ---------- the floating prototype switcher (not part of any design) ----------

export function Switcher({ variants, current, names }) {
  const i = variants.indexOf(current)
  const go = (d) => {
    const v = variants[(i + d + variants.length) % variants.length]
    const u = new URL(location.href); u.searchParams.set('variant', v); location.href = u.toString()
  }
  useEffect(() => {
    const k = (e) => {
      if (e.target.closest('input, textarea, [contenteditable]')) return
      if (e.key === 'ArrowLeft') go(-1)
      if (e.key === 'ArrowRight') go(1)
    }
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  })
  if (import.meta.env.PROD) return null
  return (
    <div className="switcher">
      <button onClick={() => go(-1)}>←</button>
      <span>{current} — {names[current]}</span>
      <button onClick={() => go(1)}>→</button>
    </div>
  )
}
