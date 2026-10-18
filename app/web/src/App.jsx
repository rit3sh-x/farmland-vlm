import { useEffect, useState } from 'react'
import { api, geocode } from './api.js'
import { ChatPanel } from './Chat.jsx'
import { FieldMap, NdviLegend, SeasonChart, TileView } from './components.jsx'
import { MODELS } from './scheme.js'

function useField(box) {
  const [s, set] = useState({})
  useEffect(() => {
    if (!box) return
    let live = true
    set({ loading: true })
    ;(async () => {
      for await (const ev of api.field(box)) {
        if (!live) return
        if (ev.type === 'unavailable') set((x) => ({ ...x, unavailable: [...(x.unavailable || []), ev] }))
        else if (ev.type === 'done') set((x) => ({ ...x, loading: false }))
        else if (ev.type === 'error') set((x) => ({ ...x, loading: false, error: ev.message }))
        else set((x) => ({ ...x, [ev.type]: ev }))
      }
    })()
    return () => { live = false }
  }, [box && box.join()])
  return s
}

function FieldPanel({ box, field, layers, setLayers }) {
  if (!box) return <div className="field-empty"><span className="grid9 hint">{Array.from({ length: 9 }, (_, i) => <i key={i} />)}</span><p>Press <b>Select area</b> and drag over a field, or search for a place.</p></div>
  const sc = field.scene, w = field.weather
  return (
    <div className="field-panel">
      {field.summary ? <p className="summary big">{field.summary.text}</p> : <p className="summary big muted">Reading satellite data for this box…</p>}
      <SeasonChart season={field.season?.series ? field.season : null} weather={w} />
      {w && <p className="muted small">Rain since {w.daily[0]?.date}: {w.rain_mm} mm ({w.rain_vs_normal >= 0 ? '+' : ''}{Math.round(w.rain_vs_normal * 100)}% vs NASA POWER normal).</p>}
      <div className="layers">
        {[['s2', 'Sentinel-2 field NDVI', sc ? `10 m · ${sc.date} · ${sc.cloud_pct}% cloud` : 'loading…'],
          ['hls', 'NASA HLS true colour', sc?.hls_date ? `30 m · ${sc.hls_date}` : 'no clear pass found'],
          ['modis', 'NASA MODIS NDVI', 'regional · 16-day']].map(([k, l, sub]) => (
          <label key={k} className="layer-row"><input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} /><span>{l}<small>{sub}</small></span></label>
        ))}
      </div>
      <NdviLegend />
      {field.unavailable?.map((u) => <p key={u.part} className="muted small">{u.part} unavailable: {u.reason}</p>)}
      <p className="muted small">Satellite pixels are 10–250 m, so the model never reads them. It reads aerial tiles at 10 cm.</p>
    </div>
  )
}

function Search({ onPick, showcase }) {
  const [q, setQ] = useState('')
  const [hits, setHits] = useState([])
  const go = async (e) => { e.preventDefault(); if (q.trim()) setHits(await geocode(q.trim()).catch(() => [])) }
  return (
    <div className="D-top">
      <span className="brand sm"><span className="logo">◩</span> Farmland VLM</span>
      <form onSubmit={go} className="search"><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search a place…" /></form>
      <select className="showpick" value="" onChange={(e) => { const s = showcase.find((x) => x.name === e.target.value); s && onPick({ bbox: s.bbox, box: true }) }}>
        <option value="">Showcase fields</option>
        {showcase.map((s) => <option key={s.name}>{s.name}</option>)}
      </select>
      {hits.length > 0 && <div className="hits">{hits.map((h) => <button key={h.name + h.lon} onClick={() => { onPick(h); setHits([]) }}>{h.name}</button>)}</div>}
    </div>
  )
}

export default function App() {
  const [tiles, setTiles] = useState([])
  const [showcase, setShowcase] = useState([])
  const [image, setImage] = useState(null)
  const [model, setModel] = useState('ft')
  const [view, setView] = useState('rgb')
  const [found, setFound] = useState([])
  const [truth, setTruth] = useState(null)
  const [tab, setTab] = useState('chat')
  const [box, setBox] = useState(null)
  const [flyTo, setFlyTo] = useState(null)
  const [layers, setLayers] = useState({ s2: true, hls: false, modis: false })
  const field = useField(box)

  useEffect(() => { api.tiles().then((t) => { setTiles(t); t[0] && pickTile(t[0]) }); api.showcase().then(setShowcase) }, [])
  useEffect(() => { setFound([]); setTruth(null) }, [image?.key, model])

  const pickTile = (t) => {
    setImage({ key: t.id, ref: { tile_id: t.id }, rgb: `/api/tiles/${t.id}/rgb.jpg`, ndvi: `/api/tiles/${t.id}/ndvi.png`, hasNir: true, tile: t })
    setTab('chat')
  }
  const upload = async (e) => {
    const f = e.target.files[0]
    if (!f) return
    const { image_id } = await api.upload(f)
    if (model === 'ndvi') setModel('ft')
    setImage({ key: image_id, ref: { image_id }, rgb: URL.createObjectURL(f), ndvi: null, hasNir: false })
    setTab('chat')
  }
  const toggleTruth = async () => setTruth(truth ? null : await api.truth(image.key))

  return (
    <div className="D">
      <FieldMap box={box} onBox={(b) => { setBox(b); setTab('field') }} layers={layers} scene={field.scene} flyTo={flyTo} className="D-map" />
      <Search showcase={showcase} onPick={(p) => { setFlyTo({ ...p, t: Date.now() }); if (p.box) { setBox(p.bbox); setTab('field') } }} />
      <div className="D-strip">
        {tiles.map((t) => <button key={t.id} title={t.label} className={'thumb sm' + (image?.key === t.id ? ' on' : '')} onClick={() => pickTile(t)}><img src={`/api/tiles/${t.id}/rgb.jpg`} /></button>)}
        <label className="upload sm" title="Upload a photo">+<input type="file" accept="image/*" onChange={upload} /></label>
      </div>
      <aside className="D-panel">
        <div className="D-tabs">
          <button className={tab === 'chat' ? 'on' : ''} onClick={() => setTab('chat')}>Ask the tile</button>
          <button className={tab === 'field' ? 'on' : ''} onClick={() => setTab('field')}>Field context{box && <span className="dotmark" />}</button>
        </div>
        {tab === 'chat' ? (
          <>
            <div className="D-image">
              <TileView image={image} mode={view} findings={found} truth={truth} />
              <div className="D-imgbar">
                <div className="seg sm">{['rgb', 'ndvi', 'split'].map((v) => <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{{ rgb: 'RGB', ndvi: 'NDVI', split: 'Both' }[v]}</button>)}</div>
                {image?.tile && <button className={'sm' + (truth ? ' on' : '')} onClick={toggleTruth}>Ground truth</button>}
                <select className="modelpick" value={model} onChange={(e) => setModel(e.target.value)}>
                  {MODELS.map((m) => <option key={m.id} value={m.id} disabled={m.needsNir && !image?.hasNir}>{m.name}{m.needsNir && !image?.hasNir ? ' (needs NIR)' : ''}</option>)}
                </select>
              </div>
            </div>
            <ChatPanel image={image} model={model} onFindings={setFound} />
          </>
        ) : (
          <FieldPanel box={box} field={field} layers={layers} setLayers={setLayers} />
        )}
      </aside>
    </div>
  )
}
