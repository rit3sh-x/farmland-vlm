// PROTOTYPE — Variant D "Map + chat": B's full-bleed map, with one glass panel on the right that holds
// the image and a real chat (tab 1) or the field context (tab 2).
import { useState, useEffect } from 'react'
import { TILES, SHOWCASE, boxAround, fieldContext } from './fake.js'
import { TileView, NdviLegend, FieldMap, SeasonChart } from './shared.jsx'
import { ChatPanel, ModelPick } from './chat.jsx'

export function FieldPanel({ box, layers, setLayers }) {
  const ctx = box && fieldContext(box)
  if (!ctx) return <div className="field-empty"><Grid4 /><p>Press <b>Select area</b> and drag over a field, or search for a place.</p></div>
  return (
    <div className="field-panel">
      <p className="summary big">{ctx.summary}</p>
      <SeasonChart ctx={ctx} />
      <div className="layers">
        {[['s2', 'Sentinel-2 field NDVI', '10 m · 17 Oct · ' + ctx.cloud + '% cloud'], ['hls', 'NASA HLS true colour', '30 m · latest clear pass'], ['modis', 'NASA MODIS NDVI', 'regional · 16-day']].map(([k, l, sub]) => (
          <label key={k} className="layer-row"><input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} /><span>{l}<small>{sub}</small></span></label>
        ))}
      </div>
      <NdviLegend />
      <p className="muted small">Satellite pixels are 10–250 m, so the model never reads them. It reads aerial tiles at 10 cm.</p>
    </div>
  )
}

const Grid4 = () => <span className="grid9 hint">{Array.from({ length: 9 }, (_, i) => <i key={i} />)}</span>

export default function VariantD() {
  const [box, setBox] = useState(null)
  const [layers, setLayers] = useState({ s2: true, hls: false, modis: false })
  const [tile, setTile] = useState(TILES[0])
  const [model, setModel] = useState('ft')
  const [tab, setTab] = useState('chat')
  const [view, setView] = useState('rgb')
  const [found, setFound] = useState([])
  useEffect(() => setFound([]), [tile.id, model])
  useEffect(() => { if (!tile.nir && model === 'ndvi') setModel('ft') }, [tile.id])
  const upload = (e) => { const f = e.target.files[0]; if (f) setTile({ id: 'up' + f.name, src: URL.createObjectURL(f), label: f.name, upload: true, nir: false }) }
  const go = (name) => { const f = SHOWCASE.find((x) => x.name === name); if (f) { setBox({ ...boxAround(f.lon, f.lat), fly: true }); setTab('field') } }

  return (
    <div className="D">
      <FieldMap box={box} onBox={(b) => { setBox(b); setTab('field') }} layers={layers} className="D-map" zoom={5} center={[-92, 40]} />

      <div className="D-top">
        <span className="brand sm"><span className="logo">◩</span> Farmland VLM</span>
        <input placeholder="Search a place…" />
        <select className="showpick" value="" onChange={(e) => go(e.target.value)}>
          <option value="">Showcase fields</option>
          {SHOWCASE.map((f) => <option key={f.name}>{f.name}</option>)}
        </select>
      </div>

      <div className="D-strip">
        {TILES.map((t) => <button key={t.id} className={'thumb sm' + (tile.id === t.id ? ' on' : '')} onClick={() => { setTile(t); setTab('chat') }}><img src={t.src} /></button>)}
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
              <TileView tile={tile} mode={view} items={found} round />
              <div className="D-imgbar">
                <div className="seg sm">{['rgb', 'ndvi', 'split'].map((v) => <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{{ rgb: 'RGB', ndvi: 'NDVI', split: 'Both' }[v]}</button>)}</div>
                <ModelPick model={model} setModel={setModel} tile={tile} />
              </div>
            </div>
            <ChatPanel tile={tile} model={model} onFindings={setFound} />
          </>
        ) : (
          <FieldPanel box={box} layers={layers} setLayers={setLayers} />
        )}
      </aside>
    </div>
  )
}
