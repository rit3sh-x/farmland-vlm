// PROTOTYPE — Variant B "Map-first": full-bleed map is the home screen; field context in a side sheet,
// images in a bottom tray, the model's reading in a floating card over the map.
import { useState } from 'react'
import { TILES, MODELS, SUGGESTED, SHOWCASE, boxAround, fieldContext } from './fake.js'
import { useCaption, useQA, Waiting, Answer, Chip, TileView, NdviLegend, FieldMap, SeasonChart } from './shared.jsx'

export default function VariantB() {
  const [box, setBox] = useState(null)
  const [layers, setLayers] = useState({ modis: false, hls: false, s2: true })
  const [tile, setTile] = useState(null)
  const [model, setModel] = useState('ft')
  const [view, setView] = useState('rgb')
  const [q, setQ] = useState('')
  const s = useCaption(tile, model)
  const { thread, ask } = useQA(tile, model)
  const ctx = box && fieldContext(box)
  const upload = (e) => { const f = e.target.files[0]; if (f) setTile({ id: 'up' + f.name, src: URL.createObjectURL(f), label: f.name, upload: true, nir: false }) }

  return (
    <div className="B">
      <FieldMap box={box} onBox={setBox} layers={layers} className="B-map" zoom={5} center={[-92, 40]} />

      <div className="B-search">
        <div className="brand dark"><span className="logo">◩</span> Farmland VLM</div>
        <input placeholder="Search a place…  (prototype: use a showcase field)" />
        <div className="showcase col">{SHOWCASE.map((f) => <button key={f.name} onClick={() => setBox({ ...boxAround(f.lon, f.lat), fly: true })}>{f.name}<small>{f.note}</small></button>)}</div>
        <p className="muted small">Or press <b>Select area</b> and drag over any field.</p>
      </div>

      {ctx && (
        <aside className="B-sheet">
          <div className="sheet-head"><h3>Field context</h3><button className="x" onClick={() => setBox(null)}>×</button></div>
          <p className="summary big">{ctx.summary}</p>
          <SeasonChart ctx={ctx} />
          <div className="layers">
            {[['s2', 'Sentinel-2 field NDVI', '10 m · 17 Oct · ' + ctx.cloud + '% cloud'], ['hls', 'NASA HLS true colour', '30 m'], ['modis', 'NASA MODIS NDVI', 'regional · 16-day']].map(([k, l, sub]) => (
              <label key={k} className="layer-row"><input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} /><span>{l}<small>{sub}</small></span></label>
            ))}
          </div>
          <NdviLegend />
          <p className="muted small">Satellite pixels are 10–250 m. The model never reads them; it reads aerial tiles at 10 cm.</p>
        </aside>
      )}

      {tile && (
        <div className="B-card">
          <div className="card-head">
            <div className="tabs">{MODELS.map((m) => <button key={m.id} disabled={m.needsNir && !tile.nir} className={model === m.id ? 'on' : ''} onClick={() => setModel(m.id)}>{m.name}</button>)}</div>
            <button className="x" onClick={() => setTile(null)}>×</button>
          </div>
          <div className="card-body">
            <div className="card-img">
              <TileView tile={tile} mode={view} items={s.phase === 'done' ? s.findings : []} round />
              <div className="viewbar mini">{['rgb', 'ndvi', 'split'].map((v) => <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{{ rgb: 'RGB', ndvi: 'NDVI', split: 'Both' }[v]}</button>)}</div>
            </div>
            <div className="card-text">
              {s.phase !== 'done' && <Waiting s={s} compact />}
              {s.text && <p className="caption">{s.text}</p>}
              {s.phase === 'done' && <div className="chips">{s.findings.map((a) => <Chip key={a.cls} a={a} />)}</div>}
              <div className="thread small">{thread.map((t) => <div key={t.id} className="qa"><div className="q">{t.q}</div><div className="a">{t.pending ? '…' : <Answer a={t.a} />}</div></div>)}</div>
              <div className="suggest small">{SUGGESTED.slice(0, 3).map((x) => <button key={x} onClick={() => ask(x)}>{x}</button>)}</div>
              <form className="ask" onSubmit={(e) => { e.preventDefault(); if (q.trim()) { ask(q); setQ('') } }}><input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask…" /><button>Ask</button></form>
            </div>
          </div>
        </div>
      )}

      <div className="B-tray">
        <span className="tray-label">Analyse an image</span>
        {TILES.map((t) => <button key={t.id} className={'thumb sm' + (tile?.id === t.id ? ' on' : '')} onClick={() => setTile(t)}><img src={t.src} /></button>)}
        <label className="upload sm">⇪<input type="file" accept="image/*" onChange={upload} /></label>
        {box && tile?.upload && <span className="pinned"><span className="pin" /> pinned to this field</span>}
      </div>
    </div>
  )
}
