// PROTOTYPE — Variant A "Workbench": three columns (source · image · reading) with the field map in a bottom drawer.
import { useState } from 'react'
import { TILES, MODELS, SUGGESTED, SHOWCASE, boxAround, fieldContext } from './fake.js'
import { useCaption, useQA, Waiting, Answer, Chip, TileView, NdviLegend, FieldMap, SeasonChart } from './shared.jsx'

function Reading({ tile, model, compact }) {
  const s = useCaption(tile, model)
  const m = MODELS.find((x) => x.id === model)
  return (
    <div className={'reading' + (compact ? ' compact' : '')}>
      {compact && <div className="reading-head">{m.name}<small>{m.sub}</small></div>}
      {s.phase !== 'done' ? <Waiting s={s} compact={compact} /> : null}
      {s.text && <p className="caption">{s.text}{s.phase === 'writing' && <span className="caret" />}</p>}
      {s.phase === 'done' && (
        <div className="chips">
          {s.findings.length ? s.findings.map((a) => <Chip key={a.cls} a={a} />) : <span className="muted">{model === 'blip' ? 'Zero-shot BLIP names no anomalies' : 'No anomalies found'}</span>}
        </div>
      )}
    </div>
  )
}

export default function VariantA() {
  const [tile, setTile] = useState(TILES[0])
  const [model, setModel] = useState('ft')
  const [compare, setCompare] = useState(false)
  const [view, setView] = useState('rgb')
  const [truth, setTruth] = useState(false)
  const [drawer, setDrawer] = useState(false)
  const [box, setBox] = useState(null)
  const [layers, setLayers] = useState({ modis: false, hls: false, s2: true })
  const [q, setQ] = useState('')
  const { thread, ask } = useQA(tile, model)
  const cap = useCaption(tile, model)
  const upload = (e) => { const f = e.target.files[0]; if (f) setTile({ id: 'up' + f.name, src: URL.createObjectURL(f), label: f.name, upload: true, nir: false }) }
  const ctx = box && fieldContext(box)
  const items = cap.phase === 'done' ? cap.findings : []

  return (
    <div className="A">
      <header className="A-head">
        <div className="brand"><span className="logo">◩</span> Farmland VLM <small>crop anomaly reader</small></div>
        <div className="seg">
          {MODELS.map((m) => (
            <button key={m.id} disabled={m.needsNir && !tile.nir} className={model === m.id && !compare ? 'on' : ''} onClick={() => { setModel(m.id); setCompare(false) }} title={m.needsNir && !tile.nir ? 'needs a NIR band' : m.sub}>{m.name}</button>
          ))}
          <button className={compare ? 'on compare' : 'compare'} onClick={() => setCompare(!compare)}>Compare all</button>
        </div>
      </header>

      <aside className="A-src">
        <h4>Dataset tiles</h4>
        <div className="thumbs">
          {TILES.map((t) => <button key={t.id} className={'thumb' + (tile.id === t.id ? ' on' : '')} onClick={() => setTile(t)}><img src={t.src} /><span>{t.label}</span></button>)}
        </div>
        <label className="upload">⇪ Upload a photo<input type="file" accept="image/*" onChange={upload} /></label>
        <h4>Location</h4>
        {box ? <div className="pin-on"><span className="pin" /> Field box set · {ctx.summary.split(';')[0]}<button className="link" onClick={() => setDrawer(true)}>open</button></div>
          : <button className="ghost" onClick={() => setDrawer(true)}><span className="pin" /> Add where this was taken</button>}
      </aside>

      <main className="A-main">
        <div className="viewbar">
          {['rgb', 'ndvi', 'split'].map((v) => <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{{ rgb: 'RGB', ndvi: 'NDVI', split: 'Side by side' }[v]}</button>)}
          <span className="spacer" />
          {!tile.upload && <label className="toggle"><input type="checkbox" checked={truth} onChange={(e) => setTruth(e.target.checked)} /> show ground truth</label>}
          {view !== 'rgb' && <NdviLegend />}
        </div>
        <TileView tile={tile} mode={view} items={items} truth={truth} />
      </main>

      <section className="A-read">
        {compare ? (
          <div className="compare-cols">{MODELS.filter((m) => !m.needsNir || tile.nir).map((m) => <Reading key={m.id} tile={tile} model={m.id} compact />)}</div>
        ) : (
          <>
            <h4>What the model sees</h4>
            <Reading tile={tile} model={model} />
            <h4>Ask about this tile</h4>
            <div className="thread">
              {thread.map((t) => (
                <div key={t.id} className="qa"><div className="q">{t.q}</div><div className="a">{t.pending ? <span className="dots">…</span> : <Answer a={t.a} />}</div></div>
              ))}
            </div>
            <div className="suggest">{SUGGESTED.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div>
            <form className="ask" onSubmit={(e) => { e.preventDefault(); if (q.trim()) { ask(q); setQ('') } }}>
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a question about this tile…" />
              <button>Ask</button>
            </form>
          </>
        )}
      </section>

      <section className={'A-drawer' + (drawer ? ' open' : '')}>
        <button className="drawer-handle" onClick={() => setDrawer(!drawer)}>{drawer ? '▾' : '▴'} Field context {box ? '· ' + ctx.summary.split(';')[0] : '— select an area on the map'}</button>
        {drawer && (
          <div className="drawer-body">
            <FieldMap box={box} onBox={setBox} layers={layers} className="A-map" />
            <div className="ctx">
              <div className="showcase">{SHOWCASE.map((f) => <button key={f.name} onClick={() => setBox({ ...boxAround(f.lon, f.lat), fly: true })}>{f.name}<small>{f.note}</small></button>)}</div>
              <div className="layers">
                {[['s2', 'Sentinel-2 field NDVI · 10 m'], ['hls', 'NASA HLS true colour · 30 m'], ['modis', 'NASA MODIS NDVI · regional']].map(([k, l]) => (
                  <label key={k}><input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} /> {l}</label>
                ))}
              </div>
              {ctx ? <><SeasonChart ctx={ctx} /><p className="summary">{ctx.summary}</p></> : <p className="muted">Use “Select area” and drag over a field, or pick a showcase field.</p>}
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
