// PROTOTYPE — Variant C "Chat-first": one conversation per image. Captions, findings, hedges and the
// field context all arrive as messages; other models are asked from under any reply.
import { useState, useEffect, useRef } from 'react'
import { TILES, MODELS, SUGGESTED, SHOWCASE, boxAround, fieldContext } from './fake.js'
import { useCaption, useQA, Waiting, Answer, Chip, TileView, NdviLegend, FieldMap, SeasonChart } from './shared.jsx'

function CaptionMsg({ tile, model }) {
  const s = useCaption(tile, model)
  const m = MODELS.find((x) => x.id === model)
  return (
    <div className="msg bot">
      <div className="who">{m.name} <small>{m.sub}</small></div>
      {s.phase !== 'done' && <Waiting s={s} />}
      {s.text && <p className="caption">{s.text}</p>}
      {s.phase === 'done' && s.findings.length > 0 && <div className="chips">{s.findings.map((a) => <Chip key={a.cls} a={a} />)}</div>}
    </div>
  )
}

function ContextMsg({ box, setBox }) {
  const [layers, setLayers] = useState({ s2: true, hls: false, modis: false })
  const ctx = box && fieldContext(box)
  return (
    <div className="msg bot wide">
      <div className="who">Field context <small>satellite · not read by the model</small></div>
      <FieldMap box={box} onBox={setBox} layers={layers} className="C-map" />
      <div className="showcase">{SHOWCASE.map((f) => <button key={f.name} onClick={() => setBox({ ...boxAround(f.lon, f.lat), fly: true })}>{f.name}</button>)}</div>
      {ctx && <>
        <div className="layers inline">{[['s2', 'S2 NDVI'], ['hls', 'HLS'], ['modis', 'MODIS']].map(([k, l]) => <label key={k}><input type="checkbox" checked={layers[k]} onChange={(e) => setLayers({ ...layers, [k]: e.target.checked })} /> {l}</label>)}<NdviLegend /></div>
        <SeasonChart ctx={ctx} w={420} />
        <p className="summary">{ctx.summary}</p>
      </>}
    </div>
  )
}

export default function VariantC() {
  const [tile, setTile] = useState(TILES[1])
  const [models, setModels] = useState(['ft'])
  const [view, setView] = useState('split')
  const [pin, setPin] = useState(false)
  const [box, setBox] = useState(null)
  const [q, setQ] = useState('')
  const { thread, ask } = useQA(tile, 'ft')
  const end = useRef()
  useEffect(() => { setModels(['ft']); setPin(false) }, [tile.id])
  useEffect(() => { end.current?.scrollIntoView({ behavior: 'smooth' }) }, [thread.length, models.length, pin])
  const upload = (e) => { const f = e.target.files[0]; if (f) setTile({ id: 'up' + f.name, src: URL.createObjectURL(f), label: f.name, upload: true, nir: false }) }
  const others = MODELS.filter((m) => !models.includes(m.id) && (!m.needsNir || tile.nir))

  return (
    <div className="C">
      <aside className="C-side">
        <div className="brand"><span className="logo">◩</span> Farmland VLM</div>
        <label className="upload block">+ New: upload a photo<input type="file" accept="image/*" onChange={upload} /></label>
        <h4>Dataset tiles</h4>
        {TILES.map((t) => <button key={t.id} className={'conv' + (tile.id === t.id ? ' on' : '')} onClick={() => setTile(t)}><img src={t.src} /><span>{t.label}</span></button>)}
      </aside>

      <main className="C-main">
        <div className="C-thread">
          <div className="msg user img">
            <TileView tile={tile} mode={view} round />
            <div className="viewbar mini">{['rgb', 'ndvi', 'split'].map((v) => <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{{ rgb: 'RGB', ndvi: 'NDVI', split: 'Both' }[v]}</button>)}</div>
          </div>
          <div className="replies">{models.map((m) => <CaptionMsg key={tile.id + m} tile={tile} model={m} />)}</div>
          {others.length > 0 && <div className="also">Also ask: {others.map((m) => <button key={m.id} onClick={() => setModels([...models, m.id])}>{m.name}</button>)}</div>}
          {thread.map((t) => (
            <div key={t.id}>
              <div className="msg user text">{t.q}</div>
              <div className="msg bot short">{t.pending ? <span className="dots">…</span> : <Answer a={t.a} />}</div>
            </div>
          ))}
          {pin && <ContextMsg box={box} setBox={setBox} />}
          <div ref={end} />
        </div>
        <div className="C-composer">
          <div className="suggest">{SUGGESTED.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div>
          <form className="ask big" onSubmit={(e) => { e.preventDefault(); if (q.trim()) { ask(q); setQ('') } }}>
            <button type="button" className="ghost" title="Where was this taken?" onClick={() => setPin(true)}><span className="pin" /></button>
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this tile…" />
            <button>Send</button>
          </form>
        </div>
      </main>
    </div>
  )
}
