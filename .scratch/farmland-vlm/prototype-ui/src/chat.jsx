// PROTOTYPE — wipe me. Chat + square-grid pieces shared by variants D and E (not layout).
import { useState, useEffect, useRef } from 'react'
import { CLASSES, MODELS, SUGGESTED } from './fake.js'
import { useCaption, useQA, Conf } from './shared.jsx'

const CELLS = ['top left', 'top', 'top right', 'left', 'centre', 'right', 'bottom left', 'bottom', 'bottom right']
const cellsOf = (loc) => (loc === 'throughout' ? CELLS : loc.split(' or '))
const striped = (c) => `repeating-linear-gradient(45deg, ${c} 0 3px, ${c}55 3px 6px)`

// The square-grid motif: where in the tile, at a glance. Hedged cells are striped.
export function Grid({ items, size = 18 }) {
  const cell = (c) => {
    const a = items.find((x) => cellsOf(x.loc).includes(c))
    if (!a) return null
    const col = CLASSES[a.cls].color
    return { background: a.hedge ? striped(col) : col }
  }
  return (
    <span className="grid9" style={{ width: size, height: size }}>
      {CELLS.map((c) => <i key={c} style={cell(c)} />)}
    </span>
  )
}

function Bars({ extent, color }) {
  const n = { small: 1, moderate: 2, large: 3 }[extent] || 0
  return <span className="bars">{[1, 2, 3].map((i) => <i key={i} style={i <= n ? { background: color } : null} />)}</span>
}

export function FindingRow({ a }) {
  const c = CLASSES[a.cls]
  return (
    <div className={'frow' + (a.hedge ? ' hedged' : '')}>
      <Grid items={[a]} size={30} />
      <span className="frow-name">{a.hedge === 'possibly' && <em>possibly </em>}{c.name}</span>
      <span className="frow-meta">{a.hedge && a.hedge.includes(' to ') ? a.hedge : a.extent} · {a.loc === 'throughout' ? 'throughout' : a.loc}</span>
      <Bars extent={a.extent} color={c.color} />
      <Conf v={a.conf} />
    </div>
  )
}

function Reading({ tile, model, onDone }) {
  const s = useCaption(tile, model)
  const m = MODELS.find((x) => x.id === model)
  useEffect(() => { if (s.phase === 'done') onDone?.(s.findings) }, [s.phase])
  return (
    <div className="bubble bot">
      <div className="who"><span>{m.name}</span><small>{m.sub}</small></div>
      {s.phase !== 'done' && (
        <div className="think"><span className="pulse" />{s.phase === 'encoding' ? (s.cold ? 'Looking at the tile' : 'Using the cached tile') : 'Writing'}<span className="secs">{((s.elapsed || 0) / 1000).toFixed(1)}s</span></div>
      )}
      {s.text && <p className="caption sm">{s.text}{s.phase === 'writing' && <span className="caret" />}</p>}
      {s.phase === 'done' && s.findings.length > 0 && (
        <div className="frows">{s.findings.map((a) => <FindingRow key={a.cls} a={a} />)}</div>
      )}
      {s.phase === 'done' && !s.findings.length && model === 'blip' && <p className="muted small">Zero-shot BLIP doesn’t name any anomaly.</p>}
      {s.cold && s.phase !== 'done' && <p className="muted small">The first look takes a few seconds on the laptop. Follow-up questions are quick.</p>}
    </div>
  )
}

function BotAnswer({ a }) {
  const col = a.cls ? CLASSES[a.cls].color : '#15201a'
  return (
    <div className={'bubble bot short' + (a.hedged ? ' hedged' : '')}>
      <div className="ans">
        {a.family === 'location' && a.text !== 'none' && <Grid items={[{ cls: a.cls, loc: a.text, hedge: a.hedged }]} size={30} />}
        {a.family === 'extent' && a.text !== 'none' && <Bars extent={a.text.split(' to ').pop()} color={col} />}
        <b>{a.text}</b>
        {a.hedged && <span className="hedge-tag">not sure</span>}
        <Conf v={a.conf} />
      </div>
      {a.why && <div className="why">{a.why}</div>}
    </div>
  )
}

export function ChatPanel({ tile, model, onFindings }) {
  const [extra, setExtra] = useState([])
  const [q, setQ] = useState('')
  const { thread, ask } = useQA(tile, model)
  const box = useRef()
  useEffect(() => setExtra([]), [tile?.id, model])
  useEffect(() => { const el = box.current; if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }) }, [thread.length, extra.length])
  if (!tile) return <div className="chat empty"><p>Pick a tile or upload a photo to start.</p></div>
  const others = MODELS.filter((m) => m.id !== model && !extra.includes(m.id) && (!m.needsNir || tile.nir))
  return (
    <div className="chat">
      <div className="chat-scroll" ref={box}>
        <Reading key={tile.id + model} tile={tile} model={model} onDone={onFindings} />
        {extra.map((m) => <Reading key={tile.id + m} tile={tile} model={m} />)}
        {others.length > 0 && <div className="also">Compare with {others.map((m) => <button key={m.id} onClick={() => setExtra([...extra, m.id])}>{m.name}</button>)}</div>}
        {thread.map((t) => (
          <div key={t.id} className="pair">
            <div className="bubble user">{t.q}</div>
            {t.pending ? <div className="bubble bot short"><span className="dots">•••</span></div> : <BotAnswer a={t.a} />}
          </div>
        ))}
      </div>
      <div className="chat-compose">
        <div className="sugg">{SUGGESTED.map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div>
        <form onSubmit={(e) => { e.preventDefault(); if (q.trim()) { ask(q); setQ('') } }}>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this tile…" />
          <button className="send">↑</button>
        </form>
      </div>
    </div>
  )
}

export function ModelPick({ model, setModel, tile }) {
  return (
    <select className="modelpick" value={model} onChange={(e) => setModel(e.target.value)}>
      {MODELS.map((m) => <option key={m.id} value={m.id} disabled={m.needsNir && !tile?.nir}>{m.name}{m.needsNir && !tile?.nir ? ' (needs NIR)' : ''}</option>)}
    </select>
  )
}
