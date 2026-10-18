import { useEffect, useRef, useState } from 'react'
import { api } from './api.js'
import { Bars, Conf, FindingRow, Grid } from './components.jsx'
import { CLASS_BY_KEY, MODELS, suggestions } from './scheme.js'

function useRead(image, model) {
  const [s, set] = useState({ phase: 'idle' })
  useEffect(() => {
    if (!image) return
    let live = true
    const t0 = performance.now()
    const tick = setInterval(() => live && set((x) => (x.phase === 'done' ? x : { ...x, elapsed: performance.now() - t0 })), 100)
    set({ phase: 'queued', text: '' })
    ;(async () => {
      let text = ''
      for await (const ev of api.read(image.ref, model)) {
        if (!live) return
        if (ev.type === 'queued') set((x) => ({ ...x, phase: ev.ahead ? 'queued' : 'encoding', ahead: ev.ahead }))
        else if (ev.type === 'phase') set((x) => ({ ...x, phase: ev.phase }))
        else if (ev.type === 'token') { text += ev.text; set((x) => ({ ...x, phase: 'writing', text })) }
        else if (ev.type === 'done') set({ phase: 'done', text: ev.caption, findings: ev.findings, timings: ev.timings, cached: ev.cached })
        else if (ev.type === 'error') set({ phase: 'error', message: ev.message })
      }
    })()
    return () => { live = false; clearInterval(tick) }
  }, [image?.key, model])
  return s
}

function Reading({ image, model, onFindings }) {
  const s = useRead(image, model)
  const m = MODELS.find((x) => x.id === model)
  useEffect(() => { if (s.phase === 'done') onFindings?.(s.findings) }, [s.phase])
  const secs = ((s.elapsed || 0) / 1000).toFixed(1)
  return (
    <div className="bubble bot">
      <div className="who"><span>{m.name}</span><small>{m.sub}</small></div>
      {['queued', 'encoding', 'writing'].includes(s.phase) && (
        <div className="think"><span className="pulse" />
          {s.phase === 'queued' && s.ahead ? `Waiting for ${s.ahead} other request${s.ahead > 1 ? 's' : ''}` : s.phase === 'writing' ? 'Writing' : 'Looking at the tile'}
          <span className="secs">{secs}s</span></div>
      )}
      {s.phase === 'error' && <p className="error">{s.message}</p>}
      {s.text && <p className="caption sm">{s.text}{s.phase === 'writing' && <span className="caret" />}</p>}
      {s.phase === 'done' && s.findings?.length > 0 && <div className="frows">{s.findings.map((f, i) => <FindingRow key={i} f={f} />)}</div>}
      {s.phase === 'done' && !s.findings?.length && <p className="muted small">{model === 'blip' ? 'Zero-shot BLIP names no anomaly in our vocabulary.' : 'No anomalies found.'}</p>}
      {s.phase === 'done' && s.timings && <p className="muted tiny">{(s.timings.total_ms / 1000).toFixed(1)} s{s.cached ? ' · cached tile' : ''}</p>}
      {['queued', 'encoding'].includes(s.phase) && !s.cached && <p className="muted small">The first look takes a few seconds on the laptop. Follow-up questions are quick.</p>}
    </div>
  )
}

function Answer({ a }) {
  const cls = a.cls ? CLASS_BY_KEY[a.cls] : null
  return (
    <div className={'bubble bot short' + (a.hedged ? ' hedged' : '')}>
      <div className="ans">
        {a.family === 'location' && a.answer !== 'none' && cls && <Grid items={[{ cls: cls.index, loc: a.answer, hedge: a.hedged }]} size={30} />}
        {a.family === 'extent' && a.answer !== 'none' && cls && <Bars extent={a.answer} color={cls.color} />}
        <b>{a.answer}</b>
        {a.hedged && <span className="hedge-tag">not sure</span>}
        <Conf v={a.conf} />
      </div>
      {a.why && <div className="why">{a.why}</div>}
    </div>
  )
}

export function ChatPanel({ image, model, onFindings }) {
  const [extra, setExtra] = useState([])
  const [thread, setThread] = useState([])
  const [found, setFound] = useState([])
  const [q, setQ] = useState('')
  const box = useRef()
  useEffect(() => { setExtra([]); setThread([]); setFound([]) }, [image?.key, model])
  useEffect(() => { const el = box.current; if (el) el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' }) }, [thread.length, extra.length])
  if (!image) return <div className="chat empty"><p>Pick a tile or upload a photo to start.</p></div>

  const ask = async (question) => {
    const id = Math.random()
    setThread((t) => [...t, { id, q: question, pending: true }])
    for await (const ev of api.ask(image.ref, model, question)) {
      if (ev.type === 'done' || ev.type === 'error') setThread((t) => t.map((x) => (x.id === id ? { ...x, pending: false, a: ev } : x)))
    }
  }
  const others = MODELS.filter((m) => m.id !== model && !extra.includes(m.id) && (!m.needsNir || image.hasNir))
  return (
    <div className="chat">
      <div className="chat-scroll" ref={box}>
        <Reading image={image} model={model} onFindings={(f) => { setFound(f); onFindings?.(f) }} />
        {extra.map((m) => <Reading key={m} image={image} model={m} />)}
        {others.length > 0 && <div className="also">Compare with {others.map((m) => <button key={m.id} onClick={() => setExtra([...extra, m.id])}>{m.name}</button>)}</div>}
        {thread.map((t) => (
          <div key={t.id} className="pair">
            <div className="bubble user">{t.q}</div>
            {t.pending ? <div className="bubble bot short"><span className="dots">•••</span></div>
              : t.a.type === 'error' ? <div className="bubble bot short error">{t.a.message}</div> : <Answer a={t.a} />}
          </div>
        ))}
      </div>
      <div className="chat-compose">
        <div className="sugg">{suggestions(found).map((s) => <button key={s} onClick={() => ask(s)}>{s}</button>)}</div>
        <form onSubmit={(e) => { e.preventDefault(); if (q.trim()) { ask(q.trim()); setQ('') } }}>
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this tile…" />
          <button className="send" aria-label="Send">↑</button>
        </form>
      </div>
    </div>
  )
}
