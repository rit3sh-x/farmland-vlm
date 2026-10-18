// Client for the FastAPI backend (issues/14-api-contract.md). Model calls stream NDJSON, one event per line.

async function* ndjson(res) {
  if (!res.ok) {
    let detail = await res.text()
    try { detail = JSON.parse(detail).detail } catch {}
    yield { type: 'error', code: detail?.code || res.status, message: detail?.message || String(detail) }
    return
  }
  const reader = res.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let i
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim()
      buf = buf.slice(i + 1)
      if (line) yield JSON.parse(line)
    }
  }
}

const post = (path, body) => fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export const api = {
  health: () => fetch('/api/health').then((r) => r.json()),
  tiles: () => fetch('/api/tiles').then((r) => r.json()),
  truth: (id) => fetch(`/api/tiles/${id}/truth`).then((r) => r.json()),
  showcase: () => fetch('/api/showcase').then((r) => r.json()),
  upload: async (file) => {
    const fd = new FormData()
    fd.append('file', file)
    const r = await fetch('/api/images', { method: 'POST', body: fd })
    if (!r.ok) throw new Error('upload failed')
    return r.json()
  },
  read: async function* (image, model) { yield* ndjson(await post('/api/read', { image, model })) },
  ask: async function* (image, model, question) { yield* ndjson(await post('/api/ask', { image, model, question })) },
  field: async function* (bbox) { yield* ndjson(await post('/api/field', { bbox })) },
}

export async function geocode(q) {
  const r = await fetch(`https://photon.komoot.io/api/?q=${encodeURIComponent(q)}&limit=5`)
  const j = await r.json()
  return j.features.map((f) => ({
    name: [f.properties.name, f.properties.state, f.properties.country].filter(Boolean).join(', '),
    lon: f.geometry.coordinates[0], lat: f.geometry.coordinates[1], extent: f.properties.extent,
  }))
}
