// PROTOTYPE — wipe me. Fake tiles, a fake model and fake field numbers. Nothing here is real output.

export const CLASSES = {
  drydown: { name: 'drydown', color: '#c8893a' },
  nutrient_deficiency: { name: 'nutrient deficiency', color: '#c9b21f' },
  weed_cluster: { name: 'weed cluster', color: '#7a5bd6' },
  water: { name: 'standing water', color: '#2f7fd1' },
  endrow: { name: 'endrow', color: '#d1567a' },
  double_plant: { name: 'double plant', color: '#3aa58a' },
  planter_skip: { name: 'planter skip', color: '#8a6d4f' },
  waterway: { name: 'waterway', color: '#3bb3c9' },
  storm_damage: { name: 'storm damage', color: '#555' },
}

// truth = what the masks say (made up for the prototype). hedge = the ambiguous-case hedge, if flagged.
export const TILES = [
  { id: 'drydown', src: '/tiles/drydown.jpg', label: 'Val tile A · 2019', nir: true,
    truth: [{ cls: 'drydown', extent: 'large', loc: 'throughout' }] },
  { id: 'weed', src: '/tiles/weed_cluster.jpg', label: 'Val tile B · 2018', nir: true,
    truth: [{ cls: 'weed_cluster', extent: 'small', loc: 'top left' },
            { cls: 'nutrient_deficiency', extent: 'small', loc: 'bottom', hedge: 'possibly' }] },
  { id: 'nutrient', src: '/tiles/nutrient_deficiency.jpg', label: 'Val tile C · 2018', nir: true,
    truth: [{ cls: 'nutrient_deficiency', extent: 'moderate', loc: 'centre', hedge: 'small to moderate' }] },
  { id: 'water', src: '/tiles/water.jpg', label: 'Val tile D · 2017', nir: true,
    truth: [{ cls: 'water', extent: 'moderate', loc: 'right or bottom right', hedge: 'right or bottom right' }] },
]

export const MODELS = [
  { id: 'ft', name: 'Fine-tuned', sub: 'served model · RGB', cold: 5200 },
  { id: 'ndvi', name: 'Fine-tuned + NDVI', sub: 'RGB + NDVI map', cold: 8400, needsNir: true },
  { id: 'blip', name: 'Zero-shot BLIP', sub: 'baseline · no fine-tune', cold: 3100 },
]

const conf = (seed) => 0.55 + ((seed * 9301 + 49297) % 233280) / 233280 * 0.4

export function findings(tile, model) {
  if (model === 'blip' || !tile.truth) return []
  return tile.truth.map((a, i) => ({ ...a, conf: a.hedge ? 0.48 + i * 0.05 : conf(i + (model === 'ndvi' ? 7 : 1)) + (model === 'ndvi' ? 0.04 : 0) }))
}

const cap = (s) => s[0].toUpperCase() + s.slice(1)
function phrase(a) {
  const ext = a.hedge && a.hedge.includes(' to ') ? a.hedge : a.extent
  const loc = a.loc === 'throughout' ? 'throughout the tile' : a.loc === 'centre' ? 'in the centre' : `at the ${a.loc}`
  return `${a.hedge === 'possibly' ? 'possibly ' : ''}${ext} ${CLASSES[a.cls].name} ${loc}`
}

export function caption(tile, model) {
  if (model === 'blip') return tile.upload ? 'a picture of a field with grass' : 'an aerial view of a farm field with trees'
  const f = findings(tile, model)
  if (!f.length) return 'No anomalies visible.'
  const [first, ...rest] = f
  return cap(phrase(first)) + (rest.length ? ', with ' + rest.map(phrase).join(' and ') : '') + '.'
}

export const SUGGESTED = [
  'What anomalies are visible?',
  'Is there weed cluster in this tile?',
  'How much of the tile does the drydown cover?',
  'Where is the standing water?',
]

// Very rough fake VQA: family by keyword, answer from the fake truth.
export function answer(q, tile, model) {
  const t = q.toLowerCase()
  const named = Object.keys(CLASSES).find((k) => t.includes(CLASSES[k].name) || t.includes(k.replace('_', ' ')) || (k === 'water' && t.includes('water') && !t.includes('waterway')))
  const f = findings(tile, model)
  const hit = f.find((a) => a.cls === named)
  if (model === 'blip') {
    if (t.startsWith('is ') || t.startsWith('are ')) return { text: 'yes', conf: 0.71 }
    if (t.startsWith('where')) return { text: 'field', conf: 0.33 }
    if (t.startsWith('how much')) return { text: 'lot', conf: 0.29 }
    return { text: 'grass', conf: 0.41 }
  }
  if (t.startsWith('where')) {
    if (!hit) return { text: 'none', conf: 0.8, family: 'location' }
    return hit.hedge && hit.hedge.includes(' or ') ? { text: hit.hedge, hedged: true, conf: 0.46, why: 'Two parts of the tile hold almost the same share of it.', family: 'location', cls: hit.cls }
      : { text: hit.loc, conf: hit.conf, family: 'location', cls: hit.cls }
  }
  if (t.startsWith('how much') || t.includes('extent') || t.includes('cover')) {
    if (!hit) return { text: 'none', conf: 0.8, family: 'extent' }
    return hit.hedge && hit.hedge.includes(' to ') ? { text: hit.hedge, hedged: true, conf: 0.51, why: 'Its area is close to the small/moderate cut-off.', family: 'extent', cls: hit.cls }
      : { text: hit.extent, conf: hit.conf, family: 'extent', cls: hit.cls }
  }
  if (t.startsWith('is ') || t.startsWith('are ') || t.startsWith('any')) {
    if (!named) return { text: f.length ? 'yes' : 'no', conf: 0.9, family: 'presence' }
    if (hit && hit.hedge === 'possibly') return { text: 'possibly', hedged: true, conf: 0.49, why: 'It covers under 1% of the field — a few pixels decide it.', family: 'presence' }
    return { text: hit ? 'yes' : 'no', conf: hit ? hit.conf : 0.93, family: 'presence' }
  }
  const names = f.map((a) => (a.hedge === 'possibly' ? 'possibly ' : '') + CLASSES[a.cls].name)
  return { text: names.length ? names.join(', ') : 'none', hedged: f.some((a) => a.hedge === 'possibly'), conf: 0.82, family: 'identify',
    why: f.some((a) => a.hedge === 'possibly') ? 'One of them covers only a sliver of the field.' : undefined }
}

// ---- Location imagery (fake numbers, real map) ----

export const SHOWCASE = [
  { name: 'Story County, Iowa', note: 'corn · cached', lon: -93.700, lat: 42.000 },
  { name: 'Champaign, Illinois', note: 'soy · cached', lon: -88.280, lat: 40.080 },
  { name: 'Lancaster, Nebraska', note: 'corn · cached', lon: -96.850, lat: 40.750 },
  { name: 'Rupnagar, Punjab', note: 'near IIT Ropar · cached', lon: 76.470, lat: 30.970 },
]

export function boxAround(lon, lat, m = 800) {
  const dLat = m / 2 / 110540, dLon = m / 2 / (111320 * Math.cos(lat * Math.PI / 180))
  return { w: lon - dLon, e: lon + dLon, s: lat - dLat, n: lat + dLat }
}

export function boxSize(b) {
  const lat = (b.n + b.s) / 2
  const wm = (b.e - b.w) * 111320 * Math.cos(lat * Math.PI / 180), hm = (b.n - b.s) * 110540
  return { wm, hm, ha: (wm * hm) / 1e4 }
}

function rng(seed) { let s = Math.floor(Math.abs(seed) * 1e6) % 2147483647 || 7; return () => (s = (s * 16807) % 2147483647) / 2147483647 }

export function fieldContext(b) {
  const r = rng(b.w * 13 + b.n * 7)
  const months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
  const series = []
  for (let i = 0; i < 14; i++) {
    const x = i / 13, base = 0.25 + 0.55 * Math.exp(-((x - 0.55) ** 2) / 0.06)
    series.push({ x, v: Math.max(0.05, Math.min(0.92, base + (r() - 0.5) * 0.08)) })
  }
  const peak = series.reduce((a, p) => (p.v > a.v ? p : a))
  const now = series[series.length - 1]
  const parts = ['south-west third', 'north edge', 'east half', 'centre']
  const low = parts[Math.floor(r() * parts.length)]
  const dateOf = (x) => { const d = new Date(2026, 4, 1 + Math.round(x * 170)); return d.toLocaleDateString('en-GB', { day: 'numeric', month: 'short' }) }
  return {
    series, months, now, peak, low, sceneDate: '17 Oct', cloud: Math.round(r() * 12),
    summary: `Field NDVI ${now.v.toFixed(2)} on 17 Oct, ${(peak.v - now.v).toFixed(2)} below its ${dateOf(peak.x)} peak; the ${low} is lowest. Satellite data shows where, not why.`,
  }
}

export function ndviColor(v) {
  const stops = [[0, [165, 0, 38]], [0.25, [244, 109, 67]], [0.5, [254, 224, 139]], [0.75, [166, 217, 106]], [1, [26, 152, 80]]]
  v = Math.max(0, Math.min(1, v))
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i], t = (v - a) / (b - a)
      return ca.map((c, k) => Math.round(c + (cb[k] - c) * t))
    }
  }
  return stops[4][1]
}

// Fake Sentinel-2 NDVI cut-out for the field box: smooth noise, coloured like the tile NDVI map.
export function fakeS2Ndvi(b, size = 96) {
  const r = rng(b.e * 31 + b.s * 17)
  const blobs = Array.from({ length: 5 }, () => ({ x: r(), y: r(), s: 0.08 + r() * 0.2, d: (r() - 0.6) * 0.7 }))
  const c = document.createElement('canvas'); c.width = c.height = size
  const ctx = c.getContext('2d'), img = ctx.createImageData(size, size)
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let v = 0.7 + (r() - 0.5) * 0.06
    for (const bl of blobs) v += bl.d * Math.exp(-(((x / size - bl.x) ** 2 + (y / size - bl.y) ** 2) / (bl.s * bl.s)))
    const [R, G, B] = ndviColor(v), i = (y * size + x) * 4
    img.data[i] = R; img.data[i + 1] = G; img.data[i + 2] = B; img.data[i + 3] = 235
  }
  ctx.putImageData(img, 0, 0)
  return c.toDataURL()
}
