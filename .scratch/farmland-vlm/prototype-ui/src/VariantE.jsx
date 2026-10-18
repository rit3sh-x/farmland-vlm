// PROTOTYPE — Variant E "Focus": a cleaner A. Two zones only: a big stage (the tile, or the field map)
// and a chat column. No side rail, no drawer; tiles live in a filmstrip under the stage.
import { useState, useEffect } from 'react'
import { TILES, SHOWCASE, boxAround } from './fake.js'
import { TileView, NdviLegend, FieldMap } from './shared.jsx'
import { ChatPanel, ModelPick } from './chat.jsx'
import { FieldPanel } from './VariantD.jsx'

export default function VariantE() {
  const [tile, setTile] = useState(TILES[2])
  const [model, setModel] = useState('ft')
  const [stage, setStage] = useState('image')
  const [view, setView] = useState('rgb')
  const [truth, setTruth] = useState(false)
  const [found, setFound] = useState([])
  const [box, setBox] = useState(null)
  const [layers, setLayers] = useState({ s2: true, hls: false, modis: false })
  useEffect(() => setFound([]), [tile.id, model])
  useEffect(() => { if (!tile.nir && model === 'ndvi') setModel('ft') }, [tile.id])
  const upload = (e) => { const f = e.target.files[0]; if (f) setTile({ id: 'up' + f.name, src: URL.createObjectURL(f), label: f.name, upload: true, nir: false }) }

  return (
    <div className="E">
      <header className="E-head">
        <span className="brand"><span className="logo">◩</span> Farmland VLM</span>
        <div className="seg">
          <button className={stage === 'image' ? 'on' : ''} onClick={() => setStage('image')}>Tile</button>
          <button className={stage === 'field' ? 'on' : ''} onClick={() => setStage('field')}>Field{box && <span className="dotmark" />}</button>
        </div>
        <ModelPick model={model} setModel={setModel} tile={tile} />
      </header>

      <main className="E-stage">
        {stage === 'image' ? (
          <div className="E-imagewrap">
            <div className="E-image">
              <TileView tile={tile} mode={view} items={found} truth={truth} />
              <div className="E-float">
                <div className="seg sm glass">{['rgb', 'ndvi', 'split'].map((v) => <button key={v} className={view === v ? 'on' : ''} onClick={() => setView(v)}>{{ rgb: 'RGB', ndvi: 'NDVI', split: 'Both' }[v]}</button>)}</div>
                {!tile.upload && <button className={'glass sm' + (truth ? ' on' : '')} onClick={() => setTruth(!truth)}>Ground truth</button>}
                {view !== 'rgb' && <span className="glass pad"><NdviLegend /></span>}
              </div>
            </div>
            <div className="E-film">
              {TILES.map((t) => <button key={t.id} className={'thumb sm' + (tile.id === t.id ? ' on' : '')} onClick={() => setTile(t)}><img src={t.src} /></button>)}
              <label className="upload sm" title="Upload a photo">+<input type="file" accept="image/*" onChange={upload} /></label>
              <span className="spacer" />
              <button className="ghost sm" onClick={() => setStage('field')}><span className="pin" /> {box ? 'Field set' : 'Where was this taken?'}</button>
            </div>
          </div>
        ) : (
          <div className="E-field">
            <FieldMap box={box} onBox={setBox} layers={layers} className="E-map" />
            <div className="E-fieldcard">
              <div className="showcase">{SHOWCASE.map((f) => <button key={f.name} onClick={() => setBox({ ...boxAround(f.lon, f.lat), fly: true })}>{f.name}</button>)}</div>
              <FieldPanel box={box} layers={layers} setLayers={setLayers} />
            </div>
          </div>
        )}
      </main>

      <aside className="E-chat"><ChatPanel tile={tile} model={model} onFindings={setFound} /></aside>
    </div>
  )
}
