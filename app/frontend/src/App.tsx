import { useEffect, useState, type ChangeEvent, type FormEvent } from "react";
import { api, geocode, type GeoHit } from "./api";
import { ChatPanel } from "./Chat";
import {
    FieldMap,
    NdviLegend,
    SeasonChart,
    TileView,
    boxOk,
    boxSize,
    describeBox,
    type FlyTo,
    type Layers,
    type ViewMode,
} from "./components";
import { IconMap, IconTile, IconUpload, Mark } from "./icons";
import type {
    BBox,
    FieldState,
    Finding,
    ModelId,
    ModelInfo,
    Place,
    Tile,
    ViewImage,
} from "./types";

type Tool = "reader" | "field";

function useField(box: BBox | null) {
    const [s, set] = useState<FieldState>({});
    useEffect(() => {
        set({});
        if (!box || !boxOk(box)) return;
        let live = true;
        set({ loading: true });
        (async () => {
            for await (const ev of api.field(box)) {
                if (!live) return;
                if (ev.type === "unavailable")
                    set((x) => ({
                        ...x,
                        unavailable: [...(x.unavailable || []), ev],
                    }));
                else if (ev.type === "done")
                    set((x) => ({ ...x, loading: false }));
                else if (ev.type === "error")
                    set((x) => ({ ...x, loading: false, error: ev.message }));
                else set((x) => ({ ...x, [ev.type]: ev }));
            }
        })();
        return () => {
            live = false;
        };
    }, [box && box.join()]);
    return s;
}

interface FieldPanelProps {
    box: BBox | null;
    field: FieldState;
    layers: Layers;
    setLayers: (l: Layers) => void;
}

function FieldPanel({ box, field, layers, setLayers }: FieldPanelProps) {
    if (!box)
        return (
            <div className="field-empty">
                <span className="grid9 hint">
                    {Array.from({ length: 9 }, (_, i) => (
                        <i key={i} />
                    ))}
                </span>
                <p>
                    Choose <b>Select area</b> and drag over a field, or search
                    for a place. Boxes from 50 m to 5 km a side work.
                </p>
            </div>
        );
    if (!boxOk(box))
        return (
            <div className="field-empty">
                <p className="error tnum">That box is {describeBox(box)}.</p>
                <p>
                    Draw between 50 m and 5 km a side so the satellite pixels
                    (10 m) can describe a single field.
                </p>
            </div>
        );
    const sc = field.scene,
        w = field.weather,
        size = boxSize(box);
    const rows: [keyof Layers, string, string][] = [
        [
            "s2",
            "Sentinel-2 field NDVI",
            sc ? `10 m · ${sc.date} · ${sc.cloud_pct}% cloud` : "loading…",
        ],
        [
            "hls",
            "NASA HLS true colour",
            sc?.hls_date ? `30 m · ${sc.hls_date}` : "no clear pass found",
        ],
        ["modis", "NASA MODIS NDVI", "regional · 16-day"],
    ];
    return (
        <div className="field-panel">
            <p className="muted small tnum">
                {describeBox(box)} · {size.ha.toFixed(1)} ha
            </p>
            {field.error && <p className="error">{field.error}</p>}
            {field.summary ? (
                <p className="summary">{field.summary.text}</p>
            ) : (
                !field.error && (
                    <p className="summary muted">
                        {field.loading
                            ? "Reading satellite data for this box…"
                            : "No summary came back for this box."}
                    </p>
                )
            )}
            <SeasonChart
                season={field.season?.series ? field.season : null}
                weather={w}
            />
            {field.season && (
                <p className="chart-key">
                    Dots: field NDVI from NASA HLS · dashed: root-zone soil
                    wetness · bars: daily rain (NASA POWER)
                </p>
            )}
            {w && (
                <p className="muted small tnum">
                    Rain since {w.daily[0]?.date}: {w.rain_mm} mm (
                    {w.rain_vs_normal >= 0 ? "+" : ""}
                    {Math.round(w.rain_vs_normal * 100)}% vs NASA POWER normal).
                </p>
            )}
            <fieldset className="layers">
                <legend>Map layers</legend>
                {rows.map(([k, l, sub]) => (
                    <label key={k} className="layer-row">
                        <input
                            type="checkbox"
                            checked={layers[k]}
                            onChange={(e) =>
                                setLayers({ ...layers, [k]: e.target.checked })
                            }
                        />
                        <span>
                            {l}
                            <small>{sub}</small>
                        </span>
                    </label>
                ))}
            </fieldset>
            <NdviLegend />
            {field.unavailable?.map((u) => (
                <p key={u.part} className="muted small">
                    {u.part} unavailable: {u.reason}
                </p>
            ))}
            <p className="muted small">
                Satellite pixels are 10 to 250 m, far coarser than the 10 cm
                aerial tiles the model was trained on.
            </p>
        </div>
    );
}

function Search({ onPick }: { onPick: (p: Place) => void }) {
    const [q, setQ] = useState("");
    const [hits, setHits] = useState<GeoHit[]>([]);
    const [failed, setFailed] = useState(false);
    const go = async (e: FormEvent) => {
        e.preventDefault();
        if (!q.trim()) return;
        setFailed(false);
        try {
            const found = await geocode(q.trim());
            setHits(found);
            setFailed(found.length === 0);
        } catch {
            setHits([]);
            setFailed(true);
        }
    };
    return (
        <div className="search-wrap">
            <form onSubmit={go} className="search" role="search">
                <input
                    value={q}
                    onChange={(e) => setQ(e.target.value)}
                    placeholder="Search a place"
                    aria-label="Search a place"
                />
                <button>Search</button>
            </form>
            {failed && <p className="hits muted small">No place found.</p>}
            {hits.length > 0 && (
                <div className="hits">
                    {hits.map((h) => (
                        <button
                            key={h.name + h.lon}
                            onClick={() => {
                                onPick(h);
                                setHits([]);
                            }}
                        >
                            {h.name}
                        </button>
                    ))}
                </div>
            )}
        </div>
    );
}

const VIEW_LABEL: Record<ViewMode, string> = {
    rgb: "RGB",
    ndvi: "NDVI",
    split: "Both",
};

function useModels() {
    const [models, setModels] = useState<ModelInfo[]>([]);
    useEffect(() => {
        let timer: ReturnType<typeof setTimeout>;
        const poll = () =>
            api.models().then(
                (m) => {
                    setModels(m);
                    if (m.some((x) => x.status === "loading"))
                        timer = setTimeout(poll, 3000);
                },
                () => {
                    timer = setTimeout(poll, 3000);
                },
            );
        poll();
        return () => clearTimeout(timer);
    }, []);
    return models;
}

const statusNote = (m: ModelInfo, hasNir: boolean) =>
    m.status === "error"
        ? " (unavailable)"
        : m.ndvi && !hasNir
          ? " (needs NIR)"
          : m.status === "loading"
            ? " (loading…)"
            : "";

export default function App() {
    const [tiles, setTiles] = useState<Tile[]>([]);
    const [image, setImage] = useState<ViewImage | null>(null);
    const models = useModels();
    const [modelId, setModelId] = useState<ModelId | null>(null);
    const model = models.find((m) => m.id === modelId) ?? models[0];
    const [view, setView] = useState<ViewMode>("rgb");
    const [found, setFound] = useState<Finding[]>([]);
    const [truth, setTruth] = useState<Finding[] | null>(null);
    const [tool, setTool] = useState<Tool>("reader");
    const [box, setBox] = useState<BBox | null>(null);
    const [flyTo, setFlyTo] = useState<FlyTo | null>(null);
    const [layers, setLayers] = useState<Layers>({
        s2: true,
        hls: false,
        modis: false,
    });
    const field = useField(box);

    useEffect(() => {
        api.tiles().then(
            (t) => {
                setTiles(t);
                if (t[0]) pickTile(t[0]);
            },
            () => setTiles([]),
        );
    }, []);
    useEffect(() => {
        setFound([]);
        setTruth(null);
    }, [image?.key, model?.id]);

    const pickTile = (t: Tile) =>
        setImage({
            key: t.id,
            ref: { tile_id: t.id },
            rgb: `/api/tiles/${t.id}/rgb.jpg`,
            ndvi: `/api/tiles/${t.id}/ndvi.png`,
            hasNir: true,
            tile: t,
        });
    const upload = async (e: ChangeEvent<HTMLInputElement>) => {
        const f = e.target.files?.[0];
        if (!f) return;
        const { image_id } = await api.upload(f);
        if (model?.ndvi) setModelId(null);
        setView("rgb");
        setImage({
            key: image_id,
            ref: { image_id },
            rgb: URL.createObjectURL(f),
            ndvi: null,
            hasNir: false,
        });
    };
    const toggleTruth = async () =>
        setTruth(truth || !image ? null : await api.truth(image.key));

    const loading = models.some((m) => m.status === "loading");
    const railNote = !models.length
        ? "Connecting to the backend."
        : loading
          ? "Models loading. The first start downloads the weights."
          : "Models ready.";

    return (
        <div className="shell">
            <nav className="rail" aria-label="Tools">
                <div className="brand">
                    <Mark />
                    <span>Farmland VLM</span>
                </div>
                <button
                    className={"tool-btn" + (tool === "reader" ? " on" : "")}
                    aria-current={tool === "reader" ? "page" : undefined}
                    onClick={() => setTool("reader")}
                >
                    <IconTile />
                    Aerial reader
                </button>
                <button
                    className={"tool-btn" + (tool === "field" ? " on" : "")}
                    aria-current={tool === "field" ? "page" : undefined}
                    onClick={() => setTool("field")}
                >
                    <IconMap />
                    Field map
                    {box && (
                        <span
                            className="dotmark"
                            title="Area selected"
                            role="img"
                            aria-label="Area selected"
                        />
                    )}
                </button>
                <p className="rail-note">{railNote}</p>
            </nav>

            <main className="views">
                <section
                    className="view"
                    hidden={tool !== "reader"}
                    aria-labelledby="h-reader"
                >
                    <header className="vhead">
                        <div className="vtitle">
                            <h1 id="h-reader">Aerial reader</h1>
                            <p className="scope">
                                {image && (
                                    <span className="stamp tone-info">
                                        {image.tile
                                            ? "Sample tile"
                                            : "Your photo"}
                                    </span>
                                )}
                                Reads 10 cm aerial tiles and uploaded photos. It
                                does not read the map.
                            </p>
                        </div>
                        <div className="controls">
                            {image && (
                                <div
                                    className="seg"
                                    role="group"
                                    aria-label="View"
                                >
                                    {(
                                        Object.keys(VIEW_LABEL) as ViewMode[]
                                    ).map((v) => (
                                        <button
                                            key={v}
                                            className={view === v ? "on" : ""}
                                            aria-pressed={view === v}
                                            disabled={
                                                v !== "rgb" && !image.hasNir
                                            }
                                            title={
                                                v !== "rgb" && !image.hasNir
                                                    ? "Needs a near-infrared band. Uploads have none."
                                                    : undefined
                                            }
                                            onClick={() => setView(v)}
                                        >
                                            {VIEW_LABEL[v]}
                                        </button>
                                    ))}
                                </div>
                            )}
                            {image?.tile && (
                                <button
                                    className={"outline" + (truth ? " on" : "")}
                                    aria-pressed={!!truth}
                                    onClick={toggleTruth}
                                >
                                    Ground truth
                                </button>
                            )}
                            {model && (
                                <select
                                    className="modelpick"
                                    aria-label="Model"
                                    value={model.id}
                                    onChange={(e) => setModelId(e.target.value)}
                                >
                                    {models.map((m) => (
                                        <option
                                            key={m.id}
                                            value={m.id}
                                            title={m.error}
                                            disabled={
                                                m.status === "error" ||
                                                (m.ndvi && !image?.hasNir)
                                            }
                                        >
                                            {m.name}
                                            {statusNote(m, !!image?.hasNir)}
                                        </option>
                                    ))}
                                </select>
                            )}
                        </div>
                    </header>
                    <div className="rbody">
                        <div className="stage">
                            {image ? (
                                <TileView
                                    image={image}
                                    mode={view}
                                    findings={found}
                                    truth={truth}
                                />
                            ) : (
                                <label className="drop">
                                    <IconUpload />
                                    <b>Upload an aerial photo</b>
                                    <span>
                                        JPEG or PNG from above. Uploads have no
                                        near-infrared band, so the NDVI model
                                        stays off.
                                    </span>
                                    <input
                                        type="file"
                                        accept="image/*"
                                        onChange={upload}
                                    />
                                </label>
                            )}
                            {(tiles.length > 0 || image) && (
                                <div
                                    className="strip"
                                    aria-label="Sample tiles"
                                >
                                    {tiles.length === 0 && (
                                        <p className="muted small">
                                            Sample tiles come from the
                                            Agriculture-Vision dataset and are
                                            not installed here. Upload your own
                                            aerial photo.
                                        </p>
                                    )}
                                    {tiles.map((t) => (
                                        <button
                                            key={t.id}
                                            title={t.label}
                                            aria-label={t.label}
                                            className={
                                                "thumb" +
                                                (image?.key === t.id
                                                    ? " on"
                                                    : "")
                                            }
                                            onClick={() => pickTile(t)}
                                        >
                                            <img
                                                src={`/api/tiles/${t.id}/rgb.jpg`}
                                                alt=""
                                            />
                                        </button>
                                    ))}
                                    <label
                                        className="upload"
                                        title="Upload a photo"
                                    >
                                        <IconUpload />
                                        <span>Upload</span>
                                        <input
                                            type="file"
                                            accept="image/*"
                                            onChange={upload}
                                        />
                                    </label>
                                </div>
                            )}
                        </div>
                        {model ? (
                            <ChatPanel
                                image={image}
                                model={model}
                                models={models}
                                onFindings={setFound}
                            />
                        ) : (
                            <div className="chat empty">
                                <p>Connecting to the models…</p>
                            </div>
                        )}
                    </div>
                </section>

                <section
                    className="view"
                    hidden={tool !== "field"}
                    aria-labelledby="h-field"
                >
                    <header className="vhead">
                        <div className="vtitle">
                            <h1 id="h-field">Field map</h1>
                            <p className="scope">
                                Satellite context at 10 m for the area you draw.
                                The aerial reader does not use it.
                            </p>
                        </div>
                        <Search
                            onPick={(p) => {
                                setFlyTo({ ...p, t: Date.now() });
                                if (p.box && p.bbox) setBox(p.bbox);
                            }}
                        />
                    </header>
                    <div className="fbody">
                        <FieldMap
                            box={box}
                            onBox={setBox}
                            layers={layers}
                            scene={field.scene}
                            flyTo={flyTo}
                            visible={tool === "field"}
                            className="fmap"
                        />
                        <aside className="fpanel" aria-label="Field context">
                            <h2>Field context</h2>
                            <FieldPanel
                                box={box}
                                field={field}
                                layers={layers}
                                setLayers={setLayers}
                            />
                        </aside>
                    </div>
                </section>
            </main>
        </div>
    );
}
