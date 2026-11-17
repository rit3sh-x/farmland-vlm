// Shapes of the FastAPI backend's responses and NDJSON events.

export type ModelId = string;

export interface ModelInfo {
    id: ModelId;
    name: string;
    about: string;
    tuned: boolean;
    ndvi: boolean;
    status: "loading" | "ready" | "error";
    error?: string;
}
export type BBox = [number, number, number, number];

export interface Tile {
    id: string;
    label: string;
    year: number;
    has_nir: boolean;
}
export interface ImageRef {
    tile_id?: string;
    image_id?: string;
}

export interface Finding {
    cls: number;
    extent: string | null;
    loc: string | null;
    hedge: boolean;
    possibly?: boolean;
    conf?: number;
}

export interface Timings {
    queue_ms: number;
    encode_ms: number | null;
    total_ms: number;
}

export type ModelEvent =
    | { type: "queued"; ahead: number }
    | { type: "phase"; phase: "loading" | "encoding" | "writing" }
    | { type: "token"; text: string }
    | { type: "error"; code: string | number; message: string }
    | Reading
    | Answer;

interface Done {
    type: "done";
    timings: Timings;
    cached: boolean;
}
export interface Reading extends Done {
    caption: string;
    findings: Finding[];
}
export interface Answer extends Done {
    answer: string;
    hedged: boolean;
    why: string | null;
    conf: number;
    family: "presence" | "identify" | "extent" | "location" | string;
    cls: string | null;
}
export type ModelError = Extract<ModelEvent, { type: "error" }>;

export interface Scene {
    s2_png: string;
    bounds: BBox;
    date: string;
    cloud_pct: number;
    hls_date: string | null;
}
export interface Season {
    series: { date: string; ndvi: number }[];
    peak: { date: string; ndvi: number };
}
export interface WeatherDay {
    date: string;
    rain_mm: number;
    soil_wetness: number | null;
}
export interface Weather {
    daily: WeatherDay[];
    rain_mm: number;
    rain_vs_normal: number;
}

export type FieldEvent =
    | ({ type: "scene" } & Scene)
    | ({ type: "season" } & Season)
    | ({ type: "weather" } & Weather)
    | { type: "summary"; text: string }
    | { type: "unavailable"; part: string; reason: string }
    | { type: "done"; cached?: boolean }
    | { type: "error"; code: string | number; message: string };

export interface FieldState {
    loading?: boolean;
    error?: string;
    scene?: Scene;
    season?: Season;
    weather?: Weather;
    summary?: { text: string };
    unavailable?: { part: string; reason: string }[];
}

export interface ViewImage {
    key: string;
    ref: ImageRef;
    rgb: string;
    ndvi: string | null;
    hasNir: boolean;
    tile?: Tile;
}

export interface Place {
    name?: string;
    lon?: number;
    lat?: number;
    bbox?: BBox;
    box?: boolean;
}
