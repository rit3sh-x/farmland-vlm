// Client for the FastAPI backend. Model calls stream NDJSON, one event per line.
import type {
    BBox,
    FieldEvent,
    Finding,
    ImageRef,
    ModelEvent,
    ModelId,
    ModelInfo,
    Tile,
} from "./types";

type ErrorEvent = { type: "error"; code: string | number; message: string };

async function* ndjson<E>(res: Response): AsyncGenerator<E | ErrorEvent> {
    if (!res.ok || !res.body) {
        const text = await res.text();
        let detail: { code?: string; message?: string } | string = text;
        try {
            detail = JSON.parse(text).detail ?? text;
        } catch {
            /* plain text body */
        }
        yield typeof detail === "string"
            ? { type: "error", code: res.status, message: detail }
            : {
                  type: "error",
                  code: detail.code || res.status,
                  message: detail.message || text,
              };
        return;
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        let i;
        while ((i = buf.indexOf("\n")) >= 0) {
            const line = buf.slice(0, i).trim();
            buf = buf.slice(i + 1);
            if (line) yield JSON.parse(line) as E;
        }
    }
}

const post = (path: string, body: unknown) =>
    fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
    });
async function json<T>(path: string): Promise<T> {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
    return r.json() as Promise<T>;
}

export const api = {
    models: () => json<ModelInfo[]>("/api/models"),
    tiles: () => json<Tile[]>("/api/tiles"),
    truth: (id: string) => json<Finding[]>(`/api/tiles/${id}/truth`),
    upload: async (
        file: File,
    ): Promise<{ image_id: string; has_nir: boolean }> => {
        const fd = new FormData();
        fd.append("file", file);
        const r = await fetch("/api/images", { method: "POST", body: fd });
        if (!r.ok) throw new Error("upload failed");
        return r.json();
    },
    read: async function* (image: ImageRef, model: ModelId) {
        yield* ndjson<ModelEvent>(await post("/api/read", { image, model }));
    },
    ask: async function* (image: ImageRef, model: ModelId, question: string) {
        yield* ndjson<ModelEvent>(
            await post("/api/ask", { image, model, question }),
        );
    },
    field: async function* (bbox: BBox) {
        yield* ndjson<FieldEvent>(await post("/api/field", { bbox }));
    },
};

export interface GeoHit {
    name: string;
    lon: number;
    lat: number;
    extent?: number[];
}

interface PhotonFeature {
    properties: {
        name?: string;
        state?: string;
        country?: string;
        extent?: number[];
    };
    geometry: { coordinates: [number, number] };
}

export async function geocode(q: string): Promise<GeoHit[]> {
    const r = await fetch(
        `https://photon.komoot.io/api/?q=${encodeURIComponent(q)}&limit=5`,
    );
    const j = (await r.json()) as { features: PhotonFeature[] };
    return j.features.map((f) => ({
        name: [f.properties.name, f.properties.state, f.properties.country]
            .filter(Boolean)
            .join(", "),
        lon: f.geometry.coordinates[0],
        lat: f.geometry.coordinates[1],
        extent: f.properties.extent,
    }));
}
