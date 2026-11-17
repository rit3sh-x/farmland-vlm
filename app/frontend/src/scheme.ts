// Mirror of farmvlm/scheme.py's class order plus display colours.
import type { Finding } from "./types";

export interface AnomalyClass {
    key: string;
    name: string;
    color: string;
}

export const CLASSES: AnomalyClass[] = [
    { key: "double_plant", name: "double plant", color: "#3aa58a" },
    { key: "drydown", name: "drydown", color: "#c8893a" },
    { key: "endrow", name: "endrow", color: "#d1567a" },
    {
        key: "nutrient_deficiency",
        name: "nutrient deficiency",
        color: "#c9b21f",
    },
    { key: "planter_skip", name: "planter skip", color: "#8a6d4f" },
    { key: "storm_damage", name: "storm damage", color: "#555b66" },
    { key: "water", name: "standing water", color: "#2f7fd1" },
    { key: "waterway", name: "waterway", color: "#3bb3c9" },
    { key: "weed_cluster", name: "weed cluster", color: "#7a5bd6" },
];
export const CLASS_BY_KEY: Record<string, AnomalyClass & { index: number }> =
    Object.fromEntries(CLASSES.map((c, i) => [c.key, { ...c, index: i }]));
export const CELLS = [
    "top left",
    "top",
    "top right",
    "left",
    "centre",
    "right",
    "bottom left",
    "bottom",
    "bottom right",
];

type RGB = [number, number, number];
export const NDVI_RAMP: RGB[] = [
    [165, 0, 38],
    [244, 109, 67],
    [254, 224, 139],
    [166, 217, 106],
    [26, 152, 80],
];
export function ndviColor(v: number, lo = -0.2, hi = 0.9): RGB {
    const x =
        Math.max(0, Math.min(1, (v - lo) / (hi - lo))) * (NDVI_RAMP.length - 1);
    const i = Math.min(Math.floor(x), NDVI_RAMP.length - 2);
    const t = x - i;
    return NDVI_RAMP[i].map((c, k) =>
        Math.round(c + (NDVI_RAMP[i + 1][k] - c) * t),
    ) as RGB;
}

export function suggestions(findings: Finding[] | undefined): string[] {
    const out = ["What anomalies are visible?"];
    for (const f of (findings || []).slice(0, 2)) {
        const n = CLASSES[f.cls].name;
        out.push(
            `Where is the ${n}?`,
            `How much of the tile does the ${n} cover?`,
        );
    }
    const absent = CLASSES.find(
        (c, i) =>
            !(findings || []).some((f) => f.cls === i) &&
            ["weed_cluster", "drydown", "water"].includes(c.key),
    );
    if (absent) out.push(`Is there ${absent.name} in this tile?`);
    return out;
}
