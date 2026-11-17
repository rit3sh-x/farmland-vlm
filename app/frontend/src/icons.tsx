import type { SVGProps } from "react";

const base: SVGProps<SVGSVGElement> = {
    width: 20,
    height: 20,
    viewBox: "0 0 20 20",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.75,
    strokeLinecap: "round",
    strokeLinejoin: "round",
    "aria-hidden": true,
};

export const Mark = () => (
    <svg {...base} stroke="none" width={26} height={26} viewBox="0 0 26 26">
        {[0, 1, 2].flatMap((r) =>
            [0, 1, 2].map((c) => (
                <rect
                    key={`${r}${c}`}
                    x={1 + c * 8.4}
                    y={1 + r * 8.4}
                    width="7"
                    height="7"
                    rx="1.2"
                    fill={r === 0 && c === 0 ? "#f5b700" : "#f3f9ee"}
                />
            )),
        )}
    </svg>
);

export const IconTile = () => (
    <svg {...base}>
        <rect x="2.5" y="2.5" width="15" height="15" rx="2" />
        <path d="M7.5 2.5v15M12.5 2.5v15M2.5 7.5h15M2.5 12.5h15" />
    </svg>
);

export const IconMap = () => (
    <svg {...base}>
        <path d="M2.5 5.5 7 3.5l6 2 4.5-2v11l-4.5 2-6-2-4.5 2z" />
        <path d="M7 3.5v11M13 5.5v11" />
    </svg>
);

export const IconSelect = () => (
    <svg {...base}>
        <path d="M3 6.5V3h3.5M13.5 3H17v3.5M17 13.5V17h-3.5M6.5 17H3v-3.5" />
    </svg>
);

export const IconSend = () => (
    <svg {...base}>
        <path d="M10 16V4M5 9l5-5 5 5" />
    </svg>
);

export const IconUpload = () => (
    <svg {...base}>
        <path d="M10 13V3.5M6 7l4-4 4 4M3.5 13.5v2a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-2" />
    </svg>
);
