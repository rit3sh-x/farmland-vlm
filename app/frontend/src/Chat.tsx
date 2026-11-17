import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { Bars, FindingRow, Grid } from "./components";
import { IconSend } from "./icons";
import { CLASS_BY_KEY, suggestions } from "./scheme";
import type {
    Answer as AnswerEvent,
    Finding,
    ModelError,
    ModelInfo,
    Timings,
    ViewImage,
} from "./types";

interface ReadState {
    phase:
        | "idle"
        | "loading"
        | "queued"
        | "encoding"
        | "writing"
        | "done"
        | "error";
    text?: string;
    ahead?: number;
    elapsed?: number;
    findings?: Finding[];
    timings?: Timings;
    cached?: boolean;
    message?: string;
}

function useRead(image: ViewImage | null, model: string) {
    const [s, set] = useState<ReadState>({ phase: "idle" });
    useEffect(() => {
        if (!image) return;
        let live = true;
        const t0 = performance.now();
        const tick = setInterval(
            () =>
                live &&
                set((x) =>
                    x.phase === "done"
                        ? x
                        : { ...x, elapsed: performance.now() - t0 },
                ),
            100,
        );
        set({ phase: "queued", text: "" });
        (async () => {
            let text = "";
            for await (const ev of api.read(image.ref, model)) {
                if (!live) return;
                if (ev.type === "queued")
                    set((x) => ({
                        ...x,
                        phase: ev.ahead ? "queued" : "encoding",
                        ahead: ev.ahead,
                    }));
                else if (ev.type === "phase")
                    set((x) => ({ ...x, phase: ev.phase }));
                else if (ev.type === "token") {
                    text += ev.text;
                    set((x) => ({ ...x, phase: "writing", text }));
                } else if (ev.type === "done" && "caption" in ev)
                    set({
                        phase: "done",
                        text: ev.caption,
                        findings: ev.findings,
                        timings: ev.timings,
                        cached: ev.cached,
                    });
                else if (ev.type === "error")
                    set({ phase: "error", message: ev.message });
            }
        })();
        return () => {
            live = false;
            clearInterval(tick);
        };
    }, [image?.key, model]);
    return s;
}

function Stamp({ hedged, empty }: { hedged: boolean; empty?: boolean }) {
    return empty ? (
        <span className="stamp tone-info">Nothing found</span>
    ) : hedged ? (
        <span className="stamp" title="The model hedged its wording">
            Not sure
        </span>
    ) : (
        <span className="stamp tone-ok" title="The model did not hedge">
            Confident
        </span>
    );
}

function ConfBar({ v }: { v: number }) {
    return (
        <div className="confbar">
            <span className="conf wide">
                <span style={{ width: `${v * 100}%` }} />
            </span>
            <b className="tnum">{Math.round(v * 100)}%</b>
            <small>confidence in the top finding</small>
        </div>
    );
}

function Reading({
    image,
    model: m,
    onFindings,
}: {
    image: ViewImage;
    model: ModelInfo;
    onFindings?: (f: Finding[]) => void;
}) {
    const s = useRead(image, m.id);
    useEffect(() => {
        if (s.phase === "done") onFindings?.(s.findings || []);
    }, [s.phase]);
    const secs = ((s.elapsed || 0) / 1000).toFixed(1);
    const busy =
        s.phase === "loading" ||
        s.phase === "queued" ||
        s.phase === "encoding" ||
        s.phase === "writing";
    const fs = s.findings || [];
    const conf = Math.max(0, ...fs.map((f) => f.conf ?? 0));
    return (
        <section className="entry">
            <div className="who">
                <span>{m.name}</span>
                <small>{m.about}</small>
            </div>
            {busy && (
                <div className="think">
                    <span className="pulse" />
                    {s.phase === "loading"
                        ? "Loading the model (the first run downloads it)"
                        : s.phase === "queued" && s.ahead
                          ? `Waiting for ${s.ahead} other request${s.ahead > 1 ? "s" : ""}`
                          : s.phase === "writing"
                            ? "Writing"
                            : "Looking at the tile"}
                    <span className="secs">{secs}s</span>
                </div>
            )}
            {s.phase === "error" && <p className="error">{s.message}</p>}
            {s.text && (
                <p className="headline">
                    {s.text[0].toUpperCase() + s.text.slice(1)}
                    {s.phase === "writing" && <span className="caret" />}
                    {s.phase === "done" && (
                        <>
                            {" "}
                            <Stamp
                                hedged={fs.some((f) => f.hedge || f.possibly)}
                                empty={!fs.length}
                            />
                        </>
                    )}
                </p>
            )}
            {s.phase === "done" && conf > 0 && <ConfBar v={conf} />}
            {s.phase === "done" && !!fs.length && (
                <div className="frows">
                    {fs.map((f, i) => (
                        <FindingRow key={i} f={f} />
                    ))}
                </div>
            )}
            {s.phase === "done" && !fs.length && (
                <p className="muted small">
                    {m.tuned
                        ? "No anomalies found."
                        : `${m.name} names no anomaly in our vocabulary.`}
                </p>
            )}
            {s.phase === "done" && s.timings && (
                <p className="muted tiny tnum">
                    {(s.timings.total_ms / 1000).toFixed(1)} s
                    {s.cached ? " · cached tile" : ""}
                </p>
            )}
            {(s.phase === "queued" || s.phase === "encoding") && !s.cached && (
                <p className="muted small">
                    The first look takes a few seconds on the laptop. Follow-up
                    questions are quick.
                </p>
            )}
        </section>
    );
}

function Answer({ a }: { a: AnswerEvent }) {
    const cls = a.cls ? CLASS_BY_KEY[a.cls] : null;
    return (
        <div>
            <div className="ans">
                {a.family === "location" && a.answer !== "none" && cls && (
                    <Grid
                        items={[
                            { cls: cls.index, loc: a.answer, hedge: a.hedged },
                        ]}
                        size={30}
                    />
                )}
                {a.family === "extent" && a.answer !== "none" && cls && (
                    <Bars extent={a.answer} color={cls.color} />
                )}
                <b className="headline">{a.answer}</b>
                <Stamp hedged={a.hedged} />
            </div>
            <ConfBar v={a.conf} />
            {a.why && <div className="why">{a.why}</div>}
        </div>
    );
}

interface Turn {
    id: number;
    q: string;
    pending: boolean;
    a?: AnswerEvent | ModelError;
}

interface ChatProps {
    image: ViewImage | null;
    model: ModelInfo;
    models: ModelInfo[];
    onFindings?: (f: Finding[]) => void;
}

export function ChatPanel({ image, model, models, onFindings }: ChatProps) {
    const [extra, setExtra] = useState<string[]>([]);
    const [thread, setThread] = useState<Turn[]>([]);
    const [found, setFound] = useState<Finding[]>([]);
    const [q, setQ] = useState("");
    const box = useRef<HTMLDivElement>(null);
    useEffect(() => {
        setExtra([]);
        setThread([]);
        setFound([]);
    }, [image?.key, model.id]);
    useEffect(() => {
        const el = box.current;
        if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    }, [thread.length, extra.length]);
    if (!image)
        return (
            <div className="chat">
                <div className="chat-scroll">
                    <section className="entry">
                        <p className="headline muted">
                            Waiting for an image{" "}
                            <span className="stamp tone-info">Idle</span>
                        </p>
                        <p className="muted">
                            Choose a sample tile or upload an aerial photo. The
                            model reads it here and says how sure it is.
                        </p>
                    </section>
                </div>
                <div className="chat-compose">
                    <form onSubmit={(e) => e.preventDefault()}>
                        <input
                            disabled
                            placeholder="Ask about the image once one is loaded"
                            aria-label="Ask about the image"
                        />
                        <button className="send" disabled aria-label="Send">
                            <IconSend />
                        </button>
                    </form>
                </div>
            </div>
        );

    const ask = async (question: string) => {
        const id = Math.random();
        setThread((t) => [...t, { id, q: question, pending: true }]);
        for await (const ev of api.ask(image.ref, model.id, question)) {
            if (ev.type === "error" || (ev.type === "done" && "answer" in ev)) {
                setThread((t) =>
                    t.map((x) =>
                        x.id === id ? { ...x, pending: false, a: ev } : x,
                    ),
                );
            }
        }
    };
    const others = models.filter(
        (m) =>
            m.id !== model.id &&
            !extra.includes(m.id) &&
            m.status !== "error" &&
            (!m.ndvi || image.hasNir),
    );
    return (
        <div className="chat">
            <div className="chat-scroll" ref={box}>
                <Reading
                    image={image}
                    model={model}
                    onFindings={(f) => {
                        setFound(f);
                        onFindings?.(f);
                    }}
                />
                {models
                    .filter((m) => extra.includes(m.id))
                    .map((m) => (
                        <Reading key={m.id} image={image} model={m} />
                    ))}
                {others.length > 0 && (
                    <div
                        className="compare"
                        role="group"
                        aria-label="Compare models"
                    >
                        <span>Compare with</span>
                        {others.map((m) => (
                            <button
                                key={m.id}
                                onClick={() => setExtra([...extra, m.id])}
                            >
                                {m.name}
                            </button>
                        ))}
                    </div>
                )}
                {thread.map((t) => (
                    <section key={t.id} className="entry">
                        <p className="asked">{t.q}</p>
                        {t.pending || !t.a ? (
                            <span className="dots">•••</span>
                        ) : t.a.type === "error" ? (
                            <p className="error">{t.a.message}</p>
                        ) : (
                            <Answer a={t.a} />
                        )}
                    </section>
                ))}
                <ul className="sugg" aria-label="Suggested questions">
                    {suggestions(found).map((s) => (
                        <li key={s}>
                            <button onClick={() => ask(s)}>{s}</button>
                        </li>
                    ))}
                </ul>
            </div>
            <div className="chat-compose">
                <form
                    onSubmit={(e) => {
                        e.preventDefault();
                        if (q.trim()) {
                            ask(q.trim());
                            setQ("");
                        }
                    }}
                >
                    <input
                        value={q}
                        onChange={(e) => setQ(e.target.value)}
                        placeholder="Ask about this image"
                        aria-label="Ask about this image"
                    />
                    <button className="send" aria-label="Send">
                        <IconSend />
                    </button>
                </form>
            </div>
        </div>
    );
}
