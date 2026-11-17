import asyncio
import math
import time
from contextlib import aclosing, asynccontextmanager

from farmvlm.text import mention_spans, normalise, why

from .models import MODELS


class ModelQueue:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._waiting = 0

    def ahead(self) -> int:
        return self._waiting + int(self._lock.locked())

    @asynccontextmanager
    async def turn(self):
        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1
        try:
            yield
        finally:
            self._lock.release()


def findings(caption: str, tokens: list[tuple[str, float]] | None, fallback_conf: float):
    starts, pos = [], 0
    for text, _ in tokens or []:
        starts.append(pos)
        pos += len(text)
    rows = []
    for m, a, b in mention_spans(caption):
        lps = [lp for (text, lp), s in zip(tokens or [], starts) if s < b and s + len(text) > a]
        conf = math.exp(sum(lps) / len(lps)) if lps else fallback_conf
        rows.append({"cls": m.cls, "extent": m.extent, "loc": m.loc, "hedge": m.tiny or bool(m.extent and " to " in m.extent)
                     or bool(m.loc and " or " in m.loc), "possibly": m.tiny, "conf": round(conf, 3)})
    return rows


class Engine:
    def __init__(self):
        self.queue = ModelQueue()
        self.engines: dict[str, asyncio.Task] = {}

    def start(self):
        from .blip import Blip
        from .qwen import Qwen
        self.engines = {"blip": asyncio.create_task(asyncio.to_thread(Blip)),
                        "qwen": asyncio.create_task(asyncio.to_thread(Qwen))}

    def status(self, model_id: str) -> dict:
        task = self.engines[MODELS[model_id].engine]
        if not task.done():
            return {"status": "loading"}
        if task.exception():
            return {"status": "error", "error": str(task.exception())}
        return {"status": "ready"}

    async def read(self, images, model_id: str):
        """Caption and findings: queued -> phase... -> token... -> done."""
        async for ev in self._call(images, None, model_id, 60):
            yield ev

    async def ask(self, images, model_id: str, question: str):
        async for ev in self._call(images, question, model_id, 20):
            yield ev

    async def _call(self, images, question, model_id, max_tokens):
        model = MODELS[model_id]
        loading = self.engines[model.engine]
        if not loading.done():
            yield {"type": "phase", "phase": "loading"}
        engine = await loading
        yield {"type": "queued", "ahead": self.queue.ahead()}
        t_queue = time.time()
        async with self.queue.turn():
            queue_ms = int((time.time() - t_queue) * 1000)
            yield {"type": "phase", "phase": "encoding"}
            t0 = time.time()
            if model.engine == "blip":
                text, conf = await engine.run(images[0], question, model.tuned)
                tokens, encode_ms, cached = None, None, False
                yield {"type": "token", "text": text}
            else:
                tokens, timings, first = [], {}, None
                async with aclosing(engine.stream(images, question, model_id, max_tokens)) as stream:
                    async for ev in stream:
                        if ev[0] == "done":
                            timings = ev[1]
                            continue
                        if first is None:
                            first = time.time()
                            yield {"type": "phase", "phase": "writing"}
                        tokens.append((ev[1], ev[2]))
                        yield {"type": "token", "text": ev[1]}
                text = "".join(t for t, _ in tokens).replace("<think>", "").replace("</think>", "").strip()
                conf = math.exp(sum(lp for _, lp in tokens) / len(tokens)) if tokens else 0.0
                encode_ms = int(timings.get("prompt_ms", ((first or time.time()) - t0) * 1000))
                cached = timings.get("prompt_n", 99) < 20
            total_ms = int((time.time() - t0) * 1000)
        done = {"type": "done", "timings": {"queue_ms": queue_ms, "encode_ms": encode_ms, "total_ms": total_ms}, "cached": cached}
        if question is None:
            done.update(caption=text, findings=findings(text, tokens, conf))
        else:
            done.update(answer=normalise(text), hedged=why(text) is not None, why=why(text), conf=round(conf, 3))
        yield done
