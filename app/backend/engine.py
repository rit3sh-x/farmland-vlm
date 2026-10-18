"""The models behind the demo: one global queue in front of llama-server (served model + its adapters)
and zero-shot BLIP on CPU. Every call yields the NDJSON events of issues/14-api-contract.md.
"""
import asyncio
import base64
import io
import json
import math
import subprocess
import time
from contextlib import asynccontextmanager

import httpx
from PIL import Image

from farmvlm.text import mention_spans, normalise, why
from farmvlm.prompts import CAPTION_PROMPT, NDVI_NOTE

from . import config

MODEL_IDS = ("ft", "ndvi", "blip")


class ModelQueue:
    """Exactly one model call at a time; callers learn how many are ahead of them."""

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


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _findings(caption: str, tokens: list[tuple[str, float]] | None, fallback_conf: float):
    """Finding rows parsed from the caption; confidence = geometric mean of the probabilities of the tokens they span."""
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


class LlamaServer:
    """The served model: llama-server with the RGB and NDVI-variant LoRAs loaded at scale 0, chosen per request."""

    def __init__(self):
        self.proc = None
        self.adapters = [k for k in ("ft", "ndvi") if config.ADAPTERS[k].exists()]
        self.url = f"http://127.0.0.1:{config.LLAMA_PORT}"

    def start(self):
        cmd = [str(config.LLAMA_SERVER), "-m", str(config.SERVED_GGUF), "--mmproj", str(config.SERVED_MMPROJ),
               "--port", str(config.LLAMA_PORT), "--host", "127.0.0.1", "-t", str(config.LLAMA_THREADS),
               "-np", "1", "-c", "4096", "--image-max-tokens", "144", "--no-webui"]
        for k in self.adapters:
            cmd += ["--lora", str(config.ADAPTERS[k])]
        if self.adapters:
            cmd.append("--lora-init-without-apply")
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    async def ready(self, timeout=120) -> bool:
        t0 = time.time()
        async with httpx.AsyncClient() as c:
            while time.time() - t0 < timeout:
                try:
                    if (await c.get(self.url + "/health", timeout=2)).status_code == 200:
                        return True
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(0.5)
        return False

    def stop(self):
        if self.proc:
            self.proc.terminate()

    async def stream(self, images, question: str | None, model: str, max_tokens: int):
        """Yield ("token", text, logprob) per generated token, then ("done", timings)."""
        text = (NDVI_NOTE if len(images) == 2 else "") + (question or CAPTION_PROMPT)
        content = [{"type": "image_url", "image_url": {"url": _data_uri(i)}} for i in images] + [{"type": "text", "text": text}]
        body = {"messages": [{"role": "user", "content": content}], "stream": True, "temperature": 0,
                "max_tokens": max_tokens, "logprobs": True, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False}}
        if self.adapters:
            body["lora"] = [{"id": i, "scale": 1.0 if k == model else 0.0} for i, k in enumerate(self.adapters)]
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", self.url + "/v1/chat/completions", json=body) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if "timings" in chunk:
                        yield ("done", chunk["timings"])
                    for ch in chunk.get("choices", []):
                        for lp in ((ch.get("logprobs") or {}).get("content") or []):
                            yield ("token", lp["token"], lp["logprob"])


class BlipCpu:
    """Zero-shot BLIP base (or BLIP-LoRA in the fallback demo), loaded on first use, run in a worker thread."""

    def __init__(self):
        self.reader = None

    def _load(self):
        import torch
        from farmvlm.blip import BlipReader
        vqa = cap = None
        if config.BLIP_VQA_ADAPTER or config.BLIP_CAPTION_ADAPTER:
            from peft import PeftModel
            from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering
            from farmvlm.blip import CAPTION_ID, VQA_ID
            vqa = BlipForQuestionAnswering.from_pretrained(VQA_ID)
            cap = BlipForConditionalGeneration.from_pretrained(CAPTION_ID)
            if config.BLIP_VQA_ADAPTER:
                vqa = PeftModel.from_pretrained(vqa, config.BLIP_VQA_ADAPTER).merge_and_unload()
            if config.BLIP_CAPTION_ADAPTER:
                cap = PeftModel.from_pretrained(cap, config.BLIP_CAPTION_ADAPTER).merge_and_unload()
            vqa, cap = vqa.eval(), cap.eval()
        self.reader = BlipReader(vqa=vqa, captioner=cap, device="cpu", dtype=torch.float32)

    async def run(self, image, question):
        def work():
            if self.reader is None:
                self._load()
            return (self.reader.answer([image], [question]) if question else self.reader.caption([image]))[0]
        return await asyncio.to_thread(work)


class Engine:
    def __init__(self):
        self.queue = ModelQueue()
        self.llama = LlamaServer()
        self.blip = BlipCpu()

    async def read(self, images, model: str):
        """Caption + findings: queued -> phase -> token... -> done."""
        async for ev in self._call(images, None, model, 60):
            yield ev

    async def ask(self, images, model: str, question: str):
        async for ev in self._call(images, question, model, 20):
            yield ev

    async def _call(self, images, question, model, max_tokens):
        yield {"type": "queued", "ahead": self.queue.ahead()}
        t_queue = time.time()
        async with self.queue.turn():
            queue_ms = int((time.time() - t_queue) * 1000)
            yield {"type": "phase", "phase": "encoding"}
            t0 = time.time()
            if model == "blip":
                text, conf = await self.blip.run(images[0], question)
                tokens, encode_ms, cached = None, None, False
                yield {"type": "token", "text": text}
            else:
                parts, timings, first = [], {}, None
                async for ev in self.llama.stream(images, question, model, max_tokens):
                    if ev[0] == "done":
                        timings = ev[1]
                        continue
                    if first is None:
                        first = time.time()
                        yield {"type": "phase", "phase": "writing"}
                    parts.append((ev[1], ev[2]))
                    yield {"type": "token", "text": ev[1]}
                text = "".join(p for p, _ in parts).replace("<think>", "").replace("</think>", "").strip()
                tokens = parts
                conf = math.exp(sum(lp for _, lp in parts) / len(parts)) if parts else 0.0
                encode_ms = int(timings.get("prompt_ms", ((first or time.time()) - t0) * 1000))
                cached = timings.get("prompt_n", 99) < 20
            total_ms = int((time.time() - t0) * 1000)
        done = {"type": "done", "timings": {"queue_ms": queue_ms, "encode_ms": encode_ms, "total_ms": total_ms}, "cached": cached}
        if question is None:
            done.update(caption=text, findings=_findings(text, tokens, conf))
        else:
            hedged = why(text) is not None
            done.update(answer=normalise(text), hedged=hedged, why=why(text), conf=round(conf, 3))
        yield done
