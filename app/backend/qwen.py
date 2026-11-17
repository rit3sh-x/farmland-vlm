import asyncio
import base64
import io
import secrets

from PIL import Image

from farmvlm.prompts import CAPTION_PROMPT, NDVI_NOTE

from .models import QWEN_ADAPTERS, adapters, qwen_base

IMAGE_TOKENS = 144


def _data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class Qwen:
    def __init__(self):
        import xllamacpp as xlc
        model, projector = qwen_base()
        root = adapters()
        p = xlc.CommonParams()
        p.model.path, p.mmproj.path = str(model), str(projector)
        p.n_ctx, p.n_parallel, p.image_max_tokens = 4096, 1, IMAGE_TOKENS
        p.lora_adapters = [xlc.CommonAdapterLoraInfo(str(root / m / "lora-f16.gguf"), 0.0) for m in QWEN_ADAPTERS]
        p.lora_init_without_apply = True
        p.api_keys = [secrets.token_urlsafe(32)]
        p.endpoint_slots = False
        self.server = xlc.Server(p)

    async def stream(self, images, question: str | None, model: str, max_tokens: int):
        text = (NDVI_NOTE if len(images) == 2 else "") + (question or CAPTION_PROMPT)
        content = [{"type": "image_url", "image_url": {"url": _data_uri(i)}} for i in images] + [{"type": "text", "text": text}]
        body = {"messages": [{"role": "user", "content": content}], "stream": True, "temperature": 0,
                "max_tokens": max_tokens, "logprobs": True, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "lora": [{"id": i, "scale": float(m == model)} for i, m in enumerate(QWEN_ADAPTERS)]}
        loop = asyncio.get_running_loop()
        chunks: asyncio.Queue = asyncio.Queue()
        closed = False

        def on_chunk(chunk) -> bool:
            loop.call_soon_threadsafe(chunks.put_nowait, chunk)
            return closed

        job = loop.run_in_executor(None, self.server.handle_chat_completions, body, on_chunk)
        job.add_done_callback(lambda _: chunks.put_nowait(None))
        try:
            while (chunk := await chunks.get()) is not None:
                if "error" in chunk:
                    raise RuntimeError(str(chunk["error"]))
                if "timings" in chunk:
                    yield ("done", chunk["timings"])
                for choice in chunk.get("choices", []):
                    for lp in (choice.get("logprobs") or {}).get("content") or []:
                        yield ("token", lp["token"], lp["logprob"])
        finally:
            closed = True
            await job
