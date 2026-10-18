"""The served-model candidates through one chat-template path: prompts, training batches, generation.

Every candidate reads 384 px images (~144 image tokens). The NDVI variant gets two images per prompt,
the RGB tile and its NDVI map, with a sentence saying which is which.
"""
import torch

MODELS = {
    "lfm-450m": "LiquidAI/LFM2.5-VL-450M",
    "lfm-1.6b": "LiquidAI/LFM2.5-VL-1.6B",
    "qwen35-2b": "Qwen/Qwen3.5-2B",
}
# Laptop cold caption + answer time order from the feasibility check, fastest first (bake-off tiebreak).
SPEED_ORDER = ("lfm-450m", "lfm-1.6b", "qwen35-2b")
MICRO_BATCH = {"lfm-450m": 16, "lfm-1.6b": 4, "qwen35-2b": 2}
CAPTION_PROMPT = "Describe the anomalies in this tile."
NDVI_NOTE = "The second image is the same tile's NDVI map: red is stressed or bare ground, green is healthy crop. "


def messages(question: str, n_images: int, answer: str | None = None):
    text = (NDVI_NOTE if n_images == 2 else "") + (question or CAPTION_PROMPT)
    m = [{"role": "user", "content": [{"type": "image"}] * n_images + [{"type": "text", "text": text}]}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m


def _template(proc, m, generation):
    return proc.apply_chat_template(m, add_generation_prompt=generation, tokenize=False, enable_thinking=False)


def as_list(img):
    return list(img) if isinstance(img, tuple) else [img]


def train_batch(proc, images, questions, answers):
    """Tokenised batch with loss on the assistant answer only."""
    tok = proc.tokenizer
    tok.padding_side = "right"
    full, pre = [], []
    for img, q, a in zip(images, questions, answers):
        n = len(as_list(img))
        pre.append(_template(proc, messages(q, n), True))
        full.append(_template(proc, messages(q, n, a), False))
    enc = proc(text=full, images=[as_list(i) for i in images], padding=True, return_tensors="pt")
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    for i, (f, p) in enumerate(zip(full, pre)):
        n_answer = len(tok(f[len(p):], add_special_tokens=False).input_ids)
        labels[i, : int(enc["attention_mask"][i].sum()) - n_answer] = -100
    enc["labels"] = labels
    return enc


def load(key, dtype=torch.float16, attn="sdpa"):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(MODELS[key])
    model = AutoModelForImageTextToText.from_pretrained(MODELS[key], dtype=dtype, device_map="cuda", attn_implementation=attn)
    return model, proc


def lora_targets(model, vision: bool) -> str:
    """Regex over every Linear leaf of the language model (plus the vision tower when vision=True)."""
    scopes = ["language_model"] + (["vision", "visual"] if vision else [])
    leaves = sorted({n.split(".")[-1] for n, m in model.named_modules()
                     if isinstance(m, torch.nn.Linear) and any(s in n for s in scopes) and "lm_head" not in n})
    return rf".*({'|'.join(scopes)}).*\.({'|'.join(leaves)})$"


class VlmReader:
    def __init__(self, model, proc, max_new_tokens=40):
        self.model, self.proc, self.max_new = model, proc, max_new_tokens

    @torch.no_grad()
    def _generate(self, images, questions):
        self.proc.tokenizer.padding_side = "left"
        texts = [_template(self.proc, messages(q, len(as_list(i))), True) for i, q in zip(images, questions)]
        enc = self.proc(text=texts, images=[as_list(i) for i in images], padding=True, return_tensors="pt").to("cuda")
        out = self.model.generate(**enc, max_new_tokens=self.max_new, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
        new = out.sequences[:, enc["input_ids"].shape[1]:]
        logp = torch.stack([s.float().log_softmax(-1) for s in out.scores], 1).gather(-1, new[..., None])[..., 0]
        eos = {self.proc.tokenizer.eos_token_id, self.proc.tokenizer.pad_token_id}
        res = []
        for toks, lp in zip(new, logp):
            n = next((k for k, tk in enumerate(toks.tolist()) if tk in eos), len(toks))
            text = self.proc.tokenizer.decode(toks[:n], skip_special_tokens=True)
            text = text.replace("<think>", "").replace("</think>", "").strip()
            res.append((text, float(lp[:max(n, 1)].mean().exp())))
        return res

    def answer(self, images, questions):
        return self._generate(images, questions)

    def caption(self, images):
        return self._generate(images, [CAPTION_PROMPT] * len(images))
