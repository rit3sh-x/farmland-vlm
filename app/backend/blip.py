import asyncio
import json
from contextlib import ExitStack

from farmvlm.metrics import prior_pick
from farmvlm.text import question_family

from .models import adapters


class Blip:
    def __init__(self):
        import torch
        from peft import PeftModel
        from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering
        from farmvlm.blip import CAPTION_ID, VQA_ID, BlipReader
        root = adapters() / "blip-lora"
        vqa = PeftModel.from_pretrained(BlipForQuestionAnswering.from_pretrained(VQA_ID), str(root / "vqa")).eval()
        cap = PeftModel.from_pretrained(BlipForConditionalGeneration.from_pretrained(CAPTION_ID), str(root / "caption")).eval()
        self.reader = BlipReader(vqa=vqa, captioner=cap, device="cpu", dtype=torch.float32)
        self.readout = json.loads((root / "readout.json").read_text())

    def _ranked(self, image, question: str) -> tuple[str, float]:
        family, _ = question_family(question)
        a, p, a2, p2 = self.reader.rank([image], [question], [self.readout["candidates"][family]])[0]
        row = {"family": family, "ranked": a, "ranked_conf": p, "ranked2": a2, "ranked2_conf": p2}
        pick = prior_pick(row, self.readout["ratios"], self.readout["weights"][family])
        return pick, p if pick == a else p2

    def _run(self, image, question: str | None, tuned: bool) -> tuple[str, float]:
        with ExitStack() as off:
            if not tuned:
                off.enter_context(self.reader.vqa.disable_adapter())
                off.enter_context(self.reader.cap.disable_adapter())
            if question is None:
                return self.reader.caption([image])[0]
            return self._ranked(image, question) if tuned else self.reader.answer([image], [question])[0]

    async def run(self, image, question: str | None, tuned: bool) -> tuple[str, float]:
        return await asyncio.to_thread(self._run, image, question, tuned)
