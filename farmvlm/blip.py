"""BLIP readers for the Baseline and BLIP-LoRA: generated answers, ranked answers and captions, each with a confidence.

Confidence is the geometric mean of the generated tokens' probabilities; for ranked answers it is the
softmax over the candidates' sequence log-likelihoods. Answer and caption sequences start with BLIP's
[DEC] token, as HF's generate does, so scoring, generation and training see the same sequences.
"""
import torch
from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor

VQA_ID = "Salesforce/blip-vqa-base"
CAPTION_ID = "Salesforce/blip-image-captioning-base"


def _geo_mean_prob(scores, sequences, start: int) -> list[float]:
    """Per sequence, exp(mean log p) over generated tokens up to and including the first [SEP]/pad."""
    logp = torch.stack([s.log_softmax(-1) for s in scores], 1)  # [B, T, V]
    toks = sequences[:, start:start + logp.shape[1]]
    lp = logp.gather(-1, toks[..., None])[..., 0]
    out = []
    for row, t in zip(lp, toks):
        stop = (t == 102).nonzero()
        n = int(stop[0]) + 1 if len(stop) else len(t)
        out.append(float(row[:n].mean().exp()))
    return out


class BlipReader:
    """Pass models to wrap them (e.g. a PEFT model during validation); missing ones load from the Hub unless load=False."""

    def __init__(self, vqa=None, captioner=None, device="cuda", dtype=torch.float16, load=True):
        self.device, self.dtype = device, dtype
        self.proc = BlipProcessor.from_pretrained(VQA_ID)
        self.cap_proc = BlipProcessor.from_pretrained(CAPTION_ID)
        if vqa is None and load:
            vqa = BlipForQuestionAnswering.from_pretrained(VQA_ID, dtype=dtype).to(device).eval()
        if captioner is None and load:
            captioner = BlipForConditionalGeneration.from_pretrained(CAPTION_ID, dtype=dtype).to(device).eval()
        self.vqa, self.cap = vqa, captioner
        self.bos = (vqa or captioner).config.text_config.bos_token_id

    def _inputs(self, images, questions):
        x = self.proc(images=images, text=questions, return_tensors="pt", padding=True).to(self.device)
        x["pixel_values"] = x["pixel_values"].to(self.dtype)
        return x

    @torch.no_grad()
    def answer(self, images, questions, max_new_tokens=10):
        x = self._inputs(images, questions)
        out = self.vqa.generate(**x, max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True)
        text = self.proc.batch_decode(out.sequences, skip_special_tokens=True)
        return list(zip([t.strip() for t in text], _geo_mean_prob(out.scores, out.sequences, 1)))

    @torch.no_grad()
    def question_states(self, images, questions):
        x = self._inputs(images, questions)
        img = self.vqa.vision_model(pixel_values=x["pixel_values"])[0]
        q = self.vqa.text_encoder(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                  encoder_hidden_states=img,
                                  encoder_attention_mask=torch.ones(img.shape[:-1], dtype=torch.long, device=self.device))[0]
        return q, x["attention_mask"]

    @torch.no_grad()
    def rank(self, images, questions, candidates: list[list[str]]):
        """Pick each question's most likely answer from its candidate list."""
        q, qmask = self.question_states(images, questions)
        results = []
        for i, cands in enumerate(candidates):
            a = self.proc.tokenizer(cands, return_tensors="pt", padding=True).to(self.device)
            ids = a["input_ids"].clone()
            ids[:, 0] = self.bos
            k = len(cands)
            logits = self.vqa.text_decoder(input_ids=ids, attention_mask=a["attention_mask"],
                                           encoder_hidden_states=q[i:i + 1].expand(k, -1, -1),
                                           encoder_attention_mask=qmask[i:i + 1].expand(k, -1), return_dict=True).logits
            lp = logits[:, :-1].float().log_softmax(-1).gather(-1, ids[:, 1:, None])[..., 0]
            seq = (lp * a["attention_mask"][:, 1:]).sum(1)
            p = seq.softmax(0)
            j = int(p.argmax())
            results.append((cands[j], float(p[j])))
        return results

    @torch.no_grad()
    def caption(self, images, max_new_tokens=40):
        x = self.cap_proc(images=images, return_tensors="pt").to(self.device)
        out = self.cap.generate(pixel_values=x["pixel_values"].to(self.dtype), max_new_tokens=max_new_tokens,
                                output_scores=True, return_dict_in_generate=True)
        text = self.cap_proc.batch_decode(out.sequences, skip_special_tokens=True)
        return list(zip([t.strip() for t in text], _geo_mean_prob(out.scores, out.sequences, 1)))
