import torch
from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor

from farmvlm.images import SIZE

BASES = {"base": ("Salesforce/blip-vqa-base", "Salesforce/blip-image-captioning-base"),
         "large": (None, "Salesforce/blip-image-captioning-large")}
VQA_ID, CAPTION_ID = BASES["base"]


def _geo_mean_prob(scores, sequences, start: int) -> list[float]:
    logp = torch.stack([s.log_softmax(-1) for s in scores], 1)
    toks = sequences[:, start:start + logp.shape[1]]
    lp = logp.gather(-1, toks[..., None])[..., 0]
    out = []
    for row, t in zip(lp, toks):
        stop = (t == 102).nonzero()
        n = int(stop[0]) + 1 if len(stop) else len(t)
        out.append(float(row[:n].mean().exp()))
    return out


def processor(model_id: str, size: int = SIZE) -> BlipProcessor:
    proc = BlipProcessor.from_pretrained(model_id)
    proc.image_processor.size = {"height": size, "width": size}
    return proc


class BlipReader:
    def __init__(self, vqa=None, captioner=None, device="cuda", dtype=torch.float16, load=True, size=SIZE, beams=1, base="base"):
        self.device, self.dtype, self.beams = device, dtype, beams
        vqa_id, cap_id = BASES[base]
        self.proc = processor(vqa_id or cap_id, size)
        self.cap_proc = processor(cap_id, size)
        self.hi_res = size != SIZE
        if vqa is None and load:
            vqa = BlipForQuestionAnswering.from_pretrained(vqa_id, dtype=dtype).to(device).eval()
        if captioner is None and load:
            captioner = BlipForConditionalGeneration.from_pretrained(cap_id, dtype=dtype).to(device).eval()
        self.vqa, self.cap = vqa, captioner
        self.bos = (vqa or captioner).config.text_config.bos_token_id

    def _inputs(self, images, questions):
        x = self.proc(images=images, text=questions, return_tensors="pt", padding=True).to(self.device)
        x["pixel_values"] = x["pixel_values"].to(self.dtype)
        return x

    @torch.no_grad()
    def answer(self, images, questions, max_new_tokens=10):
        x = self._inputs(images, questions)
        out = self.vqa.generate(**x, max_new_tokens=max_new_tokens, output_scores=True, return_dict_in_generate=True,
                                interpolate_pos_encoding=self.hi_res, num_beams=self.beams)
        text = self.proc.batch_decode(out.sequences, skip_special_tokens=True)
        return list(zip([t.strip() for t in text], self._conf(out)))

    def _conf(self, out):
        if self.beams > 1:
            return out.sequences_scores.float().exp().tolist()
        return _geo_mean_prob(out.scores, out.sequences, 1)

    @torch.no_grad()
    def question_states(self, images, questions):
        x = self._inputs(images, questions)
        img = self.vqa.vision_model(pixel_values=x["pixel_values"], interpolate_pos_encoding=self.hi_res)[0]
        q = self.vqa.text_encoder(input_ids=x["input_ids"], attention_mask=x["attention_mask"],
                                  encoder_hidden_states=img,
                                  encoder_attention_mask=torch.ones(img.shape[:-1], dtype=torch.long, device=self.device))[0]
        return q, x["attention_mask"]

    def rank(self, images, questions, candidates: list[list[str]]):
        results = []
        for cands, lp in zip(candidates, self.scores(images, questions, candidates)):
            p = lp.exp()
            j, j2 = (int(x) for x in p.topk(2).indices)
            results.append((cands[j], float(p[j]), cands[j2], float(p[j2])))
        return results

    @torch.no_grad()
    def scores(self, images, questions, candidates: list[list[str]]) -> list[torch.Tensor]:
        q, qmask = self.question_states(images, questions)
        out = []
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
            out.append(seq.log_softmax(0))
        return out

    @torch.no_grad()
    def caption(self, images, max_new_tokens=40):
        x = self.cap_proc(images=images, return_tensors="pt").to(self.device)
        out = self.cap.generate(pixel_values=x["pixel_values"].to(self.dtype), max_new_tokens=max_new_tokens,
                                output_scores=True, return_dict_in_generate=True, interpolate_pos_encoding=self.hi_res,
                                num_beams=self.beams)
        text = self.cap_proc.batch_decode(out.sequences, skip_special_tokens=True)
        return list(zip([t.strip() for t in text], self._conf(out)))


class Ensemble:
    def __init__(self, readers):
        self.readers, self.first = readers, readers[0]

    def answer(self, *a, **k):
        return self.first.answer(*a, **k)

    def caption(self, *a, **k):
        return self.first.caption(*a, **k)

    def scores(self, images, questions, candidates):
        per = [r.scores(images, questions, candidates) for r in self.readers]
        return [torch.stack(x).mean(0).log_softmax(0) for x in zip(*per)]

    rank = BlipReader.rank
