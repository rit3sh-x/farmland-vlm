"""Where BLIP looked: gradient-weighted cross-attention from the question to the image patches.

For "Is there <class> in this tile?", take the last text-encoder layer's cross-attention, weight it by
the gradient of log p(answer), keep the positive part, average over heads and question tokens. The
result is a 24x24 map over the 384 px image (16 px patches), scored against the class mask.
"""
import numpy as np
import torch
from PIL import Image

from farmvlm.blip import VQA_ID


class BlipAttention:
    def __init__(self, model=None, device="cuda"):
        from transformers import BlipForQuestionAnswering, BlipProcessor
        self.device = device
        self.proc = BlipProcessor.from_pretrained(VQA_ID)
        self.model = (model or BlipForQuestionAnswering.from_pretrained(VQA_ID, attn_implementation="eager")).float().to(device).eval()
        self.model.requires_grad_(True)  # PEFT's merge_and_unload leaves the weights frozen: no graph to take the attention's gradient
        self.bos = self.model.config.text_config.bos_token_id

    def heat(self, image: Image.Image, question: str, answer: str = "yes") -> np.ndarray:
        x = self.proc(images=image, text=question, return_tensors="pt").to(self.device)
        img = self.model.vision_model(pixel_values=x["pixel_values"])[0]
        out = self.model.text_encoder(input_ids=x["input_ids"], attention_mask=x["attention_mask"], encoder_hidden_states=img,
                                      encoder_attention_mask=torch.ones(img.shape[:-1], dtype=torch.long, device=self.device),
                                      output_attentions=True, return_dict=True)
        cross = out.cross_attentions[-1]  # [1, heads, q, 577]
        cross.retain_grad()
        a = self.proc.tokenizer([answer], return_tensors="pt").to(self.device)
        ids = a["input_ids"].clone()
        ids[:, 0] = self.bos
        logits = self.model.text_decoder(input_ids=ids, attention_mask=a["attention_mask"], encoder_hidden_states=out.last_hidden_state,
                                         encoder_attention_mask=x["attention_mask"], return_dict=True).logits
        logp = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None]).sum()
        self.model.zero_grad(set_to_none=True)
        logp.backward()
        h = (cross.grad * cross).clamp(min=0).mean(1)[0]  # [q, 577]
        h = h[:, 1:].mean(0)  # drop the image CLS token
        side = int(round(h.numel() ** 0.5))
        return h.reshape(side, side).detach().float().cpu().numpy()


def mask_grid(mask: np.ndarray, side: int) -> np.ndarray:
    """Share of each patch covered by the boolean mask, at side x side."""
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize((side, side), Image.BOX)
    return np.asarray(m, np.float32) / 255


def pointing(heat: np.ndarray, cover: np.ndarray) -> dict:
    """Pointing game (does the hottest patch touch the mask?) and the share of heat inside the mask."""
    peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
    total = float(heat.sum()) or 1.0
    return {"hit": bool(cover[peak] > 0), "energy": float((heat * (cover > 0)).sum() / total),
            "chance": float((cover > 0).mean())}
