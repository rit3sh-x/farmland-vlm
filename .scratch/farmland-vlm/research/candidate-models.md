# Candidate models for the served model

Research for ticket [02-candidate-models](../issues/02-candidate-models.md). As of 2026-10-18. Bracketed numbers like [3] point to Sources at the end. **UNVERIFIED** marks anything not confirmed in a primary source. **est.** marks my own estimates, which a laptop spike should measure (see Open questions).

## Ranked shortlist

1. **Qwen3.5-2B** (fallback: Qwen3.5-0.8B, same family). It has the strongest vision benchmarks at about 2B parameters [1] and is Apache-2.0 [1]. Unsloth publishes a free-T4 vision LoRA notebook for it that peaks at 5.1 GB [5]. Official-community GGUF and ONNX exports exist [41][42], and a 512×512 tile costs 256 image tokens [3].
2. **LFM2.5-VL-1.6B** (fast variant: LFM2.5-VL-450M). Its native tile is 512×512, the same size as an Agriculture-Vision tile, capped at 256 image tokens [9]. Liquid ships its own GGUF, ONNX and LoRA notebooks (Unsloth and TRL) [9][10]. It is the cheapest path on CPU: Liquid measured the 450M variant at about 0.94 s per 512×512 image on a Ryzen AI Max+ 395 [11]. The licence is LFM Open License v1.0, which is fine for academic use [13].
3. **Gemma 4 E2B**. It is Apache-2.0 [8] and has the most mature tooling: an official llama.cpp GGUF [40], an ONNX export [42], and an Unsloth T4 vision notebook that ran fastest per sample of the Unsloth T4 reference runs [5][6][7]. The image-token budget is configurable from 70 to 1120 [8], and it is a conventional transformer, which hedges the runtime question in ticket 03. On the cost side, it has 5.1B total parameters including per-layer embeddings, a Q4 GGUF plus projector takes about 3.4 GB [41], and Liquid's comparison table puts its vision scores lowest of the three (vendor-reported) [12].

Not shortlisted:
- **GeoChat-class remote-sensing (RS) VLMs** are 4–7B, too slow for an 8 GB CPU, and score no better than small general VLMs on an agricultural RS benchmark [26][27].
- **PaliGemma 2** has the best RS evidence [18], but it is 3B, gated, has no usable GGUF, and costs 1024 tokens at 448 px.
- **JEPA stacks** are not viable as the served model in 7 weeks; see the JEPA section.
- **MiniCPM-V 4.7 1B** (released 2026-10-06) is a watchlist item. It is promising but 12 days old [14].

## Comparison table

T4 means a free Kaggle/Colab T4, about 14.6 GB usable [5]. CPU means the target laptop (WSL2, 7.8 GB RAM, 16 cores) running a Q4 GGUF through llama.cpp unless noted. Latency covers one 512×512 image plus a short answer (about 20 tokens) or a caption (about 60 tokens), and is **est.** unless cited.

| Model | Params (total) | License | Caption + VQA | HF `peft` / transformers | Fits T4 (LoRA / QLoRA) | Fits 8 GB CPU, rough latency | RS / agri evidence | GGUF / ONNX |
|---|---|---|---|---|---|---|---|---|
| **Qwen3.5-2B** | 2.27B [44] | Apache-2.0 [1] | Yes, general chat VLM. Disable thinking mode for latency [1] | Native `qwen3_5` [43]; LoRA via Unsloth notebook [5] | **Yes, 16-bit LoRA**: 5.1 GB peak, 240 samples in 559 s on T4 (LaTeX-OCR demo) [5]. QLoRA not needed | **Yes**: Q4_K_M is 1.28 GB [41]. est. 3–7 s | None reported. General VLMs often match RS-specific ones [27] | GGUF: unsloth [41]. ONNX: onnx-community [42] |
| Qwen3.5-0.8B | 0.87B [44] | Apache-2.0 [2] | Yes, but weaker: MMStar 58.3 vs 71.7 for 2B [1] | Same as 2B | **Yes**: 2.4 GB peak, 240 samples in 541 s [6] | **Yes**: Q4_0 is 0.56 GB [41]. est. 2–3 s | None reported | GGUF: ggml-org [41]. ONNX: onnx-community [42] |
| **LFM2.5-VL-1.6B** | 1.60B [44] | LFM Open License v1.0, Apache-based. Commercial use limited at ≥$10M revenue [13] | Yes. Vendor recommends it for captioning and VQA [9] | Native `lfm2_vl` [43]; official Unsloth/TRL LoRA notebooks [9] | **Yes, likely 16-bit LoRA**. Memory not measured: UNVERIFIED, est. 4–6 GB | **Yes**: Q4_0 0.70 GB plus mmproj 0.58 GB [41]. est. 2–4 s | None reported | Official GGUF and ONNX from LiquidAI [9][41][42] |
| LFM2.5-VL-450M | 0.45B [44] | LFM v1.0 [13] | Yes, but weaker: MMStar 43.0 vs 50.7 for 1.6B [9][10] | Same as 1.6B | Yes, easily (est.) | **Yes**: Q4_0 0.22 GB [41]. **0.94 s** at 512×512 on Ryzen AI Max+ 395 [11]; unclear if CPU or iGPU | None reported | Official GGUF and ONNX [10][42] |
| **Gemma 4 E2B** | 5.12B total, 2.3B effective (per-layer embeddings) [8][44] | Apache-2.0 [8] | Yes, captioning and VQA; image budget 70–1120 tokens [8] | Native `gemma4` [43]; Unsloth notebook [7] | **Yes, 16-bit LoRA**: 10.8 GB peak, 240 samples in 217 s [7]. QLoRA gives headroom | **Yes, but tighter**: Q4_0 2.84 GB plus mmproj 0.56 GB [41]. est. 4–9 s | None reported. Vision average 57.2 vs 63.7 for Qwen3.5-2B (Liquid's table, vendor-reported) [12] | GGUF: ggml-org, listed in llama.cpp docs [40][41]. ONNX: onnx-community [42] |
| MiniCPM-V 4.7 1B (watchlist) | 1.55B [44]; Qwen3.5-0.8B text + SigLIP2 [14] | Apache-2.0 [14] | Yes; 4x or 16x visual-token compression [14] | Native `minicpmv4_7`; needs transformers ≥5.18 [14][43]. Fine-tuning via SWIFT or LLaMA-Factory [14] | Likely yes (est.); no T4 run found | Yes (est.): Q4_0 0.64 GB [41]. est. 2–4 s | None reported. Claims to beat Qwen3.5-0.8B (vendor) [14] | Official GGUF [41]. ONNX community-only [42] |
| InternVL3.5-1B / 2B | 1.06B / 2.35B [44] | Apache-2.0 [19] | Yes [19] | `-HF` variants native [19] | Likely yes (est.) | Yes (est.). Vendor says its heavier encoder "stalls" time-to-first-token [12] | 2B scores 64.6 on Liquid's vision average [12]. InternVL2-2B beats GeoChat on AgroMind [26] | GGUF community (bartowski etc.) [41]. ONNX: not checked |
| SmolVLM2-256M / 500M / 2.2B | 0.26 / 0.51 / 2.25B [44] | Apache-2.0 [15] | Yes; 64 tokens per 512² tile on small variants [15] | Native `smolvlm` [43] | Yes (est.) | Yes. est. 1–2 s for 500M | None. 500M is beaten by LFM2.5-VL-450M on every vision benchmark Liquid lists except MMMU (vendor) [10] | GGUF: ggml-org [40]. ONNX in repo `onnx/` [42] |
| Florence-2-base / large | 0.23B / 0.78B [44] | MIT [16] | **Captions only.** The released weights have no VQA, so VQA needs a fine-tune [16] | Native `florence2` [43] | Yes: T4 at batch size 1 with frozen vision encoder [16] | Yes. est. 1–3 s (PyTorch or ONNX) | Falcon, a 0.7B RS model on Florence-2's architecture, does 14 RS tasks [24] | GGUF: no (encoder-decoder; not in llama.cpp list [40]). ONNX: onnx-community [42] |
| Moondream 2 (2025-06-21) | 1.93B [44] | Apache-2.0 [17] | Yes: native `caption()` and `query()` [17] | `trust_remote_code` only [17] | Likely (est.); no standard PEFT recipe found | Yes. GGUF is only the 2025-04-14 build [40]. est. 3–6 s | None | GGUF: older build only [40]. ONNX community-only [42] |
| Moondream 3.1 9B-A2B | 9.27B (2B active) [17][44] | Moondream Model License 1.0 (terms not read: UNVERIFIED) [17] | Yes [17] | No `library_name`; custom engine [17] | QLoRA only, tight (est.) | **No**: 9B weights do not fit 7.8 GB at useful precision (est.) | None | None official found |
| PaliGemma 2 3B (pt-448) | 3.03B [44] | Gemma licence, gated [18] | Yes after transfer (pt checkpoints are fine-tune bases) [18] | Native `paligemma` [43] | Yes with LoRA (est.) | **Marginal**: 448 px means 1024 image tokens; no GGUF. est. 15–30 s in PyTorch | **Best RS evidence**: after transfer fine-tuning, RSVQA-lr 93.0 / RSVQA-hr 92.7 (3B@224) [18] | GGUF: effectively none (only a 0-download repo) [41] |
| Qwen3-VL-2B | 2.13B [44] | Apache-2.0 [21] | Yes [21] | Native [21] | Yes (est.) | Yes. est. 3–7 s | Superseded: Qwen3.5-2B beats it on Qwen's own table [1] | Official GGUF [41]. ONNX: onnx-community [42] |
| LLaVA-OneVision-0.5B | 0.89B [44] | Apache-2.0 [20] | Yes [20] | Native [20] | Yes (est.) | Yes. est. 2–4 s | None | Not checked. Dated (2024) |
| GeoChat-7B | about 7B (LLaVA-1.5) [22] | apache-2.0 on HF card [22]; GitHub repo has no licence file (UNVERIFIED) | Yes, RS caption, VQA and grounding [22] | LLaVA codebase | QLoRA only (est.) | **No**: est. 15–30 s+ | AgroMind overall 25.93 vs 28.40 for InternVL2-2B (human 33.15, GPT-4o 43.14) [26] | Community GGUF only (UNVERIFIED quality) [41] |
| EarthDial-4B (RGB / MS) | 4.15B [44]; InternVL2 + Phi-3-mini [23] | GitHub MIT; HF MS card apache-2.0 [23] | Yes; 11.11M RS instruction pairs incl. **NIR** multispectral [23] | Custom InternVL2 code [23] | QLoRA probably (est.); authors used 8×A100 [23] | **No**: est. >15 s; no GGUF | Strongest NIR-aware option [23] | None found |
| Falcon (Florence-2-based) | 0.7B [24] | Code MIT [24]; HF weights gated, tagged `llama3.2` (UNVERIFIED) [24] | Yes, 14 RS tasks incl. captioning [24] | Florence-2-style custom | Yes (est.) | Yes (est.) | Trained on 78M RS samples [24] | None found |
| RS-Qwen2-VL-2B (AdaptLLM) | 2.44B [44] | Apache-2.0 [25] | Yes [25] | Native (Qwen2-VL) | Yes (est.) | Yes (est.) | RS domain-adapted; numbers not checked [25] | Community GGUF [41] |

Dropped on sight:
- **LiquidAI d1-omni-600M** is a decision/classification model that "does not write text" [46].
- **Microsoft Mage-VL** is 4.7B and built for streaming video [47].
- **Qwen3.6 / Qwen3.8** have only ≥27B sizes on the Qwen Hub org today (27B, 35B-A3B, 180B Flash-Next, 2.4T) [44].

### Notes that affect the choice

- **The NIR channel.** Every shortlisted model takes 3-channel RGB. Agriculture-Vision is RGB+NIR [26]. The choices are:
  - RGB only;
  - a false-colour NIR-R-G composite;
  - two images per question (RGB and CIR), which doubles image tokens.

  The only NIR-native options found are EarthDial-MS [23] and the JEPA encoder AnySat [32]. Both are too heavy or too custom to serve. AgroMind handled non-visible bands by converting them to grayscale blocks [26].
- **Training time on T4.** Unsloth's T4 reference runs on LaTeX-OCR data, at about 240 samples each, came out at:
  - Gemma 4 E2B: about 0.9 s/sample [7];
  - Qwen3.5-2B: about 2.3 s/sample [5];
  - Qwen3.5-0.8B: about 2.25 s/sample [6].

  The 0.8B and 2B runs take nearly the same time, which suggests fixed overhead (warm-up, or the Gated-DeltaNet kernels on T4), not compute (UNVERIFIED). If that held, 20k QA pairs per epoch would be about 5 h for Gemma and up to about 13 h for Qwen3.5-2B, which needs checkpoint and resume across Kaggle sessions. All three runs show `Bfloat16 = FALSE` and `FA2 = False` on T4 [5][7].
- **Architecture variety.** Qwen3.5 is a hybrid of Gated DeltaNet and attention [1]. LFM2.5 is Liquid's hybrid [9]. Gemma 4 is a sliding-window/global transformer [8]. Shortlisting all three hedges whichever runtime ticket 03 picks.

## JEPA section

**What pairing a JEPA encoder with a language model costs.** I-JEPA and V-JEPA encoders are self-supervised and **not language-aligned**, unlike SigLIP or CLIP, so building a captioner or VQA model from one is a full LLaVA-style VLM build:
1. A frozen JEPA encoder. Options:
   - I-JEPA ViT-H/14: 630M, **CC-BY-NC-4.0** [31].
   - V-JEPA 2 ViT-L: 326M, MIT [29][44].
   - V-JEPA 2.1, released 2026-03-16, which adds image tokenizers and dense features [29].
   - AnySat: MIT, JEPA-based, and accepts `aerial` B×4×H×W **RGB+NIR at 0.2 m**, which matches Agriculture-Vision's sensor [32].
2. An MLP projector into a small LM, for example Qwen3.5-0.8B's text tower.
3. Two-stage training: projector alignment on image-text pairs, then instruction tuning [28][35].

**The data and compute behind that recipe in the literature:**
- **V-JEPA 2 → LLM:** 18M image/video-text pairs in the controlled setup and 88.5M for state of the art, with Qwen2-7B or Llama-3.1-8B, three-stage LLaVA training [28].
- **LLaVA-1.5:** 1.2M public samples, about 1 day on an 8×A100 node, and that starts from an already language-aligned CLIP encoder [35].
- **VL-JEPA (Meta, 2025-12):** 1.6B parameters; predicts text embeddings and decodes them with a lightweight decoder. Pretraining took 4 weeks on 24 nodes × 8 H200 [30]. No official code or weights found on GitHub or the HF Hub, only community re-implementations (UNVERIFIED that none exist).
- **Remote sensing:** 2026 RS-JEPA language work (AlignJEPA) aligns an AnySat encoder to RemoteCLIP text for **retrieval**, and uses RSVQA only as a closed-set probe, not generation [33]. A 2026 V-JEPA traffic VQA system that ranked first used frozen V-JEPA for **closed-set** questions and handed caption generation to Qwen3-VL-8B [34].

**Viability, solo, in 7 weeks: not viable as the served model.**
- The auto-generated QA pairs (9 anomaly types, templated) are far too few and too narrow to teach a new projector-plus-LM language grounding. The model would memorise templates and fail the graded paraphrase-robustness test.
- No GGUF or ONNX path exists for a custom stack. A community `jepa.cpp` covers only V-JEPA 2-AC [41].
- Effort, est.: 3–5 weeks with a high risk of a non-functional captioner.

**A cheap JEPA option that fits (est. 2–3 days, after the core is done):** a frozen **AnySat** (RGB+NIR) with a linear or attentive probe for 9-way multi-label anomaly presence. This gives one "NIR-aware JEPA features" comparison row, and can feed a hedging signal (probe confidence), without touching captioning. The weights ship as `.pth` pickles [32], so load them with care.

## LoRA vs QLoRA vs DoRA

- **Plain 16-bit LoRA is enough for every shortlisted model.** Measured T4 peaks were 2.4 GB (Qwen3.5-0.8B), 5.1 GB (Qwen3.5-2B) and 10.8 GB (Gemma 4 E2B) [5][6][7]. That keeps the adapter math identical to the BLIP-LoRA row, which keeps the ablation clean.
- **QLoRA** (NF4 through bitsandbytes [37]) only matters for ≥4B models such as EarthDial-4B, GeoChat-7B or Gemma 4 E4B, or for buying batch size. NF4 runs on both T4 (CC 7.5) and P100 (CC 6.0); LLM.int8 needs CC ≥7.5, so not P100 [38]. T4 has no bf16 [5], so set `bnb_4bit_compute_dtype=torch.float16`; the PEFT docs' example uses bf16 [37]. Merge the adapter into the fp16 base before GGUF or ONNX export. Any accuracy shift from training against a 4-bit base and then merging into fp16 is UNVERIFIED.
- **DoRA** is one flag, `use_dora=True`. It splits updates into magnitude and direction, "can improve performance … especially at low ranks", and costs more during training. The PEFT docs recommend merging for inference, so serving cost after merge is zero. It supports only Linear and Conv2D layers [36][39]. It is worth one cheap extra row (LoRA vs DoRA at the same rank) only if time allows; it is not required.

## Open questions (for the map)

1. **NIR handling** for both BLIP and the served model: RGB only, a NIR-R-G composite, or dual images. This changes the QA scheme (ticket 05) and the evaluation.
2. **A one-day latency spike** on the actual laptop: Qwen3.5-2B, LFM2.5-VL-1.6B and Gemma 4 E2B as Q4 GGUFs on one 512² tile. It replaces every est. latency above before locking the served model.
3. **Can a merged LoRA be converted to GGUF** for Qwen3.5 (Gated DeltaNet) and LFM2.5-VL with the llama.cpp converter? This feeds the "weight hand-off" item.
4. **The official Kaggle GPU quota and session cap.** About 30 h/week and 12 h/session come only from secondary sources (UNVERIFIED) [45]. That decides whether Qwen3.5-2B needs resume-across-sessions.
5. Whether to LoRA the **vision tower** too. The Unsloth default is on [5]; aerial imagery is a large domain shift. This belongs to the PEFT-config item.

## Sources

1. Qwen3.5-2B model card (architecture, licence, vision table incl. 0.8B and Qwen3-VL-2B): https://huggingface.co/Qwen/Qwen3.5-2B
2. Qwen3.5-0.8B model card: https://huggingface.co/Qwen/Qwen3.5-0.8B
3. Qwen3.5-2B preprocessor/config (patch 16, merge 2 → 256 tokens per 512²): https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/preprocessor_config.json , https://huggingface.co/Qwen/Qwen3.5-2B/blob/main/config.json
4. Unsloth notebooks index: https://docs.unsloth.ai/get-started/unsloth-notebooks
5. Unsloth Qwen3.5 (2B) Vision notebook, T4 outputs: https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(2B)_Vision.ipynb
6. Unsloth Qwen3.5 (0.8B) Vision notebook: https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_5_(0_8B)_Vision.ipynb
7. Unsloth Gemma4 (E2B) Vision notebook: https://github.com/unslothai/notebooks/blob/main/nb/Gemma4_(E2B)-Vision.ipynb
8. Gemma 4 E2B-it model card: https://huggingface.co/google/gemma-4-E2B-it
9. LFM2.5-VL-1.6B model card: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B
10. LFM2.5-VL-450M model card: https://huggingface.co/LiquidAI/LFM2.5-VL-450M
11. Liquid AI blog, LFM2.5-VL-450M (edge latency table): https://www.liquid.ai/blog/lfm2-5-vl-450m
12. Liquid AI blog, LFM2.5-VL-3B (comparison vs Gemma 4 E2B, Qwen3.5-2B, InternVL3.5; vendor-reported): https://www.liquid.ai/blog/lfm2-5-vl-3b
13. LFM Open License v1.0: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B/blob/main/LICENSE
14. MiniCPM-V 4.7 1B model card and config: https://huggingface.co/openbmb/MiniCPM-V-4.7-1B
15. SmolVLM 256M/500M blog (93M SigLIP, 4096 px/token): https://huggingface.co/blog/smolervlm ; SmolVLM2: https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct
16. Florence-2-large card: https://huggingface.co/microsoft/Florence-2-large ; HF blog "Fine-tuning Florence-2" (no VQA in released weights; T4 bs 1): https://huggingface.co/blog/finetune-florence2
17. Moondream 2 card: https://huggingface.co/vikhyatk/moondream2 ; Moondream 3.1 card: https://huggingface.co/moondream/moondream3.1-9B-A2B
18. PaliGemma paper (RSVQA transfer): https://arxiv.org/abs/2407.07726 ; PaliGemma 2 paper: https://arxiv.org/abs/2412.03555 ; card: https://huggingface.co/google/paligemma2-3b-pt-448
19. InternVL3.5-1B-HF card: https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF
20. LLaVA-OneVision-0.5B: https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf
21. Qwen3-VL-2B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
22. GeoChat paper: https://arxiv.org/abs/2311.15826 ; weights: https://huggingface.co/MBZUAI/geochat-7B ; repo: https://github.com/mbzuai-oryx/GeoChat
23. EarthDial repo: https://github.com/hiyamdebary/EarthDial ; weights: https://huggingface.co/akshaydudhane/EarthDial_4B_MS
24. Falcon paper: https://arxiv.org/abs/2503.11070 ; repo: https://github.com/TianHuiLab/Falcon ; weights: https://huggingface.co/TianHuiLab/Falcon-Single-Instruction-Large
25. AdaptLLM RS-Qwen2-VL-2B: https://huggingface.co/AdaptLLM/remote-sensing-Qwen2-VL-2B-Instruct
26. AgroMind benchmark (includes Agriculture-Vision imagery; GeoChat vs InternVL2-2B): https://arxiv.org/abs/2505.12207
27. "MLLMs for RS Image Understanding: Domain-Specific or General-Purpose?" (2026): https://arxiv.org/abs/2607.20284
28. V-JEPA 2 paper (LLM alignment, 18M / 88.5M pairs): https://arxiv.org/abs/2506.09985
29. V-JEPA 2 / 2.1 repo (MIT; 2.1 released 2026-03-16): https://github.com/facebookresearch/vjepa2 ; weights: https://huggingface.co/facebook/vjepa2-vitl-fpc64-256
30. VL-JEPA paper: https://arxiv.org/abs/2512.10942
31. I-JEPA weights (cc-by-nc-4.0): https://huggingface.co/facebook/ijepa_vith14_1k ; repo: https://github.com/facebookresearch/ijepa
32. AnySat repo (JEPA EO model, aerial RGB+NIR 0.2 m, MIT): https://github.com/gastruc/AnySat ; paper: https://arxiv.org/abs/2412.14123
33. AlignJEPA (2026): https://arxiv.org/abs/2608.15456
34. V-JEPA traffic VQA, AI City 2026: https://arxiv.org/abs/2609.18562
35. LLaVA-1.5: https://arxiv.org/abs/2310.03744
36. PEFT LoRA guide / `use_dora`: https://huggingface.co/docs/peft/main/en/developer_guides/lora
37. PEFT quantization guide (QLoRA via bitsandbytes): https://huggingface.co/docs/peft/main/en/developer_guides/quantization ; QLoRA paper: https://arxiv.org/abs/2305.14314
38. bitsandbytes install (CC requirements): https://huggingface.co/docs/bitsandbytes/main/en/installation
39. DoRA paper: https://arxiv.org/abs/2402.09353
40. llama.cpp multimodal docs: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md
41. GGUF repos: https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF , https://huggingface.co/unsloth/Qwen3.5-2B-GGUF , https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF , https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF , https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF , https://huggingface.co/openbmb/MiniCPM-V-4.7-1B-gguf , https://huggingface.co/bartowski/OpenGVLab_InternVL3_5-1B-GGUF , https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF , https://huggingface.co/KwangHwi/Geochat-GGUF , https://huggingface.co/mradermacher/remote-sensing-Qwen2-VL-2B-Instruct-GGUF , https://huggingface.co/jepacpp/vjepa2-ac-vitg-GGUF , https://huggingface.co/Pro-Balak/paligemma2-3b-pt-224-gguf
42. ONNX repos: https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX , https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX , https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX , https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX , https://huggingface.co/LiquidAI/LFM2.5-VL-450M-ONNX , https://huggingface.co/onnx-community/Florence-2-large-ft , https://huggingface.co/onnx-community/Qwen3-VL-2B-Instruct-ONNX , https://huggingface.co/RASMUS/MiniCPM-V-4.7-1B-ONNX ; SmolVLM2 `onnx/` folder in [15]
43. transformers model docs: https://huggingface.co/docs/transformers/main/en/model_doc/qwen3_5 , …/lfm2_vl , …/gemma4 , …/minicpmv4_7 , …/florence2 , …/smolvlm , …/paligemma , …/vjepa2 , …/ijepa
44. Parameter counts from HF Hub API `safetensors.total`: https://huggingface.co/api/models/{repo_id}
45. Kaggle quota, secondary only (UNVERIFIED): https://pypi.org/p/kgz
46. d1-omni-600M card: https://huggingface.co/LiquidAI/d1-omni-600M
47. Mage-VL card: https://huggingface.co/microsoft/Mage-VL
