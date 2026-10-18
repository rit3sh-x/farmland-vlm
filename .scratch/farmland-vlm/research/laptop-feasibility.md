# Laptop feasibility: shortlisted models on CPU

Measured 2026-10-18 for [Laptop feasibility check for shortlisted models](../issues/10-laptop-feasibility-check.md). Scripts and raw JSON are in [`checks/`](../checks/) (`bench_serve.py`, `bench_pair.sh`, `lora_local.sh`, `laptop-results/`). The laptop scripts hard-code the session scratchpad path; change `S=` to rerun them.

## Setup

- **Laptop:** i5-12500H (4P+8E cores, 16 threads, AVX2 + AVX-VNNI, no AVX-512), WSL2 with 7.8 GB RAM and 12 GB swap.
- **Runtimes:**
  - llama.cpp b11719: prebuilt `ubuntu-x64` CPU build, `llama-server`.
  - mistral.rs v0.9.4: prebuilt `mistralrs-cpu-x86_64-unknown-linux-gnu`. `mistralrs doctor` reports AVX/AVX2/FMA compiled in and no MKL.
  - Both serve the **same GGUF + projector files**.
- **Input:** 4 real Agriculture-Vision val tiles (512×512 JPEG: drydown, weed cluster, nutrient deficiency, water), through the OpenAI-style `/v1/chat/completions` with the image as a data URI.
- **Procedure:** one warm-up request, then a caption (60 tokens) and an answer (20 tokens) per tile, at temperature 0.
- **Cold numbers:**
  - llama.cpp runs had `cache_prompt: false` and `ignore_eos`, so they are cold and exact-length.
  - mistral.rs has no `ignore_eos`, and its prefix cache reuses a tile's image prompt between the caption and the answer. Its cold numbers below are therefore estimated as *caption prefill + N / decode rate*.
- **Peak RAM** is the server's `VmHWM`.

## Results

**llama.cpp** (`-t 16`; Qwen3.5-2B `-t 12`; 450M `-t 8`)

| Model (files) | Download | Load | Peak RAM | Prompt tokens | Prefill | Decode | 60-token caption | 20-token answer |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-VL-450M (Q4_K_M + Q8_0 projector) | 0.33 GB | 1.0 s | 0.7 GB | 282 | 1.6 s | 104 tok/s | **2.3 s** | **1.9 s** |
| Qwen3.5-0.8B (Q4_K_M + F16) | 0.74 GB | 2.5 s | 1.5 GB | 284 | 2.1 s | ~30 tok/s | **4.3 s** | **2.9 s** |
| LFM2.5-VL-1.6B (Q4_K_M + Q8_0) | 1.31 GB | 1.5 s | 2.1 GB | 282 | 6.5 s | 35 tok/s | 8.3 s | 6.8 s |
| Qwen3.5-2B (Q4_K_M + F16) | 1.95 GB | 3.0 s | 3.0 GB | 284 | 7.1 s | 15 tok/s | 12.4 s | 9.0 s |
| Qwen3.5-2B, `--image-max-tokens 144` | | 2.5 s | 3.0 GB | 172 | 3.8 s | 15 tok/s | **8.3 s** | **5.6 s** |
| Qwen3.5-2B, `--image-max-tokens 64` | | 3.8 s | 2.9 GB | 92 | 2.1 s | 17 tok/s | 5.2 s | 3.1 s |
| Gemma 4 E2B (Q4_0 + Q8_0) | 3.40 GB | 5.0 s | 4.8 GB | 146 | 3.4 s | 11–15 tok/s | 8.7 s | 5.2 s |

**mistral.rs** (same files, `serve multimodal --cpu -f <gguf> --mmproj <proj> --tok-model-id <hf id>`)

| Model | Load | Peak RAM | Prompt tokens | Cold prefill | Decode | 60-token caption (est.) | 20-token answer (est.) | Answer, warm prefix (measured) |
|---|---|---|---|---|---|---|---|---|
| LFM2.5-VL-450M | 9.5 s | 1.5 GB | 282 | 3.7 s | 56 tok/s | 4.7 s | 4.0 s | 3.8 s |
| Qwen3.5-0.8B | 12 s | 2.8 GB | 284 | 3.7 s | 12 tok/s | 8.7 s | 5.4 s | 2.2 s |
| LFM2.5-VL-1.6B | 9 s | 4.2 GB | 282 | 22–27 s | 18 tok/s | ~28 s | ~26 s | 24 s |
| Qwen3.5-2B | 18 s | **6.0 GB** | 284 | 8.9 s | 8 tok/s | 16 s | 11 s | 5.5 s |
| Gemma 4 E2B | 34 s | **6.2 GB** | 281 | 13.5 s | 7.5 tok/s | 22 s | 16 s | 0.6 s |

## Findings

1. **Every shortlisted model runs on this laptop in both runtimes.**
   - mistral.rs 0.9.4 loads multimodal GGUF for `qwen35`, `lfm2` and `gemma4` [1]. That includes Qwen3.5-2B, with its Gated-DeltaNet layers, on CPU.
   - So one exported file set serves either backend.
2. **llama.cpp is faster and lighter on this CPU.**
   - Its prefill (image encoding plus prompt) is 1.3–4× faster: about 2× for 450M and 0.8B, 1.3× for Qwen3.5-2B, about 4× for LFM2.5-VL-1.6B. Gemma's prompts are not comparable (146 vs 281 tokens).
   - It uses roughly half the RAM. In mistral.rs, Qwen3.5-2B and Gemma 4 E2B sit at 6.0–6.2 GB of the 7.8 GB WSL limit.
   - The prebuilt mistral.rs binary already has AVX2/FMA. A native or MKL build is an untested lever.
3. **Image cost dominates latency.** A 512×512 tile is about 256 image tokens for LFM2.5-VL and Qwen3.5. Gemma's default budget differs between the runtimes: about 130 tokens in llama.cpp, about 265 in mistral.rs.
   - Capping image tokens is the main speed knob: Qwen3.5-2B goes from 12.4 to 8.3 s at 144 tokens, and to 5.2 s at 64.
   - The flags are `--image-max-tokens` in llama.cpp and `--max-edge` / `--max-image-length` in mistral.rs. The mistral.rs ones are untested.
   - Training should then use the same image budget.
4. **Follow-up questions on the same tile are cheap.** mistral.rs's prefix cache cut Gemma's answer after a caption to 0.6 s. llama-server does the same by default (`cache_prompt`, disabled here to measure cold).
5. **Per-token probabilities work for image requests in both runtimes.** `logprobs` + `top_logprobs` return, for example, `Yes` −0.08 vs `No` −2.55 (LFM2.5-VL-450M). That is usable as a hedging signal.
6. **Qwen3.5 thinks by default.** Send `chat_template_kwargs: {"enable_thinking": false}`. llama.cpp still leaked an empty `<think></think>` once, so strip it.

## Fine-tuned weights → laptop

Adapters came from [the Colab training check](colab-training-check.md) (r=16, language model only, 200 steps). Base behaviour was verbose ("Yes, there is a visible drydown…"). The trained behaviour is terse (lowercase "no", plus captions like "This tile shows nutrient deficiency.").

| Route | Laptop files | Result |
|---|---|---|
| **A. Runtime adapter on the official base GGUF** | Official Q4 GGUF from HF, plus the adapter: 22 MB (LFM 1.6B) / 34 MB (Qwen 2B) fp16 | Same trained answers in both runtimes for both models.<br>llama.cpp: `--lora lora-f16.gguf` (made by `convert_lora_to_gguf.py`).<br>mistral.rs: `--lora ft=<PEFT dir>` plus `"adapter": "ft"` in the request.<br>Latency matches the base model. |
| **B. Merged → GGUF** | `convert_hf_to_gguf.py` (+ `--mmproj`) → `llama-quantize Q4_K_M`: 0.73 + 0.85 GB (LFM), 1.27 + 0.67 GB (Qwen) | Trained answers reproduced in llama.cpp and mistral.rs (run on Colab's CPU). |
| **C. Merged safetensors + mistral.rs quantize-at-load** | 4.5 GB bf16 (Qwen 2B) | Works ("no"), but loading took 84 s on 2 vCPUs with **8.9 GB peak RAM**. That exceeds the laptop's 7.8 GB, so it's not for the laptop. |

- **Vision projector:** a dummy LoRA on the vision tower, merged and exported with `--mmproj`, changed 163 of 441 projector tensors (LFM) and 100 of 298 (Qwen). A vision-tower LoRA survives into the projector, but only through route B. Runtime adapters reach the language model only [2].
- **Transfer speeds:**
  - `colab download`: 0.28 MB/s, measured on a 44 MB file. Fine for adapters (about 2–3 min); impractical for route B's 1.6–2 GB (about 1.5–2 h).
  - HF Hub to this laptop: about 3.5–5.5 MB/s.
  - So merged weights would travel through a private HF repo. That needs an HF token; `mistralrs doctor` found one in the local HF cache.
- **Disk:** peak use stayed around 2.5 GB, since files were deleted after each model. Windows free space went from 45 to 44 GB.

## Sources

1. mistral.rs v0.9.4, GGUF compatibility reference: https://github.com/EricLBuehler/mistral.rs/blob/v0.9.4/docs/src/content/docs/reference/gguf-support.md
2. mistral.rs v0.9.4, LoRA adapters guide ("For multimodal models, the adapter applies only to the language model"): https://github.com/EricLBuehler/mistral.rs/blob/v0.9.4/docs/src/content/docs/guides/customize/lora-adapters.mdx
3. llama.cpp release b11719: https://github.com/ggml-org/llama.cpp/releases/tag/b11719
4. Model files: `LiquidAI/LFM2.5-VL-{450M,1.6B}-GGUF`, `unsloth/Qwen3.5-{0.8B,2B}-GGUF`, `ggml-org/gemma-4-E2B-it-GGUF` on Hugging Face.
