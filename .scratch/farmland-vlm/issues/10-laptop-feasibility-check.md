# Laptop feasibility check for shortlisted models

Type: task
Status: resolved
Blocked by: 03

## Question

Is each shortlisted model actually servable on this laptop (WSL2, 7.8 GB RAM, 16 cores, CPU only)? Agent-driven (AFK); record the numbers here for the served-model decision.

- Models: Qwen3.5-2B (and 0.8B), LFM2.5-VL-1.6B (and 450M), Gemma 4 E2B.
- For each, as Q4 GGUF through llama.cpp and through mistral.rs v0.9.4 (in-situ quantization at load), which the Rust inference research recommends: load time, peak RAM including during load-time quantization, latency on one 512×512 tile for a ~60-token caption and a ~20-token answer.
- Qwen3.5 on CPU in mistral.rs: v0.9.4 has `vision_models/qwen3_5` (including Gated-DeltaNet code) and a multimodal server example, but the examples only name 27B / 35B. Confirm the 2B loads and runs at a usable speed on CPU.
- Whether mistral.rs returns per-token probabilities for image requests (a possible hedging signal), and any image-size limits.
- Confirm a LoRA-merged checkpoint converts to the served format with its vision projector intact. A dummy adapter is enough; do the merge on Kaggle if the laptop runs out of RAM.

## Answer

**Yes: every shortlisted model runs on this laptop's CPU, in both llama.cpp and mistral.rs, from the same Q4 GGUF + projector files.** Numbers and method are in [laptop-feasibility](../research/laptop-feasibility.md).

- **Cold latency per 512×512 tile, llama.cpp (60-token caption / 20-token answer):**
  - LFM2.5-VL-450M: 2.3 / 1.9 s (0.7 GB RAM)
  - Qwen3.5-0.8B: 4.3 / 2.9 s (1.5 GB)
  - LFM2.5-VL-1.6B: 8.3 / 6.8 s (2.1 GB)
  - Qwen3.5-2B: 12.4 / 9.0 s (3.0 GB), or 8.3 / 5.6 s with image tokens capped at 144
  - Gemma 4 E2B: 8.7 / 5.2 s (4.8 GB)
- **mistral.rs v0.9.4** (prebuilt CPU binary) loads all of them, Qwen3.5-2B included. Its prefill is 1.3–4× slower and it uses about 2× the RAM. Qwen3.5-2B and Gemma 4 E2B reach 6.0–6.2 GB of the 7.8 GB WSL limit.
- **Image tokens dominate latency.** Capping them (`--image-max-tokens`) is the main speed knob, so train at the same budget.
- **Follow-up questions on the same tile** are near-free with prefix caching.
- **Hedging signal:** per-token log-probabilities with top-k alternatives come back for image requests in both runtimes.
- **Fine-tuned weights reach the laptop intact.**
  - A language-model-only LoRA (22–34 MB) applies at runtime to the official base GGUF in both runtimes.
  - A merged checkpoint converts to Q4_K_M GGUF + projector and keeps its trained answers.
  - A vision-tower LoRA survives into the exported projector, but only via the merge route.
  - mistral.rs quantize-at-load from safetensors peaks at 8.9 GB RAM, so use pre-quantized GGUF on the laptop.
- **Transfer:** `colab download` is 0.28 MB/s, fine for adapters. Merged weights (1.6–2 GB) need HF Hub (about 4–5 MB/s here).
