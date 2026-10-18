# Rust inference support for BLIP and candidates

Type: research
Status: resolved
Blocked by: —

## Question

For BLIP and each likely candidate model, can inference run fully in Rust on a CPU laptop (~8 GB RAM) with fine-tuned LoRA weights?

This decides the backend: Rust end-to-end if the served model runs in Rust, otherwise fully Python. No hybrid.

- Models: BLIP (captioning and VQA checkpoints, base and large), Florence-2, SmolVLM / SmolVLM2, Moondream, PaliGemma 2, small Qwen-VL / InternVL variants.
- Runtimes: candle, mistral.rs, ort (ONNX Runtime), tch-rs, llama.cpp via Rust bindings. Mark native Rust vs FFI.
- For each model × runtime: what's supported (captioning, VQA / chat), how LoRA weights get in (load adapter vs merge into base then export), CPU quantization, reported CPU latency and RAM, project maturity.
- Specifically: does candle implement BLIP VQA (`BlipForQuestionAnswering`) or only captioning?
- Which Rust HTTP framework pairs naturally with the winning runtime.

## Answer

- Pure Rust works for the served model with **mistral.rs v0.9.4** (git dependency; crates.io lags at 0.8.1). It covers SmolVLM v1 (256M/500M), Qwen2.5-VL-3B, Qwen3-VL-2B and LFM2.5-VL: load the PEFT-merged safetensors and apply ISQ Q4K/Q8_0 at load. Runtime LoRA adapters cover the LM only.
- candle has **BLIP captioning only** (f32 plus GGUF Q8/Q4K). There is **no `BlipForQuestionAnswering`**, so BLIP VQA in Rust is a custom port.
- No native-Rust support: Florence-2, PaliGemma 2, current Moondream2, InternVL, SmolVLM2 (UNVERIFIED). These need C++ FFI (llama.cpp / ort) or a Python backend.
- HTTP: axum 0.8. mistral.rs ships an axum router you can mount under a sub-path.

[findings](../research/rust-inference-support.md)

## Comments

- 2026-10-18 follow-up (orchestrator): the research ran before the shortlist existed, so it never checked Qwen3.5. mistral.rs v0.9.4 has `mistralrs-core/src/vision_models/qwen3_5/` (with `packed_gdn.rs`, `packed_visual.rs`) and `examples/server/qwen3_5.py`, which sends an `image_url` (example models: Qwen3.5-27B / 35B-A3B). So all three shortlisted models (Qwen3.5, LFM2.5-VL, Gemma 4) have a mistral.rs vision path; CPU speed is unmeasured. Source: https://github.com/EricLBuehler/mistral.rs/tree/v0.9.4/mistralrs-core/src/vision_models/qwen3_5
