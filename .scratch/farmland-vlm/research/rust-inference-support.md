# Rust inference support for BLIP and candidate VLMs (CPU, fine-tuned LoRA)

Researched 2026-10-18 against candle 0.11.0, mistral.rs v0.9.4, llama.cpp b11719 + llama-cpp-2 0.1.160, ort 2.0.0-rc.14, tch 0.26.0.

## Verdict

1. **A pure-Rust backend can serve a fine-tuned newer small VLM through mistral.rs** (native Rust, built on candle). Supported architectures that fit 7.8 GB: SmolVLM v1 (256M / 500M / 2.2B, `Idefics3ForConditionalGeneration`), Qwen2-VL-2B / Qwen2.5-VL-3B / Qwen3-VL-2B, LFM2/LFM2.5-VL (450M / 1.6B), Gemma 3 / 3n / 4. Fine-tuned weights go in as a PEFT-merged safetensors checkpoint, with ISQ Q4K/Q8_0 applied at load on CPU. An LM-only adapter can also be loaded at runtime. An axum router is built in. Use a git dependency on tag `v0.9.4`, because crates.io still has 0.8.1, which predates the CPU kernel rewrite and the current LoRA API.
2. **BLIP: candle has captioning only.** It has `BlipForConditionalGeneration` (f32, plus GGUF Q8_0/Q4K via `quantized_blip`) and **no `BlipForQuestionAnswering`**. BLIP VQA in Rust means porting the text encoder and the VQA wiring yourself (my estimate: 1–3 days plus a parity test against HF). Merged LoRA safetensors load directly because candle uses HF tensor names.
3. **No native-Rust path today** for Florence-2, PaliGemma 2, current Moondream2, InternVL, or SmolVLM2. SmolVLM2 has a different HF architecture string that mistral.rs does not map; whether it loads under an `idefics3` override is UNVERIFIED.
4. **FFI routes avoid Python but are not native Rust.** llama.cpp via `llama-cpp-2` adds SmolVLM2, InternVL 2.5/3 and Moondream2-2025 as GGUF; its `mtmd` binding is marked experimental. ort covers Florence-2 and PaliGemma 2 as ONNX, but optimum cannot export these fine-tuned architectures and the generation loop has to be hand-written. tch-rs is not worth it.
5. **Recommendation:** use **mistral.rs + axum** for the served model, and pick it from SmolVLM-256M/500M, Qwen2.5-VL-3B, Qwen3-VL-2B or LFM2.5-VL. Add **candle** for BLIP-LoRA only if the demo must serve BLIP too. If ticket 02 picks Florence-2, PaliGemma 2, InternVL or Moondream, the backend goes Python, unless the user accepts C++ FFI as "Rust".

## Support grid (model × runtime)

"Chat" means free-form prompts, so captioning works by prompt and VQA by question. "Port" means the building blocks exist but you have to write the model glue.

| Model | candle (native) | mistral.rs (native) | llama.cpp via llama-cpp-2 (FFI) | ort / ONNX Runtime (FFI) | tch-rs / libtorch (FFI) |
|---|---|---|---|---|---|
| BLIP caption (base, large) | **Yes** (large preset; base via config.json is UNVERIFIED) | No | No | Port: no public ONNX found, custom `torch.onnx.export` | Port: TorchScript trace |
| BLIP VQA (`blip-vqa-base`, `capfilt-large`) | **Port**: no `BlipForQuestionAnswering` | No | No | Port | Port |
| Florence-2 (base 0.23B / large 0.78B) | No (PR #3470 open, vision encoder only) | No | No | Base ONNX exists (onnx-community); fine-tuned export needs a custom script | Port |
| SmolVLM v1 (256M / 500M / 2.2B) | No (proposal #2828 closed, Idefics3 PR #3446 closed unmerged) | **Yes** (Idefics3) | Yes | ONNX in HF repo; fine-tuned export has no optimum support (issue #62 open) | Port |
| SmolVLM2 (256M / 500M / 2.2B) | No | UNVERIFIED (`SmolVLMForConditionalGeneration` not mapped) | Yes | ONNX in HF repo (256M/500M); same export problem | Port |
| Moondream2 (current, 2025+) | Only moondream1 / early moondream2 (`Config::v2`, pinned old revision) | No | Yes (`moondream2-20250414`) | Xenova ONNX (older revision, UNVERIFIED which) | Port |
| PaliGemma 2 (3B) | No: PaliGemma **1** only (Gemma-1 backbone) | No (issue #2482 open) | No (issue #7553 open, PR #24200 closed) | Base ONNX exists (onnx-community) | Port |
| Qwen2-VL-2B / Qwen2.5-VL-3B / Qwen3-VL-2B | Qwen3-VL only (no quantized variant, no example) | **Yes** (all three) | Yes | onnx-community exports exist | Port |
| InternVL 2.5 / 3 (1B, 2B) | No | No | Yes (non-HF checkpoints only) | UNVERIFIED | Port |
| Newer: LFM2.5-VL (450M / 1.6B), Gemma 4 E2B/E4B | Gemma 4 module exists | **Yes** | Yes | UNVERIFIED | Port |

## Detail matrix (supported pairs)

| Model × runtime | Caption | VQA / chat | LoRA path | CPU quantization | Reported CPU latency / RAM | Native vs FFI | Maturity |
|---|---|---|---|---|---|---|---|
| BLIP-large caption × candle | Yes | No | Merge in Python (`merge_and_unload` + `save_pretrained`) → `model.safetensors` (keys `vision_model.*`, `text_decoder.*` match) | GGUF Q8_0 / Q4K made by candle `tensor-tools quantize`, loaded by `quantized_blip` | ~4 s per caption, f32, release build (user report, CPU unknown, 2024). Weights: 1.88 GB f32, 505 MB q8_0, 271 MB q4k | Native | In tree since 2023; example maintained; conditional prompt is a TODO |
| BLIP VQA × candle | (n/a) | Port | Same merge path (1.54 GB f32 for vqa-base) | Same `tensor-tools` path once ported | None (no implementation) | Native | `TextModel` is private and `TextLMHeadModel::forward` hard-codes a causal mask. You need a bidirectional text encoder with cross-attention to the image, a decoder that cross-attends to the question states, and the answer loop |
| SmolVLM v1 × mistral.rs | By prompt | Yes | Merged checkpoint (always works). Dynamic PEFT adapter is LM-only, safetensors only (GGUF `llama` arch rejects dynamic LoRA). Vision/projector adapters unsupported | ISQ at load: Q4K, Q8_0, etc. on CPU; or UQFF prebuilt | Apr 2025 report: minutes per answer for 256M on CPU (issue #1300, before the v0.9.0 CPU rewrite); current numbers UNVERIFIED. Weights: 513 MB (256M) / 1.0 GB (500M) bf16 | Native | Idefics3 loader exists; old open issue #1068 "idefics3 example fails" (Jan 2025) |
| Qwen2.5-VL-3B / Qwen3-VL-2B × mistral.rs | By prompt | Yes | As above (LM-only adapter, or merge) | ISQ / UQFF | Text decode only: Qwen3-4B Q4K 33.9 tok/s on a 32-vCPU Sapphire Rapids (v0.9.0 report); vision encode not benchmarked. Qwen3-VL-2B is 4.26 GB bf16 before ISQ, so peak RAM at load on 7.8 GB is a risk (UNVERIFIED); use UQFF | Native | Mature families in mistral.rs; v0.9.x released Jul–Sep 2026 |
| SmolVLM v1/v2, Qwen-VL, InternVL, Moondream2 × llama.cpp | By prompt | Yes | Text-only adapter: `convert_lora_to_gguf.py` → `lora_adapter_set` in llama-cpp-2. Vision LoRA: merge, then `convert_hf_to_gguf.py` (text) + `--mmproj` (vision). Conversion is a Python step on Kaggle, not in the backend | GGUF Q4_K_M / Q8_0 (`llama-quantize`) | No first-party VLM CPU numbers found (UNVERIFIED). Files: SmolVLM-256M Q8 175 MB + 104 MB mmproj; Qwen2.5-VL-3B Q4_K_M 1.93 GB + 0.85 GB; InternVL3-1B Q8 675 MB + 333 MB; Moondream2 f16 2.84 GB + 0.91 GB | FFI (C/C++ built via cmake by `llama-cpp-sys-2`) | llama.cpp very mature (131k stars, daily builds); Rust `mtmd` wrapper says "experimental and subject to breaking changes" |
| Florence-2, PaliGemma 2, SmolVLM, Qwen-VL × ort | By prompt / task token | Yes | Merge, then **custom ONNX export**: optimum-onnx registers none of florence2 / idefics3 / paligemma / qwen2_vl | ONNX int8 / q4 variants (pre-exported repos ship several dtypes) | None found. Weights: Florence-2-base 463 MB, PaliGemma2-3B 6.06 GB bf16 | FFI (ONNX Runtime 1.31 shared lib) | ort still `2.0.0-rc.14`; Rust side must hand-write preprocessing, chat template, embed-merge and the KV-cache loop over `decoder_model_merged` |
| Any × tch-rs | Port | Port | Merge → TorchScript trace of encoder and one decoder step | libtorch dynamic int8 (UNVERIFIED for these models) | None | FFI (libtorch 2.13) | Maintained (0.26.0), but HF `generate()` does not trace; heavy libtorch dependency. Not recommended |

## Runtime summary

| Runtime | Version / activity | Native? | LoRA in | CPU quant | Effort for this project |
|---|---|---|---|---|---|
| candle | 0.11.0, 21k stars, pushed 2026-10-14 | Pure Rust (optional `mkl`) | No PEFT loader in candle itself; `candle-lora` is third-party, not on crates.io, last push 2025-04. Merge instead | GGUF k-quants for `quantized_*` models only | Low for BLIP caption; medium for BLIP VQA (port) |
| mistral.rs | v0.9.4 (2026-09-24), 7.8k stars; crates.io lags at 0.8.1 | Pure Rust, on a pinned candle 0.11.0 git rev | Dynamic PEFT LoRA (LM only) since v0.9.1; merged checkpoints always | ISQ (Q2K–Q8_0), UQFF, GGUF | Low: load checkpoint, set ISQ, mount router |
| llama.cpp / llama-cpp-2 | b11719 / 0.1.160 | FFI to C++ | GGUF LoRA (text) or merge | GGUF | Low–medium: C++ toolchain, conversion step, experimental binding |
| ort | 2.0.0-rc.14 (ORT 1.31) | FFI to C++ | Merge + custom export | ONNX int8/int4 | High |
| tch-rs | 0.26.0 (libtorch 2.13) | FFI to C++ | Merge + TorchScript | Limited | High |

Third-party crates (unvetted, not recommended for a graded build): `rlx-florence2` 0.2.11 (GPL-3.0, its own RLX framework) and `ffai-argus` 0.8.0 (SmolVLM captioning). Their quality is UNVERIFIED.

## Weight hand-off (if mistral.rs or candle is chosen)

On Kaggle, run `peft` `merge_and_unload()` and `save_pretrained(safe_serialization=True)`, then copy the folder to the laptop.

- **mistral.rs:** load with ISQ 4 or 8. Alternatively, run `mistralrs quantize` once to produce UQFF, so the full bf16 copy is never loaded on the 7.8 GB box.
- **candle BLIP:** load the merged safetensors as f32, or run `tensor-tools quantize` to GGUF Q8_0/Q4K for `quantized_blip`.

If LoRA also targets the vision tower, merging is mandatory, because the mistral.rs runtime adapter covers the LM only. For CPU latency, cap image size: mistral.rs has `--max-edge`, and SmolVLM's tile count grows with resolution.

## HTTP framework pairing

**axum 0.8 + tokio + tower-http (CORS for the Vite dev server).**

- mistral.rs's `mistralrs-server-core` builds an axum `Router`, OpenAI-compatible (`/v1/chat/completions`, image parts), that mounts under a sub-path (`.nest("/ai", router)`). Custom handlers can share `SharedMistralRsState` (`Arc<MistralRs>`). mistral.rs itself depends on axum 0.8.8.
- candle models are synchronous and hold KV caches behind `&mut self`. Run them on a dedicated inference thread fed by an mpsc channel, or call `tokio::task::spawn_blocking` behind a `Mutex`.
- Two candle builds would end up in one binary: mistral.rs's git rev plus crates.io 0.11.0. Cargo allows this, but whether it compiles cleanly is UNVERIFIED. Pin `candle-transformers` to the same git rev as mistral.rs to be safe.

## Sources

- candle model list: https://github.com/huggingface/candle/tree/main/candle-transformers/src/models
- candle BLIP: https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/blip.rs , https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/blip_text.rs , https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/quantized_blip.rs
- candle BLIP example (model ids, q4k GGUF): https://github.com/huggingface/candle/blob/main/candle-examples/examples/blip/main.rs , https://huggingface.co/lmz/candle-blip
- candle BLIP CPU timing report: https://github.com/huggingface/candle/issues/1735
- candle tensor-tools quantizer: https://github.com/huggingface/candle/blob/main/tensor-tools/src/main.rs
- candle Moondream (old revision, 0.68 tok/s CPU log): https://github.com/huggingface/candle/blob/main/candle-examples/examples/moondream/main.rs , https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/moondream.rs
- candle PaliGemma (Gemma-1 backbone, 3B-224/448 presets): https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/paligemma.rs
- candle Qwen3-VL, Gemma 4, `smol` (SmolLM3 only): https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/qwen3_vl , https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/gemma4 , https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/smol
- candle Florence-2 / BLIP-2 / SmolVLM status: https://github.com/huggingface/candle/pull/3470 , https://github.com/huggingface/candle/pull/3469 , https://github.com/huggingface/candle/issues/2828 , https://github.com/huggingface/candle/pull/3446
- candle-lora: https://github.com/EricLBuehler/candle-lora ; candle crate: https://crates.io/crates/candle-transformers
- mistral.rs supported models: https://github.com/EricLBuehler/mistral.rs/blob/master/docs/src/content/docs/reference/supported-models.md
- mistral.rs arch mapping (Idefics3 only): https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-core/src/model_metadata.rs
- mistral.rs LoRA (LM-only, PEFT files, CPU support, unsupported variants): https://github.com/EricLBuehler/mistral.rs/blob/master/docs/src/content/docs/guides/customize/lora-adapters.mdx
- mistral.rs GGUF and LoRA boundaries: https://github.com/EricLBuehler/mistral.rs/blob/master/docs/src/content/docs/reference/gguf-support.md
- mistral.rs quantization (ISQ on CPU): https://github.com/EricLBuehler/mistral.rs/blob/master/docs/src/content/docs/reference/quantization-types.md
- mistral.rs v0.9.0 CPU benchmark: https://github.com/EricLBuehler/mistral.rs/blob/master/releases/v0.9.0/report.md , https://github.com/EricLBuehler/mistral.rs/pull/2311
- mistral.rs dynamic LoRA (v0.9.1): https://github.com/EricLBuehler/mistral.rs/pull/2351 , https://github.com/EricLBuehler/mistral.rs/releases/tag/v0.9.1
- mistral.rs issues: https://github.com/EricLBuehler/mistral.rs/issues/1300 , https://github.com/EricLBuehler/mistral.rs/issues/1068 , https://github.com/EricLBuehler/mistral.rs/issues/2482 , https://github.com/EricLBuehler/mistral.rs/issues/1666
- mistral.rs axum embedding: https://github.com/EricLBuehler/mistral.rs/blob/master/docs/src/content/docs/guides/rust/embed-in-axum.md , https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-server-core/Cargo.toml , https://github.com/EricLBuehler/mistral.rs/blob/master/Cargo.toml
- mistral.rs Rust multimodal API: https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/getting_started/multimodal/main.rs ; crate: https://crates.io/crates/mistralrs
- llama.cpp multimodal: https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md , https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md , https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/clip-impl.h
- llama.cpp PaliGemma: https://github.com/ggml-org/llama.cpp/issues/7553 , https://github.com/ggml-org/llama.cpp/pull/24200
- llama.cpp conversion: https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py , https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py
- llama-cpp-2: https://crates.io/crates/llama-cpp-2 , https://github.com/utilityai/llama-cpp-rs/blob/main/llama-cpp-2/src/mtmd.rs , https://github.com/utilityai/llama-cpp-rs/blob/main/llama-cpp-2/src/context.rs
- ort: https://crates.io/crates/ort , https://github.com/pykeio/ort
- optimum-onnx export coverage: https://github.com/huggingface/optimum-onnx/blob/main/optimum/exporters/onnx/model_configs.py , https://github.com/huggingface/optimum-onnx/issues/62
- ONNX exports: https://huggingface.co/onnx-community/Florence-2-base-ft , https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct , https://huggingface.co/onnx-community/paligemma2-3b-pt-224 , https://huggingface.co/onnx-community/Qwen2.5-VL-3B-Instruct-ONNX , https://huggingface.co/onnx-community/Qwen3-VL-2B-Instruct-ONNX , https://huggingface.co/Xenova/moondream2
- GGUF repos: https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF , https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF , https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF , https://huggingface.co/ggml-org/InternVL3-1B-Instruct-GGUF , https://huggingface.co/ggml-org/moondream2-20250414-GGUF
- Model sizes / architecture strings: https://huggingface.co/Salesforce/blip-image-captioning-large , https://huggingface.co/Salesforce/blip-vqa-base , https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct , https://huggingface.co/microsoft/Florence-2-base , https://huggingface.co/google/paligemma2-3b-pt-224 , https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct , https://huggingface.co/LiquidAI/LFM2.5-VL-450M
- tch-rs: https://crates.io/crates/tch , https://github.com/LaurentMazare/tch-rs
- axum / tower-http: https://crates.io/crates/axum , https://crates.io/crates/tower-http
- Third-party: https://crates.io/crates/rlx-florence2 , https://crates.io/crates/ffai-argus
