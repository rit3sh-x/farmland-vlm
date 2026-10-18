# Colab training check: fp16 LoRA on a free T4

Measured 2026-10-18 for [Training feasibility check (Colab T4)](../issues/12-kaggle-training-check.md). Scripts and raw results are in [`checks/`](../checks/): `av_stream.py`, `train_bench.py`, `run_all.sh`, `lora_export.py`, `convert_check.py`, `gguf_check.py`, `isq_check.py`, `colab-*-results.jsonl`.

## Setup

- **Hardware:** Colab free tier (0 compute units), driven entirely from the terminal with `colab --auth adc exec/upload/download`. Tesla T4 with 15 GB, 2 vCPU Xeon (AVX-512), 12 GB RAM, 71 GB free disk, Ubuntu 24.04.
- **Software:** torch 2.11 (CUDA 13.0), transformers 5.18, PEFT 0.21.1.
- **Training setup:**
  - Base weights frozen in fp16, LoRA r=16 / α=32 / dropout 0.05 in fp32.
  - AdamW at lr 1e-4, fp16 autocast + `GradScaler`, SDPA attention.
  - Gradient checkpointing for the VLMs, not for BLIP.
- **Data:**
  - Real train tiles at 512×512.
  - Targets: half "Is there <anomaly> in this tile?" → yes/no, half a "This tile shows …" caption. Class presence comes from label-file size, a stand-in until the real QA-pair scheme exists.
  - Loss on answer tokens only.
- **Run:** 200 steps per model at the largest batch size that fit; throughput timed over steps 10–200.
- **LoRA targets:**
  - VLMs: every linear layer in the language model (attention, MLP, Qwen3.5's DeltaNet `in_proj_*`/`out_proj`, Gemma's per-layer projections).
  - BLIP: `query`/`value`/`qkv` in the text and vision encoders.

## Results

| Model | Trainable | Batch | Seq len | Samples/s | s/step | Peak GPU | Non-finite loss | Scaler overflows (final scale) | Loss, first 20 → last 20 steps | Est. h per 75k pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| BLIP VQA base | 3.0 M | 16 | 10 | 17.4 | 0.92 | 8.4 GB | 0 | 0 (65536) | 0.33 → 0.14 | 1.2 |
| BLIP caption base | 1.8 M | 16 | 9 | 18.8 | 0.85 | 8.2 GB | 0 | 0 (65536) | 6.5 → 1.5 | 1.1 |
| LFM2.5-VL-450M | 6.0 M | 16 | 285 | 9.4 | 1.70 | 14.2 GB | 0 | 1 (32768) | 1.04 → 0.21 | 2.2 |
| LFM2.5-VL-1.6B | 11.1 M | 4 | 285 | 3.0 | 1.35 | 10.2 GB | 0 | 1 (16384) | 1.30 → 0.28 | 7.0 |
| Qwen3.5-0.8B | 10.8 M | 4 | 287 | 3.8 | 1.04 | 11.9 GB | 0 | **10 (32)** | 3.25 → 0.26 | 5.4 |
| Qwen3.5-2B | 16.8 M | 2 | 288 | 2.5 | 0.81 | 10.4 GB | 0 | **10 (32)** | 3.10 → 0.29 | 8.5 |
| Gemma 4 E2B | 26.3 M | 1 | 284 | 1.6 | 0.63 | 13.6 GB | 0 | **6 (512)** | 5.6 → 0.53 | 13.2 |

"Est. h per 75k pairs" is one pass at one pair per train+val tile, at this throughput.

## Findings

1. **Everything trains in fp16 on one free T4.**
   - No run had a NaN or inf loss.
   - BLIP and LFM2.5-VL were clean.
   - Qwen3.5 (both sizes) and Gemma 4 overflowed and skipped 5% and 3% of steps; the loss scale fell to 32 and 512. That is workable but worth watching in long runs. Levers: lower LR, keep norms in fp32, or cap the loss scale's fall.
2. **The GPU is the bottleneck, not the 2 vCPUs.** Dataloader wait was 0–1% of step time, with JPEG decode and processor work in 2 workers.
3. **Full-dataset epochs are long for the bigger candidates.**
   - A pass over all 75k train+val tiles takes 7–13 h for LFM2.5-VL-1.6B, Qwen3.5-2B and Gemma 4 E2B.
   - Lowering the image budget from 256 to about 144 tokens, which serving also wants, should cut that roughly 1.5–1.8× (est.).
   - Gemma 4 E2B only fits batch 1.
   - Qwen3.5's Gated-DeltaNet layers ran on transformers' torch fallback. `flash-linear-attention` / `causal-conv1d` were not installed; whether they work on a T4 is untested.
4. **Data: re-stream every session; no private copy needed.**
   - A full pass of the official tar from S3, with gunzip + tar on the fly, moved 21.03 GB in **12.5 min** at 28 MB/s.
   - 30 s raw probes: S3 41 MB/s, HF 48 MB/s.
   - Train+val with NIR unpacks to about 20 GB: images 13 + 3.8 GB, labels 2.0 + 0.7 GB.
   - So each session can rebuild its data in about 13 min, which sidesteps the terms question about keeping a private copy.
5. **Session and persistence.**
   - One CLI session ran 2.1 h with no disconnect (VM uptime 7,466 s), mostly unattended `nohup` jobs polled with short `colab exec` calls. No free-tier limit was hit.
   - Nothing survives `colab stop`.
   - Adapters (22–34 MB in fp16) come home via `colab download` in about 2–3 min at 0.28 MB/s. That is the cheap checkpoint path. Merged weights need HF Hub, since Drive mounting is interactive.
   - Resuming mid-run from a downloaded checkpoint was not tested.
6. **Colab environment fixes the build will need:**
   - `pip uninstall -y torchao`: PEFT 0.21 refuses the preinstalled 0.10.
   - BLIP must skip `enable_input_require_grads`, because its BERT embeddings add in place.
   - Long commands go through `nohup`, since `colab exec` times out waiting for a reply. Use `--timeout`.
   - Never `pkill -f <pattern>` from an inline command that contains the pattern; it kills itself.
7. **Kaggle-only items were dropped:** DDP across two T4s, and chaining commit outputs. The user moved model work to Colab.
8. **GPU time used:** about 2.1 h of T4 session, of which about 45 min was GPU compute. The session was stopped afterwards.

## Sources

1. Official archive: `https://intelinair-data-releases.s3.amazonaws.com/agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz`, and `shi-labs/Agriculture-Vision` on HF (see [agriculture-vision-access](agriculture-vision-access.md)).
2. Colab CLI 0.7.4: `colab readme`, `colab skill`.
