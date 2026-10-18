# Training feasibility check (Colab T4)

Type: task
Status: resolved
Blocked by: —

## Question

_Run on Colab's free T4 through the `colab` CLI instead (the user's call, 2026-10-18). The Kaggle-only items, DDP across both T4s and chaining commit outputs, were dropped._

What does LoRA training actually cost on Kaggle's free T4×2? Human in the loop: this machine has no Kaggle CLI or credentials, so either the user runs a notebook the agent writes, or sets up a Kaggle API token so the agent can push runs with `kaggle kernels push`.

- fp16 LoRA throughput (samples/s) on dummy 512×512 tiles with short QA pairs, for BLIP and the three shortlisted models, on one T4 vs DDP across both.
- Whether fp16 stays stable (no NaN / inf loss) over a few hundred steps for these bf16-pretrained models.
- Whether a commit can attach its own previous version's output as input to resume from a checkpoint, or chaining needs a second notebook or a `kagglehub` upload.
- How long one streaming pass of the official 21 GB archive takes into a Kaggle session (S3 vs the official HF repo), extracting only RGB + labels.
- Record the GPU hours spent; the weekly pool is ~30 h.

## Answer

**fp16 LoRA training is feasible on one free Colab T4 for BLIP and all shortlisted models.** Numbers and method are in [colab-training-check](../research/colab-training-check.md).

- **Throughput** (r=16, real 512×512 tiles, 256 image tokens):
  - BLIP base: 17–19 samples/s
  - LFM2.5-VL-450M: 9.4
  - Qwen3.5-0.8B: 3.8
  - LFM2.5-VL-1.6B: 3.0
  - Qwen3.5-2B: 2.5
  - Gemma 4 E2B: 1.6 (batch 1 only)
- **Epoch cost:** one pass over all 75k train+val tiles is about 1 h for BLIP, 7–13 h for the 1.6–2B candidates. The GPU is the bottleneck, not the 2 vCPUs.
- **Stability:** no NaN or inf losses anywhere. Qwen3.5 and Gemma 4 had recurring loss-scale overflows (5% and 3% of steps skipped), so watch them in long runs.
- **Data:** a full stream of the official 21 GB tar takes 12.5 min (S3, 28 MB/s with extraction), so each session can re-stream. No private copy of the dataset is needed.
- **Persistence:** nothing survives a session. Adapters (22–34 MB) come home via `colab download` in about 2–3 min; merged weights need HF Hub. Mid-run resume is untested.
- **GPU time used:** about 2.1 h of T4 session (about 45 min of compute). No free-tier limit was hit.
