# Kaggle and Colab free GPU limits

Research for ticket [11-free-gpu-limits](../issues/11-free-gpu-limits.md). As of 2026-10-18. Bracketed numbers like [3] point to Sources at the end. **UNVERIFIED** marks anything not confirmed in a primary source. **est.** marks my own estimates.

How the Kaggle pages were read: kaggle.com/docs pages and discussion threads render client-side, so I read the same content through Kaggle's anonymous site endpoints (`cms.LegacyCmsService/ListDocs` with `wantContent`, and `discussions.DiscussionsService/GetForumTopicById`). Strings marked "editor UI" come from Kaggle's own notebook-editor JavaScript bundle [14]. That bundle is first-party, but it is code, not documentation.

## Summary

1. **GPU types.** Kaggle's only free GPU is now **GPU T4 x2** (two T4s, 16 GB each). Kaggle retired the P100 on **2026-09-15** and auto-switched P100 notebooks to T4x2 [6]. L4, A100, H100 and RTX Pro 6000 are limited to specific competitions or Kaggle admins [6][11]. Colab does not publish which GPUs its free tier offers; they "vary over time" [15].
2. **Weekly hours.** Kaggle gives **30 h/week of GPU "or sometimes higher depending on demand"** [2]. This is a floating quota that resets weekly, on Saturday 00:00 UTC per the 2020 announcement [7]. One hour on T4x2 costs one quota hour even though it uses two GPUs [8]. TPU quota is separate: 20 h/week, 9 h per session [5]. Colab publishes no usage limits and says so deliberately [15].
3. **Session length.** Kaggle allows **12 h per CPU or GPU session** (9 h on TPU), and a Save & Run All must finish within 12 h "in order to save successfully" [1]. Colab free allows "at most 12 hours, depending on availability and your usage patterns" [15]. Both secondary figures (30 h/week, 12 h/session) are **confirmed**.
4. **Idle timeout.** Kaggle's own docs disagree: interactive sessions get **20 min** of idle time per the Notebooks page [1], but the GPU-tips page says **60 min** [2]. The idle timeout applies to interactive sessions only; for longer runs, Kaggle's docs point to Save & Run All [1]. Colab does not publish its idle timeout, and its VMs "are deleted when idle for a while" [15].
5. **Committed runs.** Save & Run All, also called "commit", starts a clean, separate session, runs the notebook top to bottom, and saves a version together with its outputs [1]. It can be launched headless with `kaggle kernels push --accelerator NvidiaTeslaT4` [11][12]. Kaggle caps the number of concurrently running GPU commits but does not publish the number (editor UI [14]). Colab free has no background execution; it is listed as a paid-plan benefit [15].
6. **RAM and disk.** A Kaggle GPU session has 4 CPU cores and **29 GB RAM**, **20 GB of saved `/kaggle/working`**, and scratch disk outside it that is not saved (size unpublished) [1]. Kaggle datasets are capped at 200 GB each, 200 GB of private datasets in total, and 50 top-level files [3]. Colab free gives a "standard" memory profile, and Colab does not publish disk size [15].
7. **What persists.** On Kaggle, each version's `/kaggle/working` output (up to 20 GB) is saved and can be attached as an input to later notebooks [1]. Interactive sessions can turn on a best-effort "Persistence" mode for files and variables [10]. Datasets and models can be uploaded with `kagglehub` [13]. On Colab, nothing on the VM survives, so checkpoints go to Google Drive through `drive.mount` and `flush_and_unmount` [16].
8. **bf16 and FlashAttention.** The T4 is compute capability (CC) 7.5 [17], and its tensor cores take **FP16, INT8 and INT4 but not BF16**; BF16 starts at CC 8.0 [19]. FlashAttention-2 needs Ampere or newer; Turing (T4) is only covered by a separate repo that implements a subset [21]. The P100 is CC 6.0 [18], which no longer matters now that it is retired. In practice: train in fp16 mixed precision with SDPA attention, not bf16 or FA2.
9. **Stale docs.** Kaggle's Notebooks and GPU-tips pages still describe a "single NVIDIA Tesla P100" [1][2]. The announcement [6] and the CLI docs [11] supersede them.

## Kaggle vs Colab (free tiers)

| | Kaggle (free) | Colab (free) |
|---|---|---|
| GPU offered | **GPU T4 x2** only: 2× T4, 16 GB each [6]. P100 retired 2026-09-15 [6]. L4 is competition-only [6]. A100, H100 and RTX Pro 6000 are competition- or admin-only [11]. The editor UI also has a "GPU T4 x2 HighMem" label [14]; whether free users can pick it is UNVERIFIED | Unpublished, "vary over time" [15]. Commonly a single T4: UNVERIFIED |
| Weekly GPU hours | 30 h, "or sometimes higher depending on demand" [2]. Floating quota, same for all users, weekly reset (Saturday 00:00 UTC per [7]; current reset time UNVERIFIED). T4x2 costs 1 quota hour per wall-clock hour [8] | Unpublished; "dynamic usage limits that sometimes fluctuate" [15] |
| Max session | 12 h CPU/GPU, 9 h TPU [1]. A commit must finish within 12 h to save [1] | At most 12 h "depending on availability and your usage patterns" [15] |
| Idle timeout | Interactive: 20 min [1] **or** 60 min [2] (Kaggle's pages conflict). Not documented for commits | Unpublished [15] |
| Background / unattended run | Save & Run All runs in its own session from a clean state [1]; also via CLI `kernels push` with `--accelerator` and `-t/--timeout` [11]. Concurrent GPU commits are capped, number unpublished [14] | None on the free tier; background execution is a paid "higher plans" feature [15] |
| CPU / RAM | 4 cores, 29 GB with a GPU [1][9]. 4 cores, 30 GB CPU-only [1] | "Standard system memory profile", size unpublished [15] |
| Disk | 20 GB saved `/kaggle/working`, plus unsaved scratch of unpublished size [1] | Unpublished. More Drive space does not add VM disk [15] |
| Persists across sessions | Version outputs (≤20 GB) can be reused as notebook inputs [1]. Interactive "Persistence" modes (Files only / Variables only / both) are best-effort and cover only `/kaggle/working`, not GPU memory [10]. Quick Save can optionally save outputs (editor UI [14]) | Nothing on the VM [15]. Use Drive via `google.colab.drive.mount` and `flush_and_unmount` [16] |
| Upload limits | Dataset ≤200 GB; private datasets ≤200 GB total; ≤50 top-level files [3]. Models via `kagglehub.model_upload` [4][13]; model size limit not stated in docs (UNVERIFIED) | n/a (Drive quota is a Google account matter, not covered by Colab docs) |
| Account gates | Accelerator selection is gated on phone verification (editor UI [14]) | Using multiple accounts to get around limits is disallowed [15] |
| bf16 / FA2 | T4: no BF16 tensor-core support (CC 7.5) [17][19][20]; no FA2 [21] | Same, if the free GPU is a T4 |
| Limits published? | Yes, partly, with internal inconsistencies (idle timeout, stale P100 text) | **No, deliberately**: "Colab does not publish these limits, in part because they can vary over time" [15]. Still true |

## Where sources disagree

- **30 h/week and 12 h/session.** Both are confirmed by Kaggle's own docs [1][2]. The secondary source [24] was right. Older community posts quoting 9 h are describing TPU sessions, or a GPU limit that has since been raised [1][5].
- **Idle timeout.** It is 20 min on the Notebooks page [1] and 60 min on the GPU-tips page [2]. Plan for 20.
- **GPU hardware.** The docs pages [1][2] still say P100. The 2026-08-14 announcement [6] and the CLI docs ("Accelerators available as of Sep 2026", no P100) [11] say T4x2. The editor bundle also carries "P100 GPUs are being phased out" [14].
- **Real availability after the P100 retirement.** One community post (2026-09-27, not staff) reports greyed-out accelerator selectors and usable GPU time well below 30 h, which it blames on LLM-hosting tutorials [22]. This is anecdotal (UNVERIFIED), but it is a real schedule risk.
- **"Google AI Pro gives 45 h on Kaggle".** This comes from a community post [23]. The Colab FAQ text it quotes [15] does not mention Kaggle; the Kaggle link is the poster's inference (UNVERIFIED). It is a paid tier in any case, so it is out of scope. Kaggle's documented paid add-on is Colab Pro or Pro+ linking, which adds +15 or +30 h/week as a "limited time promotion" [1].

## What this means for a 13 h training run

The ~13 h figure is a worst-case single-T4 epoch for Qwen3.5-2B on 20k pairs, extrapolated from Unsloth's T4 runs (est., see [candidate-models](candidate-models.md)). It **does not fit** one Kaggle commit (12 h hard cap, and an overrun fails to save [1]). It would also use about 43% of a 30 h week [2].

Options, cheapest first:

1. **Use both T4s.** T4x2 costs the same quota as one GPU [8]. DDP with `accelerate` or `torchrun` across the two T4s should land around **7–8 h per epoch** (est., UNVERIFIED; measure it), which fits one commit with margin. Whether Unsloth's free tier supports multi-GPU is UNVERIFIED. Plain HF `transformers` + `peft` + `accelerate` DDP is the safe path.
2. **Time-boxed, chained commits.** Give the training loop a wall-clock guard, for example stopping at about 11 h. It then writes a full checkpoint to `/kaggle/working` (LoRA adapter, optimizer, scheduler, RNG state, global step) and exits cleanly, because a commit that overruns 12 h does not save [1]. A LoRA checkpoint for a 2B model is roughly tens to a few hundred MB (est.), far under the 20 GB cap [1]. The next run attaches the previous version's output as an input (Add Input, or `kernel_sources` in `kernel-metadata.json` [12]) or pulls it with `kagglehub.notebook_output_download("<user>/<notebook>")` [13], then calls `trainer.train(resume_from_checkpoint=...)`.
3. **Kaggle Persistence ("Files only")** carries `/kaggle/working` across *interactive* sessions [10]. It is best-effort, does not apply to commits [10], and interactive sessions die after 20–60 idle minutes [1][2], so it is unsuited to unattended multi-hour training. Use it for eval and debugging only.
4. **Final artefacts.** Push the trained adapter to a private Kaggle Model (`kagglehub.model_upload` [13]) or a private dataset (≤200 GB total [3]), so it survives notebook edits and is easy to pull to the laptop.
5. **Colab as overflow only.** Colab offers no published hours, no free background execution, and VMs that are deleted when idle [15]. If used, checkpoint to Drive every N steps and call `drive.flush_and_unmount()` before exit [16]. Don't rotate accounts, which Colab explicitly prohibits [15].

Budget note (est.): BLIP-LoRA, the ~2B LoRA, cross-validation folds and robustness evals all draw on one 30 h/week Kaggle pool [2]. Keep at least a week of slack for queueing after the P100 retirement [22]. The quota resets weekly [7], so long runs are best started early in the quota week.

## Open questions

- Can a Kaggle notebook attach **its own** previous version's output as an input, or does chaining need two alternating notebooks or a `kagglehub` dataset upload (which needs API credentials inside the run)? Not stated in [1]: UNVERIFIED.
- Is any `/kaggle/working` output kept when a commit hits the 12 h cap? The docs only say it must finish to save [1]. The guard design assumes nothing is kept.
- Actual T4x2 DDP throughput for Qwen3.5-2B, LFM2.5-VL-1.6B and Gemma 4 E2B. This decides whether option 1 alone removes the need to resume.
- fp16 numerical stability on T4 for bf16-pretrained VLMs (overflow/NaN risk): UNVERIFIED. Unsloth's T4 runs did complete in fp16 ([candidate-models](candidate-models.md)).
- Current weekly reset time and the exact concurrent-GPU-commit cap (unpublished [7][14]).

## Sources

1. Kaggle Docs, Notebooks (Technical Specifications; Save & Run All; Adding Data Sources; Colab Pro promotion): https://www.kaggle.com/docs/notebooks
2. Kaggle Docs, Efficient GPU Usage Tips (30 h/week "or sometimes higher"; 60 min idle): https://www.kaggle.com/docs/efficient-gpu-usage
3. Kaggle Docs, Datasets (Technical Specifications: 200 GB per dataset, 200 GB private, 50 top-level files): https://www.kaggle.com/docs/datasets
4. Kaggle Docs, Models (Creating a Model via kagglehub, CLI, UI): https://www.kaggle.com/docs/models
5. Kaggle Docs, TPUs (20 h/week, 9 h per session): https://www.kaggle.com/docs/tpu
6. Kaggle product announcement, "Sunsetting the NVIDIA Tesla P100 GPU on September 15, 2026" (staff, 2026-08-14; staff replies 2026-08-21 on L4 being competition-only and 2026-09-14 on T4x2 being the only option): https://www.kaggle.com/discussions/product-announcements/735239
7. Kaggle, "[Notebooks update] More GPU hours - Introducing a 'floating' GPU quota" (staff, 2020-08-08; weekly reset Saturday 00:00 UTC): https://www.kaggle.com/product-feedback/173129
8. Kaggle, "[Notebooks update] New GPU (T4s) options & more CPU RAM" (staff, 2022-10-19; staff reply that 1 h of T4x2 = 1 h quota): https://www.kaggle.com/product-feedback/361104
9. Kaggle, "[Notebooks update] GPU Notebooks 4 CPUs & 29 GBs of RAM!" (staff, 2023-10-18): https://www.kaggle.com/discussions/product-feedback/448251
10. Kaggle, "[Notebooks update] Session Persistence for Variables and Files!" (staff, 2022-09-26; staff replies on `/kaggle/working` only and interactive-only): https://www.kaggle.com/product-feedback/355440
11. Kaggle CLI docs, kernels (`--accelerator`, `-t/--timeout`, "Accelerators available as of Sep 2026"): https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels.md
12. Kaggle CLI docs, kernel metadata (`enable_gpu`, `machine_shape`, `kernel_sources`): https://github.com/Kaggle/kaggle-api/blob/main/docs/kernels_metadata.md
13. kagglehub README (`model_upload`, `dataset_upload`, `notebook_output_download`): https://github.com/Kaggle/kagglehub/blob/main/README.md
14. Kaggle notebook-editor bundle, fetched 2026-10-18 (strings: "P100 GPUs are being phased out…", "GPU T4 x2 HighMem", "Max number of running commit GPU sessions is exceeded", persistence modes, "Save output for this version when creating a Quick Save", "Get phone verified"): https://www.kaggle.com/static/assets/KernelEditor.c58bd55e9b3ec87b.js (hashed filename, changes on deploy)
15. Google Colab FAQ (Resource Limits; "Colab does not publish these limits"; 12 h; idle VMs deleted; restricted activities; Google AI plans): https://research.google.com/colaboratory/faq.html
16. colabtools `google/colab/drive.py` (`mount`, `flush_and_unmount`): https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py
17. NVIDIA, CUDA GPU Compute Capability (T4 = 7.5): https://developer.nvidia.com/cuda-gpus
18. NVIDIA, Legacy CUDA GPU Compute Capability (Tesla P100 = 6.0): https://developer.nvidia.com/cuda-legacy-gpus
19. NVIDIA CUDA Programming Guide, Compute Capabilities, Table 33 "Input Data Types Supported by Tensor Core Acceleration" (7.5: FP16/INT8/INT4; BF16 from 8.0): https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html
20. NVIDIA T4 product page (FP32/FP16/INT8/INT4 precisions; no BF16 listed): https://www.nvidia.com/en-us/data-center/tesla-t4/
21. FlashAttention README (FA2 needs Ampere/Ada/Hopper; bf16 needs Ampere+; Turing via the separate flash-attention-turing repo): https://github.com/Dao-AILab/flash-attention/blob/main/README.md
22. Community, not staff: "Tutorial wave is converting Kaggle Notebooks into free public LLM servers - GPU availability is degrading" (2026-09-27): https://www.kaggle.com/discussions/product-feedback/743898
23. Community, not staff: "[PSA] Google AI Pro now gives 45h Kaggle GPU quota" (2026-09-18): https://www.kaggle.com/discussions/general/741987
24. Secondary source behind the original 30 h / 12 h figures: https://pypi.org/p/kgz
