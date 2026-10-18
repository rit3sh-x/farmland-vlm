# Kaggle and Colab free GPU limits

Type: research
Status: resolved
Blocked by: —

## Question

What official free-tier limits on Kaggle and Colab, as of late 2026, bound training and evaluation runs?

- GPU types offered, weekly GPU hours, max session length, idle timeout, and how committed / background runs ("Save & Run All") behave.
- RAM, disk (`/kaggle/working`, input dataset size limits), and what persists between sessions (outputs, datasets, model uploads), so a run can resume from a checkpoint.
- Kaggle's T4 vs T4×2 vs P100 choice, and bf16 / FlashAttention support on those GPUs.

## Answer

- Kaggle free means **GPU T4 x2 only** (2×16 GB). The P100 was retired on 2026-09-15, and L4/A100/H100 are competition- or admin-only. **30 h/week** (floating, sometimes more) and **12 h/session** are both confirmed in Kaggle's own docs. A T4x2 hour costs 1 quota hour.
- A commit (Save & Run All) must finish within 12 h to save. Interactive sessions time out after 20 or 60 idle minutes (Kaggle's two pages conflict). A GPU session has 29 GB RAM and 20 GB of saved `/kaggle/working`; version outputs can be reused as inputs, and datasets are capped at 200 GB.
- Colab still deliberately publishes no limits beyond "at most 12 h depending on availability". The free tier has no background execution, so treat Colab as overflow and checkpoint to Drive.
- The T4 is CC 7.5: no bf16 and no FlashAttention-2, so train in fp16 with SDPA. A 13 h epoch needs either DDP across both T4s (est. 7–8 h) or time-boxed commits chained via saved checkpoints.

[findings](../research/free-gpu-limits.md)
