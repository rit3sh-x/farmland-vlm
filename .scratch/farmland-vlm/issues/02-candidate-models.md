# Candidate models for the served model

Type: research
Status: resolved
Blocked by: —

## Question

Which 2–3 models should be shortlisted as the served model, next to the BLIP baseline?

Constraints: LoRA/QLoRA fine-tune on one free T4 or P100 (16 GB) within Kaggle session limits; inference on a laptop CPU with ~8 GB RAM at a usable latency (roughly under 10 s per answer); must both caption and answer free-form questions about an aerial farmland image.

Cover, as of late 2026:

- Small general VLMs: Florence-2, SmolVLM / SmolVLM2, Moondream, PaliGemma 2, small Qwen-VL and InternVL variants, LLaVA-OneVision-0.5B, and anything newer.
- Remote-sensing VLMs: GeoChat and its successors.
- JEPA-based options (I-JEPA, V-JEPA 2 and later): what it takes to pair a JEPA encoder with a language model for captioning/VQA, and whether that's viable solo in 7 weeks.

For each: parameter count, license, captioning + VQA ability, Hugging Face `peft` support, any reported remote-sensing or agriculture results, fits-T4 and fits-CPU verdicts. Also: whether PEFT methods beyond plain LoRA (QLoRA, DoRA) matter here.

## Answer

Shortlist, in order: **Qwen3.5-2B** (fallback 0.8B), **LFM2.5-VL-1.6B** (fast variant 450M), **Gemma 4 E2B**.
All three caption and answer questions. Each trains on a T4 with plain 16-bit LoRA (measured T4 peaks 2.4–10.8 GB) and should run on the 8 GB CPU as a Q4 GGUF (est. 2–9 s; laptop not measured yet).
QLoRA only matters for 4B+ models. DoRA is an optional one-flag ablation.
Remote-sensing VLMs (GeoChat, EarthDial) are too heavy for the CPU and score no better than small general VLMs on agricultural benchmarks.
A JEPA encoder plus an LM is not viable as the served model in 7 weeks; at most, add a frozen AnySat (RGB+NIR) probe row.
[findings](../research/candidate-models.md)
