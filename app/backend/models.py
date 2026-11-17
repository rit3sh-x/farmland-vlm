from dataclasses import dataclass
from pathlib import Path
from typing import Literal

WEIGHTS = "DarkSlayer16/farmland-vlm"
QWEN_BASE = "unsloth/Qwen3.5-2B-GGUF"
QWEN_FILES = ("Qwen3.5-2B-Q4_K_M.gguf", "mmproj-F16.gguf")


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    about: str
    engine: Literal["blip", "qwen"]
    tuned: bool = True
    ndvi: bool = False


MODELS = {m.id: m for m in (
    Model("blip-lora", "BLIP-LoRA", "fine-tuned · the report's model", "blip"),
    Model("blip-zeroshot", "Zero-shot BLIP", "baseline · no fine-tuning", "blip", tuned=False),
    Model("qwen-rgb", "Qwen3.5-2B LoRA", "fine-tuned · RGB", "qwen"),
    Model("qwen-ndvi", "Qwen3.5-2B LoRA + NDVI", "fine-tuned · RGB + NDVI map", "qwen", ndvi=True),
)}
QWEN_ADAPTERS = [m.id for m in MODELS.values() if m.engine == "qwen"]


def adapters() -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import GatedRepoError
    try:
        return Path(snapshot_download(WEIGHTS))
    except GatedRepoError as e:
        raise RuntimeError(f"accept the terms at https://huggingface.co/{WEIGHTS}, then run `hf auth login`") from e


def qwen_base() -> tuple[Path, Path]:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(QWEN_BASE, QWEN_FILES[0])), Path(hf_hub_download(QWEN_BASE, QWEN_FILES[1]))
