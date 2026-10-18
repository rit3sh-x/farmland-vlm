"""Where the demo finds its binaries, models and data on the laptop. Override any path with an env var."""
import os
from pathlib import Path

HOME = Path(os.environ.get("FARMVLM_HOME", Path.home() / ".cache" / "farmvlm"))
REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("FARMVLM_DATA", REPO / "data"))

LLAMA_SERVER = Path(os.environ.get("FARMVLM_LLAMA_SERVER", HOME / "llama-b11719" / "llama-server"))
LLAMA_PORT = int(os.environ.get("FARMVLM_LLAMA_PORT", "8091"))
LLAMA_THREADS = int(os.environ.get("FARMVLM_THREADS", "8"))
SERVED_KEY = os.environ.get("FARMVLM_SERVED", "lfm-450m")
SERVED_GGUF = Path(os.environ.get("FARMVLM_GGUF", HOME / "models" / f"{SERVED_KEY}.gguf"))
SERVED_MMPROJ = Path(os.environ.get("FARMVLM_MMPROJ", HOME / "models" / f"{SERVED_KEY}-mmproj.gguf"))
ADAPTERS = {  # model id in the API -> runtime LoRA GGUF; a missing file means "serve the base model"
    "ft": Path(os.environ.get("FARMVLM_LORA_RGB", HOME / "adapters" / "rgb-lora-f16.gguf")),
    "ndvi": Path(os.environ.get("FARMVLM_LORA_NDVI", HOME / "adapters" / "ndvi-lora-f16.gguf")),
}
BLIP_VQA_ADAPTER = os.environ.get("FARMVLM_BLIP_VQA_ADAPTER")      # set only in the BLIP-LoRA fallback
BLIP_CAPTION_ADAPTER = os.environ.get("FARMVLM_BLIP_CAPTION_ADAPTER")

DEMO_TILES = DATA / "demo_tiles"     # RGB + NIR JPEGs and truth.json for fold 0's test-sample tiles (never committed)
FIELD_CACHE = DATA / "field_cache"
SHOWCASE = Path(__file__).with_name("showcase.json")
UPLOADS_KEPT = 16
EXTERNAL_TIMEOUT = 20.0
USER_AGENT = "farmland-vlm/0.1"
