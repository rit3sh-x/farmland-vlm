"""Run storage in a private HF Hub repo, so a recycled Colab VM loses nothing (issues/15, "Weights").

The token comes only from the HF_TOKEN environment variable (passed with `colab exec --env`); without
it every call is a no-op and runs stay on the VM.
"""
import os
import time
from pathlib import Path

REPO = os.environ.get("FARMVLM_HF_REPO", "DarkSlayer16/farmvlm-runs")
RUNS = Path(os.environ.get("FARMVLM_RUNS", "/content/runs"))


def name_of(path: Path) -> str:
    """A run's Hub name: its path under RUNS (e.g. sweep-vqa/r16-lr0.0001-attn_qv).

    Symlinks are not followed: a pulled run is a link into RUNS/.hub, and its name is still the link's.
    """
    try:
        return str(Path(path).absolute().relative_to(RUNS.absolute()))
    except ValueError:
        return Path(path).name


def enabled() -> bool:
    return bool(os.environ.get("HF_TOKEN"))


def _api():
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(REPO, private=True, exist_ok=True)
    return api


def push(local: Path, name: str, include=("*.json", "*.jsonl", "*.safetensors", "*.pt", "*.gguf", "*.npz", "*.txt", "*.model")):
    """Upload a run folder; retried, since several VMs (and Kaggle's two GPUs) commit to the one repo at once."""
    if not enabled() or not Path(local).exists():
        return
    for attempt in range(4):
        try:
            _api().upload_folder(folder_path=str(local), path_in_repo=f"runs/{name}", repo_id=REPO,
                                 allow_patterns=[f"**/{p}" for p in include] + list(include), commit_message=f"update {name}")
            return
        except Exception as e:
            if attempt == 3:
                raise
            print(f"hub push of {name} failed ({type(e).__name__}); retrying", flush=True)
            time.sleep(30 * (attempt + 1))


def has(name: str, file: str = "done.json") -> bool:
    if not enabled():
        return False
    from huggingface_hub import file_exists
    return file_exists(REPO, f"runs/{name}/{file}")


def pull(name: str, root: Path) -> Path:
    """Download runs/<name>/ into root/<name>/ and return that path."""
    if enabled():
        from huggingface_hub import snapshot_download
        tmp = snapshot_download(REPO, allow_patterns=[f"runs/{name}/*", f"runs/{name}/**/*"], local_dir=str(root / ".hub"))
        src = Path(tmp) / "runs" / name
        if src.exists():
            dst = root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(src.resolve(), target_is_directory=True)
    return root / name
