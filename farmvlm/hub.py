"""Run storage in a private HF Hub repo, so a recycled Colab VM loses nothing (issues/15, "Weights").

The token comes only from the HF_TOKEN environment variable (passed with `colab exec --env`); without
it every call is a no-op and runs stay on the VM.
"""
import os
import time
from pathlib import Path

REPO = os.environ.get("FARMVLM_HF_REPO", "DarkSlayer16/farmvlm-runs")
HOME = Path(os.environ.get("FARMVLM_HOME", "/content"))  # the VM's working root; svm-ws has no /content
RUNS = HOME / "runs"


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


def _retried(what: str, fn, tries: int = 6):
    """fn(), retried with backoff: several VMs (and Kaggle's two GPUs) commit to the one repo at once, and downloads
    to svm-ws stall now and then (a stalled pull raised ReadTimeout and ended a pipeline; each retry resumes the file)."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"hub {what} failed ({type(e).__name__}); retrying", flush=True)
            time.sleep(20 * (attempt + 1))


def push(local: Path, name: str, include=("*.json", "*.jsonl", "*.safetensors", "*.pt", "*.gguf", "*.npz", "*.txt", "*.model")):
    """Upload a run folder."""
    if not enabled() or not Path(local).exists():
        return
    _retried(f"push of {name}", lambda: _api().upload_folder(
        folder_path=str(local), path_in_repo=f"runs/{name}", repo_id=REPO,
        allow_patterns=[f"**/{p}" for p in include] + list(include), commit_message=f"update {name}"))


def has(name: str, file: str = "done.json") -> bool:
    if not enabled():
        return False
    from huggingface_hub import file_exists
    return _retried(f"check of {name}/{file}", lambda: file_exists(REPO, f"runs/{name}/{file}"))


def pull(name: str, root: Path, last: bool = True) -> Path:
    """Download runs/<name>/ into root/<name>/ and return that path; last=False skips last/ (resume state a finished run
    never needs: ~95 MB of optimizer state per BLIP run)."""
    if enabled():
        from huggingface_hub import snapshot_download
        tmp = _retried(f"pull of {name}", lambda: snapshot_download(
            REPO, allow_patterns=[f"runs/{name}/*", f"runs/{name}/**/*"], local_dir=str(root / ".hub"),
            ignore_patterns=None if last else [f"runs/{name}/last/*"]))
        src = Path(tmp) / "runs" / name
        if src.exists():
            dst = root / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                dst.symlink_to(src.resolve(), target_is_directory=True)
    return root / name
