"""Runs inside the Colab VM: is the pipeline (or its setup shell) still running, and what did it print last."""
import pathlib
import subprocess

running = subprocess.run(["pgrep", "-f", "scripts/pipeline.py"], capture_output=True).returncode == 0
print("PIPELINE_RUNNING" if running else "PIPELINE_EXITED")
print("GPU (util, memory):", subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                                               "--format=csv,noheader"], capture_output=True, text=True).stdout.strip())
for name in ("setup.out", "pipeline.out"):
    p = pathlib.Path("/content") / name
    print(f"--- {name}:", p.read_text()[-1500:] if p.exists() else "missing")
