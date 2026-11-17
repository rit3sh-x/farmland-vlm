"""`farmvlm-serve`: the demo backend and the built web UI on one port. The app lives in app/ beside the package,
so this needs the editable install that `uv sync` makes."""
import argparse
import sys
from pathlib import Path


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    uvicorn.run("app.backend.main:app", host=a.host, port=a.port)
