#!/bin/bash
# Rebuild the report and the deck from the Hub's run records: mirror, tables and macros, figures, PDFs.
# Needs HF_TOKEN in the environment and Tectonic on PATH (~/.cache/farmvlm/bin on the laptop).
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
$PY scripts/mirror.py
$PY -m farmvlm.report
$PY scripts/figures.py
cd report && tectonic -X compile main.tex && tectonic -X compile deck.tex
grep -c "providecommand" tables/numbers.tex | xargs -I{} echo "{} macros still without a record (printed as ??)"; grep -c "pending{" main.tex | xargs -I{} echo "{} pending notes in main.tex"
