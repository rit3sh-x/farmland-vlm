DATA := data/av/Agriculture-Vision-2021
APP := uv run --extra app --extra dev
TRAIN := uv run --extra train --extra dev
REPORT := uv run --extra report
STATS := data/stats.npz
RUNS ?= runs

.PHONY: help app install web serve dev test install-train data stats smoke pipeline mirror report pdf

help:
	@echo "make app            install, build the UI and serve it on 127.0.0.1:8000"
	@echo "make install        Python environment for the app and tests (uv sync)"
	@echo "make web            build the UI"
	@echo "make serve          start the app (PORT= to change the port)"
	@echo "make dev            Vite dev server with live reload, beside make serve"
	@echo "make test           run the tests"
	@echo "make install-train  Python environment for training (CUDA PyTorch)"
	@echo "make data stats     download the dataset and build the mask statistics"
	@echo "make smoke           a few-minute check of every GPU path before a long run"
	@echo "make pipeline       every experiment in the report (ARGS= for --only, --folds, ...)"
	@echo "make mirror         pull the run records from the Hub runs repo into runs/ (HF_TOKEN)"
	@echo "make report pdf     rebuild the report tables and figures from RUNS=runs, then the PDFs"

app: install web serve

install:
	uv sync --extra app --extra dev

web:
	cd app/frontend && pnpm ci && pnpm run build

serve:
	$(APP) farmvlm-serve --port $(or $(PORT),8000)

dev:
	cd app/frontend && pnpm run dev

test:
	$(APP) pytest -q tests

install-train:
	uv sync --extra train --extra dev --extra report

data:
	$(TRAIN) python scripts/fetch_data.py data/av

stats:
	$(TRAIN) python -m farmvlm.stats --root $(DATA) --manifest data/av/data_manifest.csv --out $(STATS)

smoke:
	$(TRAIN) python scripts/smoke_gpu.py --stats $(STATS) --root $(DATA)

pipeline:
	FARMVLM_HOME=$${FARMVLM_HOME:-$(CURDIR)/work} $(TRAIN) python scripts/pipeline.py --stats $(STATS) --root $(DATA) $(ARGS)

mirror:
	$(REPORT) python scripts/mirror.py

report:
	$(REPORT) python -m farmvlm.report --runs $(RUNS) --out report/tables
	$(REPORT) python scripts/study_table.py --runs $(RUNS) --out report/tables/study.tex
	$(REPORT) python scripts/figures.py --runs $(RUNS) --out report/figures

pdf:
	cd report && tectonic -X compile main.tex && tectonic -X compile deck.tex
