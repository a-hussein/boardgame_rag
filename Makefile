# Python tooling
PYTHON=uv run

.PHONY: fmt lint test data index eval api run demo

#####

fmt:
uv run ruff check --select I --fix .
uv run black .
uv run ruff check --fix .

lint:
uv run ruff check .
uv run mypy src || true

test:
uv run pytest -q --maxfail=1 --disable-warnings

# 1) Generate synthetic corpus (+ optional BGG augment)
data:
$(PYTHON) python -m boardgame_rag.data_gen --out data/raw/corpus.jsonl --n 500
# Optional: add --bgg-csv data/raw/bgg_sample.csv to augment
$(PYTHON) python -m boardgame_rag.data_gen --mk-processed data/raw/corpus.jsonl data/processed/corpus.parquet

# 2) Build indices (BM25 + FAISS)
index:
$(PYTHON) python -m boardgame_rag.index_build --in data/processed/corpus.parquet --out-dir indices

# 3) Evaluate retrieval
eval:
$(PYTHON) python -m boardgame_rag.eval_harness --gold eval/gold.jsonl --indices indices --report eval/report.md

# 4) API
api:
$(PYTHON) uvicorn boardgame_rag.api:app --reload

# 5) Simple demo CLI
run:
$(PYTHON) python -m boardgame_rag.eval_harness --demo