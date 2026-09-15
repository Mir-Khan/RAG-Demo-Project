# Convenience targets. On Windows without `make`, run the underlying commands directly
# (each recipe is a one-liner) or use `py -3.13 -m docqa.ingestion.pipeline`.

.PHONY: db-up db-down db-logs ingest ingest-fresh test lint app

db-up:            ## start Postgres+pgvector and wait for health
	docker compose up -d --wait

db-down:          ## stop Postgres (keeps the volume)
	docker compose down

db-nuke:          ## stop Postgres and delete all data
	docker compose down -v

ingest:           ## run ingestion for $CORPUS (incremental upsert)
	python -m docqa.ingestion.pipeline

ingest-fresh:     ## wipe the corpus rows first, then ingest
	python -m docqa.ingestion.pipeline --fresh

test:
	pytest -q

lint:
	ruff check src tests

app:
	streamlit run src/docqa/app/main.py
