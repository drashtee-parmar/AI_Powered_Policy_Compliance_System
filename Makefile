.PHONY: setup run ingest embed test lint format neo4j-up neo4j-init

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

run:
	. .venv/bin/activate && uvicorn src.server:app --host $${SERVER_HOST:-0.0.0.0} --port $${SERVER_PORT:-8000} --reload

ingest:
	. .venv/bin/activate && python scripts/engest_guard.py

embed:
	. .venv/bin/activate && python scripts/ingest_embeddings.py

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && python -m pip install ruff && ruff check .

format:
	. .venv/bin/activate && python -m pip install ruff && ruff format .

neo4j-up:
	docker compose -f infra/docker-compose.yml up -d neo4j

neo4j-init:
	cypher-shell -a $${NEO4J_URI:-neo4j://localhost:7687} -u $${NEO4J_USER:-neo4j} -p $${NEO4J_PASSWORD:-changeme} -f infra/neo4j-init.cypher