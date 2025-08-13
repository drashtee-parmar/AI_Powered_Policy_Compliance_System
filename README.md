# AI_Powered_Policy_Compliance_System

# installing virtual env
- python -m venv venv
- source venv/bin/activate

# requirement.txt
- after adding those version 
- pip install -r requirements.txt

# Neo4j
- brew services start neo4j
- neo4j start
- neo4j status

# Creating virtual environment
```commandline
python -m venv .venv
source .venv/bin/activate
```

# pip install

pip install openai langchain langchain-openai neo4j fastapi uvicorn[standard] python-dotenv pydantic tiktoken spacy typer rich tqdm numpy scipy soundfile
python -m spacy download en_core_web_sm
pip install python-docx
pip install pypdf
pip install python-docx pypdf spacy neo4j python-dotenv openai
pip install python-docx neo4j python-dotenv openai numpy tiktoken
pip install typer
pip install typer rich click

python -m graph_rag_voice.cli ingest ./policies

# # Graph RAG + Voice Q&A (Neo4j, FAISS, LangChain, OpenAI)

Build once, run three ways: **CLI**, **FastAPI**, **Streamlit**.

## 1) Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env  # add keys/URIs
```

(Optional) start Neo4j in Docker:
```bash
docker compose up -d neo4j
```

## 2) Ingest

```bash
python -m graph_rag_voice.cli ingest ./sample_docs
```

## 3) Ask

```bash
python -m graph_rag_voice.cli ask "What did Sarah say about the compliance deadline?"
python -m graph_rag_voice.cli rebuild_faiss
python -m graph_rag_voice.cli ask_hybrid "What did Sarah say about the compliance deadline?"
```

## 4) API

```bash
uvicorn graph_rag_voice.app:app --reload --port 8000
# POST /ingest {"folder":"./sample_docs"}
# POST /ask (form: question=... or file: audio=...)
```

## 5) Streamlit UI

```bash
streamlit run streamlit_app.py
```

## Notes
- Ingestion supports `.txt/.md/.docx/.pdf` via `graph_rag_voice.loaders`.
- Retrieval is graph-first with optional FAISS hybrid. Configure via `.env`.
- Models: `gpt-4o-mini` (reasoning), `gpt-4o-transcribe` (STT), `text-embedding-3-small` (embeddings).
# ========================

pip install -r requirements.txt
pip list

# ====================
```
Connection to 37b30e3e.databases.neo4j.io port 7687 [tcp/*] succeeded!
# DNS/port reachability
python - <<'PY'
import socket
print(socket.getaddrinfo("37b30e3e.databases.neo4j.io", 7687))
PY

# TCP probe (macOS)
nc -vz 37b30e3e.databases.neo4j.io 7687
```


# Neo4j start/stop
neo4j start
neo4j stop

# check the localhost
http://localhost:7474

# check
```commandline
export NEO4J_URI="bolt://localhost:7687"         # or your Aura URI
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"

python - <<'PY'
import os
from neo4j import GraphDatabase, basic_auth

driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=basic_auth(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
)
with driver.session(database="neo4j") as s:
    print(s.run("RETURN 1 AS ok").single())
driver.close()
PY
```
