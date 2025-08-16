# qa.py
# Complete RAG pipeline with safe Neo4j optional step.
from __future__ import annotations

import os
import io
import json
import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ConfigurationError

# Optional graph utils — safe wrapper (see neo4j_utils.py you installed)
try:
    from neo4j_utils import Neo4jClient  # harmless import if not used
except Exception:  # pragma: no cover
    Neo4jClient = None  # type: ignore

# ------------------------
# Config & lazy singletons
# ------------------------

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")

INDEX_PATH = os.getenv("INDEX_PATH", "vector.index")
META_PATH = os.getenv("META_PATH", "meta.pkl")

# Retrieval config
TOP_K = int(os.getenv("TOP_K", "5"))
NORMALIZE = True  # cosine via IP on L2-normalized vectors

_oai: Optional[OpenAI] = None
_index: Optional[faiss.Index] = None
_meta: Optional[List[Dict[str, Any]]] = None


def _oai_client() -> OpenAI:
    global _oai
    if _oai is None:
        _oai = OpenAI()  # reads OPENAI_API_KEY from env
    return _oai


def _load_index_and_meta() -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """Load FAISS index + meta only once."""
    global _index, _meta
    if _index is None or _meta is None:
        print("[QA] Loading FAISS index and metadata ...")
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"[QA][ERROR] Missing FAISS index: {INDEX_PATH}. "
                "Run `python ingest.py` first."
            )
        if not os.path.exists(META_PATH):
            raise FileNotFoundError(
                f"[QA][ERROR] Missing metadata file: {META_PATH}. "
                "Run `python ingest.py` first."
            )
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            _meta = pickle.load(f)
        print(f"[QA] Index ready. Chunks: {len(_meta)}")
    # mypy: _index/_meta are now not None
    return _index, _meta  # type: ignore[return-value]


# ------------
# Embeddings
# ------------

def _embed(texts: List[str]) -> np.ndarray:
    """OpenAI Embeddings -> (N, D) float32."""
    if not texts:
        return np.zeros((0, 1536), dtype="float32")

    client = _oai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.asarray([d.embedding for d in resp.data], dtype="float32")
    if NORMALIZE and vecs.size:
        faiss.normalize_L2(vecs)
    return vecs


def _embed_one(text: str) -> np.ndarray:
    v = _embed([text])
    return v[0] if v.shape[0] else np.zeros((1536,), dtype="float32")


# ------------
# Retrieval
# ------------

def _retrieve(query: str, k: int = TOP_K) -> Tuple[List[Dict[str, Any]], List[int], List[float]]:
    index, meta = _load_index_and_meta()
    qv = _embed_one(query).reshape(1, -1)
    D, I = index.search(qv, k)
    I = I[0].tolist()
    D = D[0].tolist()
    hits: List[Dict[str, Any]] = []
    for rank, (idx, score) in enumerate(zip(I, D), 1):
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        # Truncate excerpt for UI
        excerpt = m.get("text", "")
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "…"
        hits.append(
            {
                "rank": rank,
                "file": m.get("file", "?"),
                "id": m.get("id", ""),
                "score": round(float(score), 4),
                "excerpt": excerpt,
            }
        )
    print(f"[QA] Retrieved {len(hits)} vector hits.")
    return hits, I, D


# ------------
# Transcribe (audio -> text)
# ------------

def _transcribe(audio_path: str) -> str:
    client = _oai_client()
    with open(audio_path, "rb") as f:
        # OpenAI Whisper API
        tr = client.audio.transcriptions.create(model=WHISPER_MODEL, file=f)
    text = (tr.text or "").strip()
    return text


# ------------
# Synthesis (LLM)
# ------------

SYSTEM_PROMPT = """You are a helpful assistant answering policy questions.
Use the provided context chunks verbatim when citing facts. Be concise, precise,
and do not invent facts not present in the context.
If the question cannot be answered from context, say that briefly and suggest next steps.
Return a clear, user-friendly answer (no Markdown tables unless necessary).
"""

def _build_context(hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for h in hits:
        blocks.append(
            f"[{h['rank']}] file={h['file']} score={h['score']} id={h['id']}\n{h['excerpt']}"
        )
    return "\n\n---\n\n".join(blocks)


def _synthesize_answer(query: str, hits: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    client = _oai_client()
    context = _build_context(hits)

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Context (top retrieved chunks):\n{context}\n\n"
        "Write a direct answer first. If you used specific facts from the context, cite them inline like [1], [2] etc."
    )

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    text = (resp.choices[0].message.content or "").strip()

    # A tiny model_json for your expander
    model_json = {
        "facts": [f"Used {len(hits)} retrieved chunks."],
        "confidence": 0.65 + min(0.3, 0.05 * len(hits)),  # toy heuristic
    }
    return text, model_json


# ------------
# Optional graph step (safe)
# ------------

def graph_expand(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional Neo4j step. If Neo4j is unavailable, we add a warning and continue.
    """
    if Neo4jClient is None:
        return ctx

    try:
        n4j = Neo4jClient(verbose=False)
    except ServiceUnavailable as e:
        ctx.setdefault("warnings", []).append(str(e))
        return ctx
    except Exception as e:  # any other driver issue
        ctx.setdefault("warnings", []).append(f"[NEO4J] {e}")
        return ctx

    try:
        # Example no-op that touches the DB (replace with your real graph logic)
        with n4j.session() as s:
            _ = s.run("RETURN 1 AS ok").single()
        return ctx
    finally:
        n4j.close()


# ------------
# Public entry
# ------------

@dataclass
class QAResponse:
    query: str
    unstructured: str
    citations: List[Dict[str, Any]]
    model_json: Dict[str, Any]
    warnings: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "unstructured": self.unstructured,
            "structured": {
                "citations": self.citations,
                "model_json": self.model_json,
            },
            "warnings": self.warnings,
        }


def answer_query(
    query: Optional[str] = None,
    audio_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    One public function used by Streamlit.
    Provide either `query` *or* `audio_path`.
    """
    if not (query or audio_path):
        raise ValueError("Provide either `query` text or `audio_path`.")

    # A context dict we can pass through the pipeline
    ctx: Dict[str, Any] = {"warnings": []}

    # 1) Transcribe if needed
    detected_text = None
    if audio_path:
        try:
            detected_text = _transcribe(audio_path)
        except Exception as e:
            ctx["warnings"].append(f"[Transcribe] {e}")
            detected_text = ""

    the_query = (query or detected_text or "").strip()
    if not the_query:
        return QAResponse(
            query="",
            unstructured="I couldn’t detect any speech or text.",
            citations=[],
            model_json={"facts": [], "confidence": 0.0},
            warnings=ctx["warnings"],
        ).as_dict()

    print(f"[QA] Query: {the_query}")

    # 2) Retrieve
    try:
        hits, I, D = _retrieve(the_query, k=TOP_K)
    except Exception as e:
        # If retrieval fails, return a graceful message
        return QAResponse(
            query=the_query,
            unstructured=f"Retrieval error: {e}",
            citations=[],
            model_json={"facts": [], "confidence": 0.0},
            warnings=ctx["warnings"],
        ).as_dict()

    citations = hits  # already normalized for UI

    # 3) Synthesize
    try:
        final_text, model_json = _synthesize_answer(the_query, hits)
    except Exception as e:
        final_text = f"Generation error: {e}"
        model_json = {"facts": [], "confidence": 0.0}

    # 4) Optional graph expansion (never crashes)
    ctx = graph_expand(ctx)

    # 5) Build response
    resp = QAResponse(
        query=the_query,
        unstructured=final_text,
        citations=citations,
        model_json=model_json,
        warnings=ctx["warnings"],
    ).as_dict()

    return resp