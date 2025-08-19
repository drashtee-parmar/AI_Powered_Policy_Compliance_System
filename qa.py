# # qa.py
# import os
# import json
# import pickle
# import re
# from typing import Dict, Any, List, Tuple

# import faiss
# import numpy as np
# from dotenv import load_dotenv
# from openai import OpenAI

# from neo4j_utils import Neo4jClient

# # ----------------------
# # Config
# # ----------------------
# INDEX_PATH = "vector.index"
# META_PATH = "meta.pkl"

# TOP_K = 5                   # vector hits to fetch
# GRAPH_EXPAND_PER_DOC = 1    # extra chunks per doc via Neo4j
# UI_CITATION_MAX = 3         # show at most this many unique citations by file


# # ----------------------
# # ASCII normalization
# # ----------------------
# _ASCII_MAP = {
#     "\u2018": "'",  # left single quote
#     "\u2019": "'",  # right single quote
#     "\u201A": "'",  # single low-9
#     "\u201B": "'",  # single high-reversed-9
#     "\u201C": '"',  # left double quote
#     "\u201D": '"',  # right double quote
#     "\u201E": '"',  # double low-9
#     "\u2032": "'",  # prime
#     "\u2033": '"',  # double prime
#     "\u2013": "-",  # en dash
#     "\u2014": "-",  # em dash
#     "\u2212": "-",  # minus sign
#     "\u00A0": " ",  # no-break space
#     "\u2026": "...",# ellipsis
# }

# def to_ascii(s: str) -> str:
#     if not isinstance(s, str):
#         return s
#     return s.translate(str.maketrans(_ASCII_MAP))

# def to_ascii_deep(obj):
#     if isinstance(obj, str):
#         return to_ascii(obj)
#     if isinstance(obj, list):
#         return [to_ascii_deep(x) for x in obj]
#     if isinstance(obj, dict):
#         return {k: to_ascii_deep(v) for k, v in obj.items()}
#     return obj

# # ---- NEW: final answer cleanup to prevent spacing/markdown glitches ----
# _NUM_COMMA = re.compile(r"(?<=\d)\s*,\s*(?=\d)")      # "1 , 000" -> "1,000"
# _SPACE_BEFORE_PCT = re.compile(r"\s+%")               # "3 %" -> "3%"
# _SPACE_AROUND_PUNCT = re.compile(r"\s+([,.;:!?])")    # "word ," -> "word,"
# _MULTI_SPACE = re.compile(r"\s+")
# _MD_META = re.compile(r"[*_`~]")                      # markdown metachars

# def postprocess_answer(text: str) -> str:
#     """Normalize whitespace/punctuation and neutralize markdown."""
#     if not text:
#         return text
#     t = to_ascii(text)

#     # collapse weird spacing
#     t = _NUM_COMMA.sub(",", t)
#     t = _SPACE_BEFORE_PCT.sub("%", t)
#     t = _SPACE_AROUND_PUNCT.sub(r"\1", t)
#     t = _MULTI_SPACE.sub(" ", t).strip()

#     # neutralize markdown metacharacters so chat renderers don't italicize, etc.
#     # (we'll show it as plain text in Streamlit, but this is harmless + defensive)
#     t = _MD_META.sub(lambda m: "\\" + m.group(0), t)
#     return t


# # ----------------------
# # Index & embeddings
# # ----------------------
# def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
#     print("[QA] Loading FAISS index and metadata ...", flush=True)
#     index = faiss.read_index(INDEX_PATH)
#     with open(META_PATH, "rb") as f:
#         meta = pickle.load(f)
#     print(f"[QA] Index ready. Chunks: {len(meta)}", flush=True)
#     return index, meta

# def embed_query(oai: OpenAI, q: str, model: str) -> np.ndarray:
#     emb = oai.embeddings.create(model=model, input=[q]).data[0].embedding
#     v = np.array(emb, dtype="float32").reshape(1, -1)
#     faiss.normalize_L2(v)
#     return v

# def retrieve(oai: OpenAI, query: str, index, meta: List[Dict], embed_model: str) -> List[Dict]:
#     v = embed_query(oai, query, embed_model)
#     sims, idxs = index.search(v, TOP_K)
#     sims = sims[0].tolist()
#     idxs = idxs[0].tolist()
#     print(f"[QA] Retrieved {TOP_K} vector hits.", flush=True)
#     return [
#         {"score": float(s), "file": meta[i]["file"], "text": meta[i]["text"], "id": meta[i]["id"]}
#         for s, i in zip(sims, idxs)
#     ]

# # ----------------------
# # Graph expansion (Neo4j)
# # ----------------------
# def graph_expand(contexts: List[Dict]) -> List[Dict]:
#     if not contexts:
#         return contexts
#     n4j = Neo4jClient()
#     files = list({c["file"] for c in contexts})
#     extra: List[Dict] = []
#     for f in files:
#         extra.extend(n4j.expand_related_chunks(f, limit=GRAPH_EXPAND_PER_DOC))
#     n4j.close()

#     have = {c.get("id") for c in contexts}
#     added = 0
#     for e in extra:
#         if e["id"] not in have:
#             e["score"] = 0.05
#             contexts.append(e)
#             added += 1
#     if added:
#         print(f"[QA] Graph expansion added {added} chunk(s).", flush=True)
#     return contexts

# # ----------------------
# # Citations (clean & de-duplicated)
# # ----------------------
# def build_clean_citations(contexts: List[Dict], limit: int = UI_CITATION_MAX) -> List[Dict]]:
#     best_per_file: Dict[str, Dict] = {}
#     for c in contexts:
#         f = c.get("file", "?")
#         if f not in best_per_file or c.get("score", 0.0) > best_per_file[f].get("score", 0.0):
#             best_per_file[f] = c
#     best = sorted(best_per_file.values(), key=lambda x: x.get("score", 0.0), reverse=True)[:limit]
#     cleaned = []
#     for c in best:
#         text = c.get("text", "")
#         cleaned.append({
#             "file": to_ascii(c.get("file", "?")),
#             "excerpt": to_ascii(text[:160] + ("..." if len(text) > 160 else "")),
#             "score": round(float(c.get("score", 0.0)), 4),
#         })
#     return cleaned

# # ----------------------
# # LLM synthesis + parsing
# # ----------------------
# def synthesize_answer(oai: OpenAI, query: str, contexts: List[Dict], model: str) -> Dict[str, Any]:
#     print("[QA] Synthesizing final answer ...", flush=True)
#     contexts = sorted(contexts, key=lambda x: x.get("score", 0.0), reverse=True)
#     citations_for_ui = build_clean_citations(contexts, limit=UI_CITATION_MAX)
#     top_contexts = contexts[:8]
#     context_block = "\n\n---\n\n".join(
#         [f"Source {i+1} ({c.get('file','?')}):\n{c.get('text','')}" for i, c in enumerate(top_contexts)]
#     )
#     prompt = (
#         "You are a precise support assistant. Use only the sources to answer.\n\n"
#         "Question:\n{q}\n\n"
#         "Sources:\n{sources}\n\n"
#         "Return output in EXACTLY two fenced blocks, in this order:\n\n"
#         "```text\n"
#         "<concise friendly answer, <=120 words>\n"
#         "```\n\n"
#         "```json\n"
#         "{{\"facts\": [\"...\",\"...\"], \"citations\": [1,2], \"confidence\": 0.0}}\n"
#         "```\n\n"
#         "Use 1-based indices for \"citations\" matching the numbered Sources above.\n"
#         "If information is missing, state that in the text and keep JSON consistent.\n"
#         "Use plain ASCII punctuation (single quotes, hyphens). Do not use curly quotes or long dashes."
#     ).format(q=query, sources=context_block)

#     resp = oai.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.2,
#     )
#     raw = resp.choices[0].message.content.strip()

#     # Parse fenced code blocks
#     matches = re.findall(r"```(text|json)\s*([\s\S]*?)```", raw, flags=re.I)
#     text_block, json_block = None, None
#     for lang, body in matches:
#         body = body.strip()
#         if lang.lower() == "text":
#             text_block = body
#         elif lang.lower() == "json":
#             json_block = body

#     parsed_json: Dict[str, Any] = {}
#     if json_block:
#         try:
#             parsed_json = json.loads(json_block)
#         except json.JSONDecodeError:
#             parsed_json = {"_parse_error": True, "raw": json_block}

#     # ---- sanitize final text & json ----
#     text_block = postprocess_answer(text_block or raw)
#     parsed_json = to_ascii_deep(parsed_json)

#     return {"text": text_block, "json": parsed_json, "citations": citations_for_ui}

# # ----------------------
# # Audio (Whisper) STT
# # ----------------------
# def transcribe_audio(oai: OpenAI, audio_path: str, whisper_model: str) -> str:
#     print(f"[QA] Transcribing audio: {audio_path}", flush=True)
#     with open(audio_path, "rb") as f:
#         tr = oai.audio.transcriptions.create(model=whisper_model, file=f)
#     print("[QA] Transcription complete.", flush=True)
#     return postprocess_answer(tr.text.strip())

# # ----------------------
# # Public entry point
# # ----------------------
# def answer_query(query: str = None, audio_path: str = None) -> Dict[str, Any]:
#     load_dotenv()
#     oai = OpenAI()
#     embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
#     chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
#     whisper = os.getenv("WHISPER_MODEL", "whisper-1")

#     if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
#         raise RuntimeError("Index not found. Run: python ingest.py")

#     index, meta = load_index()

#     if audio_path and not query:
#         query = transcribe_audio(oai, audio_path, whisper)
#     elif not query:
#         raise ValueError("Provide either `query` text or `audio_path`.")
#     query = postprocess_answer(query)

#     print(f"[QA] Query: {query}", flush=True)

#     ctx = retrieve(oai, query, index, meta, embed_model)
#     ctx = graph_expand(ctx)

#     out = synthesize_answer(oai, query, ctx, chat_model)
#     print("[QA] Done.", flush=True)

#     return {
#         "query": query,
#         "structured": {
#             "citations": out["citations"],
#             "model_json": out["json"],
#         },
#         "unstructured": out["text"],
#     }

# if __name__ == "__main__":
#     res = answer_query(query="Do foreign transactions incur a fee, and how much?")
#     print(json.dumps(res, indent=2, ensure_ascii=True))


# qa.py
import os
import json
import pickle
import re
from typing import Dict, Any, List, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

from neo4j_utils import Neo4jClient

# ----------------------
# Config
# ----------------------
INDEX_PATH = "vector.index"
META_PATH = "meta.pkl"

TOP_K = 5                   # vector hits to fetch
GRAPH_EXPAND_PER_DOC = 1    # extra chunks per doc via Neo4j
UI_CITATION_MAX = 3         # show at most this many unique citations by file


# ----------------------
# Index & embeddings
# ----------------------
def load_index() -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    print("[QA] Loading FAISS index and metadata ...", flush=True)
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "rb") as f:
        meta = pickle.load(f)
    print(f"[QA] Index ready. Chunks: {len(meta)}", flush=True)
    return index, meta


def embed_query(oai: OpenAI, q: str, model: str) -> np.ndarray:
    emb = oai.embeddings.create(model=model, input=[q]).data[0].embedding
    v = np.array(emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(v)
    return v


def retrieve(oai: OpenAI, query: str, index, meta: List[Dict], embed_model: str) -> List[Dict]:
    v = embed_query(oai, query, embed_model)
    sims, idxs = index.search(v, TOP_K)
    sims = sims[0].tolist()
    idxs = idxs[0].tolist()
    print(f"[QA] Retrieved {TOP_K} vector hits.", flush=True)
    return [
        {"score": float(s), "file": meta[i]["file"], "text": meta[i]["text"], "id": meta[i]["id"]}
        for s, i in zip(sims, idxs)
    ]


# ----------------------
# Graph expansion (Neo4j)
# ----------------------
# def graph_expand(contexts: List[Dict]) -> List[Dict]:
#     """
#     Pull a few additional chunks from the same docs to improve recall.
#     De-duplicate by chunk id.
#     """
#     if not contexts:
#         return contexts

#     n4j = Neo4jClient()
#     files = list({c["file"] for c in contexts})
#     extra: List[Dict] = []
#     for f in files:
#         extra.extend(n4j.expand_related_chunks(f, limit=GRAPH_EXPAND_PER_DOC))
#     n4j.close()

#     have = {c.get("id") for c in contexts}
#     added = 0
#     for e in extra:
#         if e["id"] not in have:
#             e["score"] = 0.05  # small baseline score for graph-expanded chunks
#             contexts.append(e)
#             added += 1
#     if added:
#         print(f"[QA] Graph expansion added {added} chunk(s).", flush=True)
#     return contexts

def graph_expand(contexts: List[Dict]) -> List[Dict]:
    """
    Pull a few additional chunks from the same docs to improve recall.
    De-duplicate by chunk id. If Neo4j is unavailable, continue silently.
    """
    if not contexts:
        return contexts

    try:
        n4j = Neo4jClient()
        files = list({c["file"] for c in contexts})
        extra: List[Dict] = []
        for f in files:
            extra.extend(n4j.expand_related_chunks(f, limit=GRAPH_EXPAND_PER_DOC))
        n4j.close()

        have = {c.get("id") for c in contexts}
        added = 0
        for e in extra:
            if e.get("id") not in have:
                e["score"] = 0.05  # small baseline score for graph-expanded chunks
                contexts.append(e)
                added += 1
        if added:
            print(f"[QA] Graph expansion added {added} chunk(s).", flush=True)
    except Exception as e:
        print(f"[QA][WARN] Graph expansion skipped: {e}", flush=True)

    return contexts

# ----------------------
# Citations (clean & de-duplicated)
# ----------------------
def build_clean_citations(contexts: List[Dict], limit: int = UI_CITATION_MAX) -> List[Dict]:
    """
    Deduplicate by file; keep the best-scoring snippet per file.
    Return top N by score.
    """
    best_per_file: Dict[str, Dict] = {}
    for c in contexts:
        f = c.get("file", "?")
        if f not in best_per_file or c.get("score", 0.0) > best_per_file[f].get("score", 0.0):
            best_per_file[f] = c

    best = sorted(best_per_file.values(), key=lambda x: x.get("score", 0.0), reverse=True)[:limit]

    cleaned = []
    for c in best:
        text = c.get("text", "")
        cleaned.append({
            "file": c.get("file", "?"),
            "excerpt": (text[:160] + ("..." if len(text) > 160 else "")),
            "score": round(float(c.get("score", 0.0)), 4),
        })
    return cleaned


# ----------------------
# LLM synthesis + parsing
# ----------------------
def synthesize_answer(oai: OpenAI, query: str, contexts: List[Dict], model: str) -> Dict[str, Any]:
    """
    Ask the model for two fenced blocks (```text``` and ```json```) and parse them
    into a clean, structured return object.
    """
    print("[QA] Synthesizing final answer ...", flush=True)

    # Rank contexts so the model sees strongest signals first
    contexts = sorted(contexts, key=lambda x: x.get("score", 0.0), reverse=True)
    citations_for_ui = build_clean_citations(contexts, limit=UI_CITATION_MAX)

    # Keep only the top ~8 chunks in the prompt to control prompt size
    top_contexts = contexts[:8]
    context_block = "\n\n---\n\n".join(
        [f"Source {i+1} ({c.get('file','?')}):\n{c.get('text','')}" for i, c in enumerate(top_contexts)]
    )

    # Build prompt (no nested triple quotes → avoids unterminated string issues)
    # prompt = (
    #     "You are a precise support assistant. Use only the sources to answer.\n\n"
    #     "Question:\n{q}\n\n"
    #     "Sources:\n{sources}\n\n"
    #     "Return output in EXACTLY two fenced blocks, in this order:\n\n"
    #     "```text\n"
    #     "<concise friendly answer, <=120 words>\n"
    #     "```\n\n"
    #     "```json\n"
    #     "{{\"facts\": [\"...\",\"...\"], \"citations\": [1,2], \"confidence\": 0.0}}\n"
    #     "```\n\n"
    #     "Use 1-based indices for \"citations\" matching the numbered Sources above.\n"
    #     "If information is missing, state that in the text and keep JSON consistent."
    # ).format(q=query, sources=context_block)
    
    prompt = (
        "You are a precise support assistant. Use only the sources to answer.\n\n"
        "Question:\n{q}\n\n"
        "Sources:\n{sources}\n\n"
        "Return output in EXACTLY two fenced blocks, in this order:\n\n"
        "```text\n"
        "<concise friendly answer, <=120 words>\n"
        "Do NOT include citation markers like [1], [2], etc. in this text.\n"
        "Do NOT mention sources or references in the text.\n"
        "```\n\n"
        "```json\n"
        "{{\"facts\": [\"...\",\"...\"], \"citations\": [1,2], \"confidence\": 0.0}}\n"
        "```\n\n"
        "Use 1-based indices for \"citations\" matching the numbered Sources above.\n"
        "All citation numbers MUST appear only in the JSON block, never in the text."
    ).format(q=query, sources=context_block)

    resp = oai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()

    # Parse fenced code blocks
    matches = re.findall(r"```(text|json)\s*([\s\S]*?)```", raw, flags=re.I)
    text_block, json_block = None, None
    for lang, body in matches:
        body = body.strip()
        if lang.lower() == "text":
            text_block = body
        elif lang.lower() == "json":
            json_block = body

    parsed_json: Dict[str, Any] = {}
    if json_block:
        try:
            parsed_json = json.loads(json_block)
        except json.JSONDecodeError:
            parsed_json = {"_parse_error": True, "raw": json_block}

    return {
        "text": (text_block or raw).strip(),
        "json": parsed_json,
        "citations": citations_for_ui,
    }


# ----------------------
# Audio (Whisper) STT
# ----------------------
def transcribe_audio(oai: OpenAI, audio_path: str, whisper_model: str) -> str:
    print(f"[QA] Transcribing audio: {audio_path}", flush=True)
    with open(audio_path, "rb") as f:
        tr = oai.audio.transcriptions.create(model=whisper_model, file=f)
    print("[QA] Transcription complete.", flush=True)
    return tr.text.strip()


# ----------------------
# Public entry point
# ----------------------
def answer_query(query: str = None, audio_path: str = None) -> Dict[str, Any]:
    load_dotenv()

    # OpenAI config
    oai = OpenAI()
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    whisper = os.getenv("WHISPER_MODEL", "whisper-1")

    # Index presence
    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError("Index not found. Run: python ingest.py")

    # Load FAISS + meta
    index, meta = load_index()

    # Query text source
    if audio_path and not query:
        query = transcribe_audio(oai, audio_path, whisper)
    elif not query:
        raise ValueError("Provide either `query` text or `audio_path`.")

    print(f"[QA] Query: {query}", flush=True)

    # Retrieve + expand
    ctx = retrieve(oai, query, index, meta, embed_model)
    ctx = graph_expand(ctx)

    # Synthesize final
    out = synthesize_answer(oai, query, ctx, chat_model)
    print("[QA] Done.", flush=True)

    # Response envelope
    return {
        "query": query,
        "structured": {
            "citations": out["citations"],  # small, de-duplicated list for UI
            "model_json": out["json"],      # parsed {"facts":[], "citations":[], "confidence": float}
        },
        "unstructured": out["text"],        # concise natural-language answer
    }


# ----------------------
# Manual test
# ----------------------
if __name__ == "__main__":
    res = answer_query(query="Do foreign transactions incur a fee, and how much?")
    print(json.dumps(res, indent=2))