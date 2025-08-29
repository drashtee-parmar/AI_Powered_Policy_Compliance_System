import json
import os
import pickle
import re
from typing import Any, Dict, List, Match, Tuple
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from neo4j_utils import Neo4jClient

_MD_SPECIALS = re.compile(r'([\\`*_{}$begin:math:display$$end:math:display$()#+\-!.|>])')  # escape for Markdown

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

def _ip_to_cosine_for_topk(index: faiss.Index, sims_row: np.ndarray, idxs_row: np.ndarray) -> List[float]:
    """
    Convert inner-product scores to cosine similarity for the returned top-K.
    Assumes the query vector is already L2-normalized.
    """
    out: List[float] = []
    for s, i in zip(sims_row, idxs_row):
        try:
            # FAISS reconstruct gives back the stored vector for this id
            vec = index.reconstruct(int(i))   # already a numpy.ndarray
            denom = float(np.linalg.norm(vec)) + 1e-9
            cos = float(s) / denom
            # clamp to [0, 1]
            cos = max(0.0, min(1.0, cos))
            out.append(cos)
        except Exception:
            out.append(float(s))  # fallback
    return out


def retrieve(oai: OpenAI, query: str, index, meta: List[Dict], embed_model: str) -> List[Dict]:
    v = embed_query(oai, query, embed_model)       # v is unit-normalized
    sims, idxs = index.search(v, TOP_K)            # inner-product scores
    sims_row, idxs_row = sims[0], idxs[0]

    # Convert IP -> cosine for the returned top-K if possible
    try:
        sims_row = np.array(_ip_to_cosine_for_topk(index, sims_row, idxs_row), dtype="float32")
    except Exception:
        pass

    sims_list = sims_row.tolist()
    idxs_list = idxs_row.tolist()

    print(f"[QA] Retrieved {TOP_K} vector hits (top score ~ {max(sims_list):.3f}).", flush=True)

    return [
        {
            "score": float(s),
            "file": meta[i]["file"],
            "text": meta[i]["text"],
            "id":   meta[i]["id"],
        }
        for s, i in zip(sims_list, idxs_list)
    ]

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
# At top (optional): control preview length via env, default = 0 (no truncation)
PREVIEW_CHARS = int(os.getenv("CITATION_PREVIEW_CHARS", "0"))

def build_clean_citations(contexts: List[Dict], limit: int = UI_CITATION_MAX) -> List[Dict]:
    best_per_file: Dict[str, Dict] = {}
    for c in contexts:
        f = c.get("file", "?")
        if f not in best_per_file or c.get("score", 0.0) > best_per_file[f].get("score", 0.0):
            best_per_file[f] = c

    best = sorted(best_per_file.values(), key=lambda x: x.get("score", 0.0), reverse=True)[:limit]

    cleaned = []
    for c in best:
        # text = c.get("text", "")
        text = c.get("text", "")
        excerpt = text[:200] + ("..." if len(text) > 200 else "")
        cleaned.append({
            "file": c.get("file", "?"),
            "excerpt": excerpt,
            "score": round(float(c.get("score", 0.0)), 4),
        })
    return cleaned
    #     if PREVIEW_CHARS and PREVIEW_CHARS > 0:
    #         excerpt = text[:PREVIEW_CHARS] + ("..." if len(text) > PREVIEW_CHARS else "")
    #     else:
    #         excerpt = text  # full chunk
    #     cleaned.append({
    #         "file": c.get("file", "?"),
    #         "excerpt": excerpt,
    #         "score": round(float(c.get("score", 0.0)), 4),
    #     })
    # return cleaned

# def build_clean_citations(contexts: List[Dict], limit: int = UI_CITATION_MAX) -> List[Dict]:
#     """
#     Deduplicate by file; keep the best-scoring snippet per file.
#     Return top N by score.
#     """
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
#             "file": c.get("file", "?"),
#             "excerpt": (text[:160] + ("..." if len(text) > 160 else "")),
#             "score": round(float(c.get("score", 0.0)), 4),
#         })
#     return cleaned


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
    
    # prompt = (
    #     "You are a precise support assistant. Use only the sources to answer.\n\n"
    #     "Question:\n{q}\n\n"
    #     "Sources:\n{sources}\n\n"
    #     "Return output in EXACTLY two fenced blocks, in this order:\n\n"
    #     "```text\n"
    #     "<concise friendly answer, <=120 words>\n"
    #     "Do NOT include citation markers like [1], [2], etc. in this text.\n"
    #     "Do NOT mention sources or references in the text.\n"
    #     "```\n\n"
    #     "```json\n"
    #     "{{\"facts\": [\"...\",\"...\"], \"citations\": [1,2], \"confidence\": 0.0}}\n"
    #     "```\n\n"
    #     "Use 1-based indices for \"citations\" matching the numbered Sources above.\n"
    #     "All citation numbers MUST appear only in the JSON block, never in the text."
    # ).format(q=query, sources=context_block)
    prompt = (
        "You are a precise support assistant. Use only the sources to answer.\n\n"
        "Question:\n{q}\n\n"
        "Sources:\n{sources}\n\n"
        "Return output in EXACTLY two fenced blocks, in this order:\n\n"
        "```text\n"
        "<concise friendly answer, <=120 words>\n"
        "Make sure all numbers include proper formatting (e.g., '1,000' not '1000') "
        "and add spaces around numbers when followed by words (e.g., '1,000 purchase', not '1,000purchase').\n"
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
        "text": clean_text_output((text_block or raw).strip()),
        "json": parsed_json,
        "citations": citations_for_ui,
    }
    
    # return {
    #     "text": (text_block or raw).strip(),
    #     "json": parsed_json,
    #     "citations": citations_for_ui,
    # }

# def clean_text_output(text: str) -> str:
#     # Ensure a space between numbers and words
#     text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
#     text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)
#     return text


# def clean_text_output(text: str) -> str:
#     """
#     Ensure numbers are formatted cleanly and separated from words.
#     - Adds commas for thousands
#     - Ensures space between numbers and letters
#     """
#     # Add space between numbers and letters stuck together
#     text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
#     text = re.sub(r"([A-Za-z])(\d)", r"\1 \2", text)

#     # Normalize big numbers like 1000 -> 1,000 (only when standalone or followed by space/word)
#     def add_commas(match):
#         return "{:,}".format(int(match.group(0)))

#     text = re.sub(r"\b\d{4,}\b", add_commas, text)

#     # Fix cases like "30 fee" -> "30 fee" (ensures a space stays)
#     text = re.sub(r"(\d+)\s*([a-zA-Z])", r"\1 \2", text)

#     return text.strip()


# def clean_text_output(txt: str) -> str:
#     # Ensure commas in numbers are preserved properly
#     txt = re.sub(r"(\d),(\d{3})", r"\1,\2", txt)
    
#     # Add a space if a number is directly followed by a word
#     txt = re.sub(r"(\d)([A-Za-z])", r"\1 \2", txt)
    
#     # Add a space if a word is jammed before a number
#     txt = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", txt)
    
#     # Collapse multiple spaces
#     txt = re.sub(r"\s+", " ", txt)

#     return txt.strip()



def _fmt_thousands(m: Match[str]) -> str:
    """Format a long digit run with commas: 1000 -> 1,000; 1234567 -> 1,234,567."""
    s = m.group(0)
    return f"{int(s):,}"

def clean_text_output(txt: str) -> str:
    """Remove JSON-style escapes and fix spacing issues."""
    # Remove backslashes before letters/numbers
    txt = re.sub(r'\\([a-zA-Z0-9])', r'\1', txt)
    # Collapse multiple spaces
    txt = re.sub(r'\s+', ' ', txt)
    # Normalize commas/numbers formatting
    txt = txt.replace(" ,", ",").replace(" .", ".")
    return txt.strip()


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
    print(json.dumps(res, indent=2, ensure_ascii=False))