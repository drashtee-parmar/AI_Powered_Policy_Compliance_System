# qa.py
# Core Q&A: FAISS retrieval + light Neo4j expansion + LLM synthesis
# Includes robust text normalization so examples like "1, 000purchase..."
# become "1,000 purchase", and common glued phrases are repaired.

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


# =========================
# Config
# =========================
INDEX_PATH = "vector.index"
META_PATH  = "meta.pkl"
TOP_K = 5
GRAPH_EXPAND_PER_DOC = 1
UI_CITATION_MAX = 5


# =========================
# Text normalization
# =========================

# Unicode and punctuation fixes
_ASCII_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201A": "'", "\u201B": "'",
    "\u201C": '"', "\u201D": '"', "\u201E": '"',
    "\u2032": "'", "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00A0": " ",   # non-breaking space
    "\u202F": " ",   # narrow no-break space
    "\u2009": " ",   # thin space
    "\u2026": "...",
    "\u00AD": "",    # soft hyphen
}
_CTRL = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F]")

# Number/currency spacing
_NUM_COMMA         = re.compile(r"(?<=\d)\s*,\s*(?=\d)")    # 1 , 000 -> 1,000
_NUM_LETTER_JOIN   = re.compile(r"(?<=\d)(?=[A-Za-z])")     # 1000purchase -> 1000 purchase
_CURRENCY_TIGHT    = re.compile(r"([$\u00A3\u20AC])\s+(\d)")# $ 30 -> $30; € 1,000 -> €1,000
_SPACE_BEFORE_PCT  = re.compile(r"\s+%")                    # 3 % -> 3%
_SPACE_AROUND_PUNC = re.compile(r"\s+([,.;:!?])")           # " ," -> ","
_MULTI_SPACE       = re.compile(r"[ \t]+")
_MULTI_NL          = re.compile(r"\n{3,}")

# Optional: 4+ digit numbers without commas -> add US grouping (e.g., 10000 -> 10,000)
# Comment out if you don't want this behavior.
_GROUP_4PLUS       = re.compile(r"\b(\d{1,3})(\d{3})(\d{3,})?\b")

# Targeted English "glued phrase" repairs frequently seen in LLM outputs
# Keep minimal and safe—domain-neutral and non-destructive.
PHRASE_FIXES: List[tuple[str, str]] = [
    ("purchaseina", "purchase in a "),
    ("purchasein", "purchase in "),
    ("foreigncountry", "foreign country"),
    ("wouldincur", "would incur"),
    ("transactionfeeof", "transaction fee of "),
    ("feeof", "fee of "),
    ("inac", "in a c"),  # prevents "inac" -> "in a c..." edge merge
    ("homecountry", "home country"),
]


def _to_ascii(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.translate(str.maketrans(_ASCII_MAP))
    s = _CTRL.sub("", s)
    return s


def _reflow_lines(s: str) -> str:
    # Merge single newlines inside paragraphs; keep blank lines as paragraph breaks.
    s = s.replace("\r", "")
    s = re.sub(r"\n[ \t]*\n", "<<<PARA>>>", s)
    s = re.sub(r"\n+", " ", s)
    s = s.replace("<<<PARA>>>", "\n\n")
    return s


def _format_group_4plus(m: re.Match) -> str:
    # 1234 -> 1,234 ; 1234567 -> 1,234,567
    g1, g2, g3 = m.group(1), m.group(2), m.group(3)
    if g3:
        return f"{g1},{g2},{g3}"
    return f"{g1},{g2}"


def _normalize_spaces(s: str) -> str:
    # Normalize number/currency
    s = _NUM_COMMA.sub(",", s)
    s = _CURRENCY_TIGHT.sub(r"\1\2", s)
    s = _SPACE_BEFORE_PCT.sub("%", s)
    s = _SPACE_AROUND_PUNC.sub(r"\1", s)
    s = _NUM_LETTER_JOIN.sub(" ", s)

    # Optional: add commas for large integers without grouping (safe for common outputs)
    s = _GROUP_4PLUS.sub(_format_group_4plus, s)

    # Targeted glued-phrase repairs
    low = s.lower()
    for bad, good in PHRASE_FIXES:
        if bad in low:
            # replace case-insensitively by scanning lower-cased version
            rebuilt = []
            i = 0
            while i < len(s):
                if low.startswith(bad, i):
                    rebuilt.append(good)
                    i += len(bad)
                else:
                    rebuilt.append(s[i])
                    i += 1
            s = "".join(rebuilt)
            low = s.lower()

    # Collapse extra spaces/newlines
    s = _MULTI_SPACE.sub(" ", s)
    s = _MULTI_NL.sub("\n\n", s)
    return s.strip()


def postprocess_text(s: str) -> str:
    if not s:
        return s
    s = _to_ascii(s)
    s = _reflow_lines(s)
    s = _normalize_spaces(s)
    return s


# =========================
# FAISS index / embeddings
# =========================
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


# =========================
# Graph expansion (Neo4j)
# =========================
def graph_expand(contexts: List[Dict]) -> List[Dict]:
    if not contexts:
        return contexts
    n4j = Neo4jClient(verbose=False)
    files = list({c["file"] for c in contexts})
    extra: List[Dict] = []
    for f in files:
        extra.extend(n4j.expand_related_chunks(f, limit=GRAPH_EXPAND_PER_DOC))
    n4j.close()
    have = {c.get("id") for c in contexts}
    added = 0
    for e in extra:
        if e["id"] not in have:
            e["score"] = 0.05
            contexts.append(e)
            added += 1
    if added:
        print(f"[QA] Graph expansion added {added} chunk(s).", flush=True)
    return contexts


# =========================
# Citations for UI
# =========================
def build_clean_citations(contexts: List[Dict], limit: int = UI_CITATION_MAX) -> List[Dict]:
    best_per_file: Dict[str, Dict] = {}
    for c in contexts:
        f = c.get("file", "?")
        if f not in best_per_file or c.get("score", 0.0) > best_per_file[f].get("score", 0.0):
            best_per_file[f] = c
    top = sorted(best_per_file.values(), key=lambda x: x.get("score", 0.0), reverse=True)[:limit]
    out: List[Dict[str, Any]] = []
    for c in top:
        txt = postprocess_text(c.get("text", ""))
        out.append({
            "file": c.get("file", "?"),
            "chunk_id": c.get("id"),
            "excerpt": (txt[:220] + ("..." if len(txt) > 220 else "")),
            "score": round(float(c.get("score", 0.0)), 4),
        })
    return out


# =========================
# LLM synthesis (robust JSON)
# =========================
def synthesize_answer(oai: OpenAI, query: str, contexts: List[Dict], model: str) -> Dict[str, Any]:
    print("[QA] Synthesizing final answer ...", flush=True)
    contexts = sorted(contexts, key=lambda x: x.get("score", 0.0), reverse=True)
    citations_for_ui = build_clean_citations(contexts, limit=UI_CITATION_MAX)

    top_contexts = contexts[:8]
    context_block = "\n\n---\n\n".join(
        [f"Source {i+1} ({c.get('file','?')}):\n{c.get('text','')}" for i, c in enumerate(top_contexts)]
    )

    prompt = (
        "You are a precise support assistant. Use only the sources to answer.\n\n"
        f"Question:\n{query}\n\n"
        f"Sources:\n{context_block}\n\n"
        "Return output in EXACTLY two fenced blocks, in this order:\n\n"
        "```text\n"
        "<plain ASCII answer; no markdown emphasis; use normal sentences and '-' bullets>\n"
        "Include a concrete example like: 'For example, a $1,000 purchase in a foreign country would incur a foreign transaction fee of $30.'\n"
        "```\n\n"
        "```json\n"
        "{\"facts\": [\"...\",\"...\"], \"citations\": [1,2], \"confidence\": 0.0}\n"
        "```\n\n"
        "The JSON block MUST include keys: facts (array), citations (array), confidence (number).\n"
        "If a key is unknown, still include it with an empty array or null."
    )

    resp = oai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content.strip()

    # Parse fenced blocks
    m = re.findall(r"```(text|json)\s*([\s\S]*?)```", raw, flags=re.I)
    text_block, json_block = None, None
    for lang, body in m:
        body = body.strip()
        if lang.lower() == "text":
            text_block = body
        elif lang.lower() == "json":
            json_block = body

    # Robust JSON normalization
    parsed_json: Any = {}
    if json_block:
        try:
            parsed_json = json.loads(json_block)
        except Exception:
            parsed_json = {}
    if not isinstance(parsed_json, dict):
        parsed_json = {}
    parsed_json.setdefault("facts", [])
    parsed_json.setdefault("citations", [])
    parsed_json.setdefault("confidence", None)

    # Normalize text & facts
    text_block = postprocess_text(text_block or raw)
    if isinstance(parsed_json["facts"], list):
        parsed_json["facts"] = [postprocess_text(str(f)) for f in parsed_json["facts"]]
    else:
        parsed_json["facts"] = []

    return {"text": text_block, "json": parsed_json, "citations": citations_for_ui}


# =========================
# Speech‑to‑text
# =========================
def transcribe_audio(oai: OpenAI, audio_path: str, whisper_model: str) -> str:
    with open(audio_path, "rb") as f:
        tr = oai.audio.transcriptions.create(model=whisper_model, file=f)
    return postprocess_text(tr.text.strip())


# =========================
# Public entry point
# =========================
def answer_query(query: str = None, audio_path: str = None) -> Dict[str, Any]:
    load_dotenv()
    oai = OpenAI()
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    chat_model  = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    whisper     = os.getenv("WHISPER_MODEL", "whisper-1")

    if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
        raise RuntimeError("Index not found. Run: python ingest.py")

    index, meta = load_index()

    if audio_path and not query:
        query = transcribe_audio(oai, audio_path, whisper)
    elif not query:
        raise ValueError("Provide either `query` text or `audio_path`.")
    query = postprocess_text(query)

    print(f"[QA] Query: {query}", flush=True)

    ctx = retrieve(oai, query, index, meta, embed_model)
    ctx = graph_expand(ctx)

    out = synthesize_answer(oai, query, ctx, chat_model)
    print("[QA] Done.", flush=True)

    return {
        "query": query,
        "structured": {
            "citations": out["citations"],
            "model_json": out["json"],
        },
        "unstructured": out["text"],
    }