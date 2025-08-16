#!/usr/bin/env python3
"""
ingest.py
---------
Ingest .docx policies, chunk, embed (OpenAI), index (FAISS),
and upsert into Neo4j as (:File)-[:HAS_CHUNK]->(:Chunk).

Usage:
    source .venv/bin/activate
    python ingest.py                       # uses ./policies
    python ingest.py --folder ./my_docs    # custom folder
"""

from __future__ import annotations

import os
import glob
import uuid
import pickle
import argparse
from typing import List, Dict, Tuple

import numpy as np
import faiss
from dotenv import load_dotenv
from docx import Document
from openai import OpenAI

from neo4j_utils import Neo4jClient

# --------- Config ---------
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
INDEX_PATH = "vector.index"
META_PATH = "meta.pkl"


# --------- Helpers ---------

def read_docx(path: str) -> str:
    """Read a .docx file and return a single text string."""
    doc = Document(path)
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Tokenize by whitespace and create overlapping chunks."""
    tokens = text.split()
    chunks: List[str] = []
    i = 0
    step = max(1, size - overlap)
    while i < len(tokens):
        chunk = tokens[i:i + size]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step
    return chunks


def build_embeddings(oai: OpenAI, texts: List[str], model: str, batch_size: int = 64) -> np.ndarray:
    """Call OpenAI embeddings in batches; return (N, D) float32 array."""
    print(f"[INGEST] Requesting embeddings for {len(texts)} chunks using model='{model}' ...")
    vecs: List[List[float]] = []
    for s in range(0, len(texts), batch_size):
        batch = texts[s:s + batch_size]
        resp = oai.embeddings.create(model=model, input=batch)
        vecs.extend([d.embedding for d in resp.data])
        print(f"[INGEST] Embedded {min(s + batch_size, len(texts))}/{len(texts)}")
    X = np.asarray(vecs, dtype="float32")
    print(f"[INGEST] Embedding matrix shape: {X.shape}")
    return X


def save_faiss_index(X: np.ndarray, index_path: str) -> faiss.IndexFlatIP:
    """Normalize L2 (cosine with inner product), build and persist FAISS index."""
    print("[INGEST] Building FAISS index (IP on L2-normalized vectors) ...")
    if X.ndim != 2 or X.size == 0:
        raise RuntimeError("[INGEST][ERROR] Empty embedding matrix.")
    dim = X.shape[1]
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, index_path)
    print(f"[INGEST] Saved FAISS index -> {index_path}")
    return index


def scan_docx(policies_dir: str) -> List[str]:
    """Return sorted list of .docx files under a folder."""
    files = sorted(glob.glob(os.path.join(policies_dir, "*.docx")))
    print(f"[INGEST] Scanning '{policies_dir}' -> {len(files)} .docx file(s) found.")
    return files


def ingest_one_file(file_path: str) -> Tuple[List[Dict], List[str]]:
    """Read and chunk a single .docx; return meta list and chunk texts."""
    file_name = os.path.basename(file_path)
    print(f"[INGEST] Reading: {file_name}")
    text = read_docx(file_path)
    if not text.strip():
        print(f"[INGEST][WARN] Empty text after parsing: {file_name} (skipping)")
        return [], []
    chunks = chunk_text(text)
    print(f"[INGEST] → {len(chunks)} chunk(s) from {file_name}")

    metas: List[Dict] = []
    for c in chunks:
        cid = str(uuid.uuid4())
        metas.append({"id": cid, "file": file_name, "text": c})
    return metas, chunks


# --------- Neo4j fallback upsert (if client lacks method) ---------

def _upsert_doc_with_chunks_via_cypher(n4j: Neo4jClient, file_name: str, chunks: List[Dict[str, str]]) -> None:
    """
    Safe upsert that works even if Neo4jClient doesn't have upsert_doc_with_chunks().
    Uses one statement per run (Neo4j Python driver requirement).
    """
    ids = [str(c["id"]) for c in chunks]

    with n4j.driver.session(database=getattr(n4j, "db", "neo4j")) as s:
        # Ensure File
        s.run("MERGE (:File {name: $file})", file=file_name).consume()

        if chunks:
            # Upsert chunks + relationships
            upsert_chunks = """
            UNWIND $rows AS row
            MATCH (f:File {name: $file})
            MERGE (c:Chunk {id: row.id})
              ON CREATE SET c.text = row.text, c.file = $file
              ON MATCH  SET c.text = row.text, c.file = $file
            MERGE (f)-[:HAS_CHUNK]->(c)
            """
            s.run(upsert_chunks, file=file_name, rows=chunks).consume()

            # Drop stale rels from this File
            drop_stale_rels = """
            MATCH (f:File {name: $file})-[r:HAS_CHUNK]->(c:Chunk)
            WHERE NOT c.id IN $ids
            DELETE r
            """
            s.run(drop_stale_rels, file=file_name, ids=ids).consume()

        # Delete orphan chunks (no File pointing to them)
        delete_orphans = """
        MATCH (c:Chunk)
        WHERE NOT EXISTS { MATCH (:File)-[:HAS_CHUNK]->(c) }
        DETACH DELETE c
        """
        s.run(delete_orphans).consume()


# --------- Main pipeline ---------
def main(policies_dir: str = "policies") -> None:
    load_dotenv()

    # OpenAI client
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("[INGEST][ERROR] OPENAI_API_KEY is not set in your environment.")
    print(f"[INGEST] Using OpenAI embeddings model: {embed_model}")
    oai = OpenAI()  # reads OPENAI_API_KEY from env

    # Neo4j client
    print("[INGEST] Connecting to Neo4j ...")
    n4j = Neo4jClient()  # reads NEO4J_URI/USER/PASSWORD/DB from .env
    # If your client creates constraints in __init__, great. If not, we can’t assume here.

    # Scan folder
    files = scan_docx(policies_dir)
    if not files:
        print("[INGEST][DONE] No .docx files to ingest. Add files under './policies' and re-run.")
        n4j.close()
        return

    # Aggregate all
    all_meta: List[Dict] = []
    all_chunks: List[str] = []

    # Ingest file-by-file so graph upserts are visible incrementally
    for fp in files:
        metas, chunks = ingest_one_file(fp)
        if not metas:
            continue

        # Upsert per file to Neo4j (use client method if available, else fallback)
        file_name = metas[0]["file"]
        file_chunks = [{"id": m["id"], "text": m["text"]} for m in metas]

        if hasattr(n4j, "upsert_doc_with_chunks"):
            # Preferred path if your client implements it
            n4j.upsert_doc_with_chunks(file_name, file_chunks)
        else:
            # Fallback that always works
            _upsert_doc_with_chunks_via_cypher(n4j, file_name, file_chunks)

        # Accumulate for vector index
        all_meta.extend(metas)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("[INGEST][WARN] No chunks produced from any files. Nothing to embed/index.")
        n4j.close()
        return

    # Embeddings + FAISS
    X = build_embeddings(oai, all_chunks, model=embed_model)
    _ = save_faiss_index(X, INDEX_PATH)

    # Save metadata
    with open(META_PATH, "wb") as f:
        pickle.dump(all_meta, f)
    print(f"[INGEST] Saved metadata -> {META_PATH}")

    # Wrap up
    n4j.close()
    print("[INGEST] DONE ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest .docx → Neo4j + FAISS")
    parser.add_argument("--folder", "-f", type=str, default="policies", help="Folder with .docx files")
    args = parser.parse_args()

    print(f"[INGEST] Starting ingestion with folder: {args.folder}")
    main(args.folder)