# neo4j_utils.py
from __future__ import annotations

import os
from typing import List, Dict

from dotenv import load_dotenv
from neo4j import GraphDatabase


class Neo4jClient:
    """
    Thin wrapper around neo4j.Driver used by ingest.py and qa.py.
    Expects the following env vars (a .env is fine):
      - NEO4J_URI         e.g., neo4j+s://<id>.databases.neo4j.io
      - NEO4J_USER        e.g., neo4j
      - NEO4J_PASSWORD    e.g., ********
      - NEO4J_DATABASE or NEO4J_DB (optional, defaults to 'neo4j')
    """

    def __init__(self, verbose: bool = True) -> None:
        load_dotenv()  # allow .env usage locally

        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        pwd = os.getenv("NEO4J_PASSWORD")
        db = (
            os.getenv("NEO4J_DATABASE")
            or os.getenv("NEO4J_DB")
            or "neo4j"
        )

        if not uri or not user or not pwd:
            raise RuntimeError(
                "[NEO4J] Missing NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in environment."
            )

        self.uri = uri
        self.user = user
        self.database = db
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, pwd))

        # smoke-test the connection
        with self.driver.session(database=self.database) as s:
            s.run("RETURN 1").consume()

        if verbose:
            print(f"[NEO4J] Connecting to {self.uri} as {self.user} (db={self.database}) ...")
            print("[NEO4J] Connected ✅")

    # ---------- convenience ----------
    def session(self):
        """Compatibility helper so callers can do n4j.session()."""
        return self.driver.session(database=self.database)
    
    # ---------- lifecycle ----------
    def close(self) -> None:
        try:
            self.driver.close()
        except Exception:
            pass
        print("[NEO4J] Closing driver.")

    # ---------- schema ----------
    def ensure_constraints(self) -> None:
        """
        Create id/uniqueness constraints if they don't exist.
        Aura supports IF NOT EXISTS; each executed as a separate query.
        """
        stmts = [
            "CREATE CONSTRAINT file_name_unique IF NOT EXISTS "
            "FOR (f:File) REQUIRE f.name IS UNIQUE",

            "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS "
            "FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as s:
            for cypher in stmts:
                s.run(cypher).consume()
        print("[NEO4J] Ensuring constraints ...")
        print("[NEO4J] Constraints ready.")

    # ---------- write: upsert a document and its chunks ----------
    def upsert_doc_with_chunks(self, file_name: str, chunks: List[Dict[str, str]]) -> None:
        """
        Upsert a (:File{name}) and its (:Chunk{id,text}) with [:HAS_CHUNK] edges.
        Also removes edges to chunks that are no longer in 'chunks'.
        `chunks` is a list of dicts: [{"id": "...","text": "..."}, ...]
        """
        ids = [c["id"] for c in chunks]

        with self.driver.session(database=self.database) as s:
            # Merge file node
            s.run(
                "MERGE (f:File {name: $file})",
                file=file_name,
            ).consume()

            # Upsert chunks and relationships
            s.run(
                """
                MATCH (f:File {name: $file})
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                SET c.text = row.text
                MERGE (f)-[:HAS_CHUNK]->(c)
                """,
                file=file_name,
                rows=chunks,
            ).consume()

            # Remove stale HAS_CHUNK edges for chunks not in 'ids'
            s.run(
                """
                MATCH (f:File {name: $file})-[r:HAS_CHUNK]->(c:Chunk)
                WHERE NOT c.id IN $ids
                DELETE r
                """,
                file=file_name,
                ids=ids,
            ).consume()

    # ---------- read: related chunks for a given file ----------
    def expand_related_chunks(self, file_name: str, limit: int = 5) -> List[Dict[str, str]]:
        """
        Return up to `limit` chunks (id, text) related to the given file via [:HAS_CHUNK].
        """
        with self.driver.session(database=self.database) as s:
            res = s.run(
                """
                MATCH (f:File {name: $file})-[:HAS_CHUNK]->(c:Chunk)
                RETURN c.id AS id, c.text AS text
                LIMIT $limit
                """,
                file=file_name,
                limit=limit,
            )
            return [{"id": r["id"], "text": r["text"]} for r in res]