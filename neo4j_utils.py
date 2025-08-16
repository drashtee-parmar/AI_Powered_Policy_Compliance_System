# neo4j_utils.py
from __future__ import annotations

import os
from typing import Iterable, Dict, List

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ConfigurationError
from neo4j import GraphDatabase

# Removed erroneous driver initialization outside of class context.

class Neo4jClient:
    """
    Minimal helper for this project.
    - Connects using env vars
    - Ensures unique constraints
    - Upserts a Doc node and its Chunk children
    """

    def __init__(self, uri: str | None = None, user: str | None = None,
                 password: str | None = None, database: str | None = None,
                 verbose: bool = True) -> None:
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        # Neo4j Aura uses 'neo4j' by default; we also respect NEO4J_DATABASE
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.verbose = verbose

        if not (self.uri and self.user and self.password):
            raise RuntimeError(
                "[NEO4J] Missing NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in environment."
            )

        if self.verbose:
            print(f"[NEO4J] Connecting to {self.uri} as {self.user} (db={self.database}) ...")

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # sanity check
            with self.driver.session(database=self.database) as s:
                s.run("RETURN 1")
            if self.verbose:
                print("[NEO4J] Connected ✅")
        except (ServiceUnavailable, ConfigurationError) as e:
            raise RuntimeError(f"[NEO4J] Connection failed: {e}") from e

        # Create constraints on first use
        self.ensure_constraints()

    # ---------------------------------------------------------------------

    def close(self) -> None:
        if hasattr(self, "driver") and self.driver:
            if self.verbose:
                print("[NEO4J] Closing driver.")
            self.driver.close()

    # ---------------------------------------------------------------------

    def ensure_constraints(self) -> None:
        """Create id/file uniqueness once (no-op if they already exist)."""
        if self.verbose:
            print("[NEO4J] Ensuring constraints ...")

        stmts = [
            # Unique doc by file
            "CREATE CONSTRAINT doc_file IF NOT EXISTS FOR (d:Doc) REQUIRE d.file IS UNIQUE",
            # Unique chunk by id
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        ]
        with self.driver.session(database=self.database) as s:
            for cypher in stmts:
                s.run(cypher).consume()

        if self.verbose:
            print("[NEO4J] Constraints ready.")

    # ---------------------------------------------------------------------

    def upsert_doc_with_chunks(self, file_name: str, chunks: Iterable[Dict[str, str]]) -> None:
        """
        Ensure a (:Doc {file}) exists; upsert (:Chunk {id,text}) and (:Doc)-[:HAS_CHUNK]->(:Chunk).
        Remove stale relationships to chunks no longer present for that file.
        `chunks` must be an iterable of dicts with keys {'id','text'}.
        """
        # normalize to list
        chunk_list: List[Dict[str, str]] = list(chunks)
        ids = [c["id"] for c in chunk_list]

        with self.driver.session(database=self.database) as s:
            # 1) Ensure the Doc node exists
            s.run(
                """
                MERGE (d:Doc {file: $file})
                ON CREATE SET d.createdAt = timestamp()
                """,
                file=file_name,
            ).consume()

            # 2) Upsert each chunk node + relationship
            s.run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {id: row.id})
                ON CREATE SET c.text = row.text, c.createdAt = timestamp()
                ON MATCH  SET c.text = row.text
                WITH c, row
                MATCH (d:Doc {file: $file})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                file=file_name,
                rows=chunk_list,
            ).consume()

            # 3) Remove relationships to chunks that are no longer present
            # (do NOT delete the chunks themselves, just detach from this Doc)
            s.run(
                """
                MATCH (d:Doc {file: $file})-[r:HAS_CHUNK]->(c:Chunk)
                WHERE NOT c.id IN $ids
                DELETE r
                """,
                file=file_name,
                ids=ids,
            ).consume()