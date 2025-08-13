# neo4j_utils.py
import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, basic_auth

class Neo4jClient:
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        verbose: bool = True,
    ):
        self.uri = uri or os.getenv("NEO4J_URI")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self.verbose = verbose

        if self.verbose:
            print(f"[NEO4J] Connecting to {self.uri} as {self.user} ...", flush=True)

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=basic_auth(self.user, self.password),
            max_connection_lifetime=300,
        )
        # Sanity ping
        with self.driver.session(database=self.database) as s:
            s.run("RETURN 1")
        if self.verbose:
            print("[NEO4J] Connected ✅", flush=True)

    def close(self) -> None:
        if self.driver:
            if self.verbose:
                print("[NEO4J] Closing driver.")
            self.driver.close()

    # Used by the pipeline for a light graph expansion
    def expand_related_chunks(self, file_name: str, limit: int = 1) -> List[Dict[str, Any]]:
        q = """
        MATCH (c:Chunk)
        WHERE toLower(trim(coalesce(c.file, ''))) = toLower(trim($file))
        WITH c
        ORDER BY c.id ASC
        RETURN c.id AS id, c.text AS text, c.file AS file
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as s:
            rows = s.run(q, file=file_name, limit=limit).data()
        return [{"id": r["id"], "text": r["text"], "file": r["file"]} for r in rows]

    # Kept for completeness (not used in the current UI)
    def get_all_chunks_for_file(self, file_name: str) -> List[Dict[str, Any]]:
        q = """
        MATCH (c:Chunk)
        WHERE toLower(trim(coalesce(c.file, ''))) = toLower(trim($file))
        WITH c
        ORDER BY c.id ASC
        RETURN c.id AS id, c.text AS text, c.file AS file
        """
        with self.driver.session(database=self.database) as s:
            rows = s.run(q, file=file_name).data()
        return [{"id": r["id"], "text": r["text"], "file": r["file"]} for r in rows]