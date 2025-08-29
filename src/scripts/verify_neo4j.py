from neo4j_utils import Neo4jClient

n4j = Neo4jClient()

# Check documents
docs = n4j.query("MATCH (d:Doc) RETURN d.file AS file LIMIT 10")
print("[VERIFY] Documents:", [row["file"] for row in docs])

# Check chunk count
count_chunks = n4j.query("MATCH (c:Chunk) RETURN count(c) AS chunks")
print("[VERIFY] Chunk count:", count_chunks[0]["chunks"])

n4j.close()