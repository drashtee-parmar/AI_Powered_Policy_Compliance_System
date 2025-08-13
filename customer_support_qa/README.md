# Neo4j connectior
https://console-preview.neo4j.io/projects/64be119e-9a78-48b9-9dbd-50f5ba6cba11/instances

https://browser.graphapp.io/

# remove it after its tested on bash and checking neo4j connection

cypher-shell -a "neo4j+s://37b30e3e.databases.neo4j.io" -u neo4j -p 'j992HzS8Mz5ka-AnFP6y3_BTSRr_cXjq7DogC0p3qmM' "RETURN 1;"

# Neo4j Browser
If you’re using Neo4j Aura, open the Neo4j browser:
	•	Go to https://console.neo4j.io

	•	Open the Browser for your database.

	•	Run this Cypher:
    
```

// List all documents
MATCH (d:Doc) RETURN d LIMIT 10;

// Count all documents
MATCH (d:Doc) RETURN count(d) AS docs;

// Count all chunks
MATCH (c:Chunk) RETURN count(c) AS chunks;

// See one document with chunks
MATCH (d:Doc)-[:HAS_CHUNK]->(c:Chunk)
RETURN d.file AS document, collect(c.text)[0..3] AS sample_chunks
LIMIT 5;
```

# installation 
pip install sounddevice soundfile requests
pip install faiss-cpu
python -c "import faiss; print(faiss.__version__)"
pip install streamlit streamlit-mic-recorder
