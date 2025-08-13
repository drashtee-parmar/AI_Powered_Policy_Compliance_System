# save as check_neo4j.py and run: python check_neo4j.py
import os
from neo4j import GraphDatabase
uri=os.getenv("NEO4J_URI"); user=os.getenv("NEO4J_USERNAME"); pwd=os.getenv("NEO4J_PASSWORD")
drv=GraphDatabase.driver(uri, auth=(user,pwd))
with drv.session() as s:
    print(s.run("RETURN 1 AS ok").single()["ok"])