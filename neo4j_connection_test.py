# # neo4j_connection_test.py (Aura-safe)
# import os
# from neo4j import GraphDatabase, basic_auth
# from dotenv import load_dotenv

# # ensure a CA bundle is available
# try:
#     import certifi
#     os.environ["SSL_CERT_FILE"] = certifi.where()
# except Exception:
#     pass

# # purge any proxies that can hijack Bolt/TLS
# for k in ["http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy","no_proxy","NO_PROXY"]:
#     os.environ.pop(k, None)

# load_dotenv()
# URI = os.getenv("NEO4J_URI", "").strip()     # neo4j+s://<DB-ID>.databases.neo4j.io
# USR = os.getenv("NEO4J_USER", "neo4j").strip()
# PWD = os.getenv("NEO4J_PASSWORD", "").strip()
# DB  = os.getenv("NEO4J_DATABASE", "neo4j").strip()

# assert URI.startswith("neo4j+s://"), "Aura requires neo4j+s://"
# print(f"[CHECK] URI={URI} USER={USR} DB={DB}")

# driver = GraphDatabase.driver(
#     URI,
#     auth=basic_auth(USR, PWD),
#     connection_timeout=20,
#     max_transaction_retry_time=20,
#     max_connection_lifetime=300,
# )

# with driver.session(database=DB) as s:
#     print("[CHECK] Running RETURN 1 ...")
#     rec = s.run("RETURN 1 AS ok").single()
#     print("[CHECK] RESULT:", rec["ok"])

# driver.close()
# print("[CHECK] OK ✅")


# local neo4j connection test
# neo4j_connection_test.py (local Neo4j)
import os
from neo4j import GraphDatabase, basic_auth
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

# Local Neo4j connection
URI = os.getenv("NEO4J_URI", "bolt://localhost:7687").strip()   # use 7687 (Bolt), not 7474
USR = os.getenv("NEO4J_USER", "neo4j").strip()
PWD = os.getenv("NEO4J_PASSWORD", "").strip()
DB  = os.getenv("NEO4J_DATABASE", "neo4j").strip()

print(f"[CHECK] URI={URI} USER={USR} DB={DB}")

driver = GraphDatabase.driver(
    URI,
    auth=basic_auth(USR, PWD),
    connection_timeout=20,
    max_transaction_retry_time=20,
    max_connection_lifetime=300,
)

with driver.session(database=DB) as s:
    print("[CHECK] Running RETURN 1 ...")
    rec = s.run("RETURN 1 AS ok").single()
    print("[CHECK] RESULT:", rec["ok"])

driver.close()
print("[CHECK] OK")