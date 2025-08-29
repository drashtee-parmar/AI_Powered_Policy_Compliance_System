# Neo4j connectior
<!-- https://console-preview.neo4j.io/projects/64be119e-9a78-48b9-9dbd-50f5ba6cba11/instances -->

https://browser.graphapp.io/
 - login using
 NEO4J_URI=neo4j+s://{instance}.databases.neo4j.io
NEO4J_USER=neo4j
NEO4J_PASSWORD=enter your password

nslookup 37b30e3e.databases.neo4j.io

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

<!-- running the applicaiton -->
source .venv/bin/activate
streamlit run streamlit_chat.py


# SSH into the instance
<!-- chmod 400 ~/.ssh/your-key.pem
ssh -i ~/.ssh/Policy-Compliance-Key.pem ubuntu@YOUR_EC2_IP 
-->
```
ssh -i ~/.ssh/Policy-Compliance-Key.pem ubuntu@52.205.26.10

git clone https://github.com/drashtee-parmar/AI_Powered_Policy_Compliance_System


```



```
ls -l ~/.ssh/Policy-Compliance-Key.pem
chmod 400 ~/.ssh/Policy-Compliance-Key.pem
ssh -i ~/.ssh/Policy-Compliance-Key.pem -N -L 8501:localhost:8501 ubuntu@52.205.26.10

52.205.26.10

ssh -i path/to/key.pem ec2-user@<EC2_PUBLIC_IP>


chmod 400~/.ssh/Users/drashteeparmar/Drashtee/Drashtee Projects/AI_Powered_Policy_Compliance_System
chmod 400 ~/.ssh/Policy-Compliance-Key.pem

ssh -i ~/.ssh/Policy-Compliance-Key.pem ubuntu@52.205.26.10

scp -i ~/.ssh/Policy-Compliance-Key.pem vector.index meta.pkl ubuntu@52.205.26.10:/home/ubuntu/policy-qa/

ssh -i ~/.ssh/Policy-Compliance-Key.pem -L 8501:localhost:8501 ubuntu@52.205.26.10


<!-- If your policies are on your laptop but not on EC2 yet, first upload them to the server -->

scp -i ~/.ssh/Policy-Compliance-Key.pem -r ./policies \
  ubuntu@52.205.26.10:/home/ubuntu/policy-qa/policies
  
  #===================
cd ~/policy-qa
ls -la
# See the fully-resolved compose config (this will fail if it's missing image/build)
docker compose config


ssh -i your-key.pem ec2-user@<your-instance-public-ip>
ssh -i Policy-Compliance-Key.pem ec2-user@52.205.26.10
ssh -i ~/.ssh/Policy-Compliance-Key.pem ec2-user@52.205.26.10

ssh -i Policy-Compliance-Key.pem ubuntu@52.205.26.10

yes
# See it running
docker ps

# Tail logs (Ctrl+C to stop)
docker logs -f policy-qa-app-1
curl -I http://localhost:8501 #If curl returns 200 OK, the app is healthy locally.
http://<EC2_PUBLIC_IP>:8501
 ```


osascript -e 'quit app "Docker"' 2>/dev/null || true
open -a "Docker"



 52.205.26.10


chmod 400 ~/.ssh/Users/drashteeparmar/Drashtee/Drashtee Projects/AI_Powered_Policy_Compliance_System/Policy-Compliance-Key.pem



scp -i ~/.ssh/Policy-Compliance-Key.pem \
    vector.index meta.pkl \
    ubuntu@3.81.105.216:/home/ubuntu/policy-qa/
# 3) Install Docker, Compose plugin, AWS CLI

/Users/drashteeparmar/Drashtee/Drashtee Projects/AI_Powered_Policy_Compliance_System/Policy-Compliance-Key.pem