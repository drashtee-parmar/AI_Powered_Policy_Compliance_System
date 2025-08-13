# app.py
import argparse
import json
import os
import sys

from qa import answer_query

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--query", type=str, help="Text question")
    p.add_argument("--audio", type=str, help="Path to audio file")
    args = p.parse_args()

    print(f"[APP] Python: {sys.executable}")
    print(f"[APP] CWD: {os.getcwd()}")
    print("[APP] Starting Q&A ...", flush=True)

    resp = answer_query(query=args.query, audio_path=args.audio)

    print("[APP] Response:")
    # ASCII-only by construction; ASCII escaping is fine
    print(json.dumps(resp, indent=2, ensure_ascii=True))