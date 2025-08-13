# server.py
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from qa import answer_query

app = FastAPI(title="Voice Q&A")

# Allow local-network access from phones/laptops on your LAN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your LAN/corp domain if needed
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static mic UI from ./static
if not os.path.isdir("static"):
    os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    # simple redirect to our UI
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/qa_text")
def qa_text(question: str = Form(...)):
    """
    Text → RAG → Answer (useful for quick checks from the UI)
    """
    resp = answer_query(query=question)
    return JSONResponse(resp)


@app.post("/qa")
async def qa_audio(audio: UploadFile = File(...)):
    """
    Audio → Whisper → RAG → Answer
    Accepts: audio/webm, audio/m4a, audio/mp3, audio/wav, etc.
    """
    # Write upload to a temp file on disk so qa.py can read it
    suffix = os.path.splitext(audio.filename or "upload")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        resp = answer_query(audio_path=tmp_path)
        return JSONResponse(resp)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 so other devices on your LAN can access; change port if needed
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)