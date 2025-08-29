# import os
# import tempfile
# import json
# import streamlit as st

# # Your pipeline entrypoint
# from qa import answer_query

# st.set_page_config(page_title="Voice Q&A (Local)", page_icon="🎙️", layout="centered")

# st.title("Voice Q&A (Local)")
# st.caption("Speak or upload audio → Whisper → RAG → Answer")

# # --- Sidebar ---
# with st.sidebar:
#     st.header("Input")
#     use_mic = st.toggle("Use Microphone", value=True, help="Disable to use file upload only")
#     st.divider()
#     st.header("Options")
#     show_raw = st.checkbox("Show raw JSON", value=True)
#     st.caption("All processing runs locally; your APIs (OpenAI, Neo4j Aura) are used as configured in .env.")

# # --- Main UI ---
# query_text = st.text_input("Or type a question (bypasses mic):", value="")

# audio_bytes = None
# temp_audio_path = None

# if use_mic:
#     # Mic component: returns a button that records and returns bytes
#     try:
#         from streamlit_mic_recorder import mic_recorder

#         st.write("### Microphone")
#         audio_dict = mic_recorder(
#             start_prompt="Start recording",
#             stop_prompt="Stop recording",
#             just_once=False,
#             use_container_width=True,
#             format="webm",  # 'wav' also supported; webm is typically smaller
#         )
#         if audio_dict and audio_dict.get("bytes"):
#             audio_bytes = audio_dict["bytes"]
#             st.success("Captured audio from microphone.")
#             st.audio(audio_bytes, format="audio/webm")

#     except Exception as e:
#         st.warning(f"Microphone component unavailable ({e}). Use file upload below.")
#         use_mic = False

# # Fallback / alternative: file upload
# # st.write("### Upload audio file (MP3/WAV/M4A/WEBM)")
# # up = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "webm"], accept_multiple_files=False)
# # if up is not None:
# #     audio_bytes = up.read()
# #     st.audio(audio_bytes, format=f"audio/{(up.type or 'audio')}")
# #     st.info(f"Loaded file: {up.name}")

# run_btn = st.button("Run Q&A", type="primary", use_container_width=True)

# # --- Execute pipeline ---
# if run_btn:
#     if not query_text and not audio_bytes:
#         st.error("Provide either a typed question OR an audio recording/file.")
#         st.stop()

#     try:
#         # If we have audio bytes, persist to a temp file and call qa.answer_query(audio_path=...)
#         if audio_bytes and not query_text:
#             suffix = ".webm"
#             if up is not None:
#                 # preserve file extension when uploaded
#                 _, ext = os.path.splitext(up.name)
#                 if ext:
#                     suffix = ext
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 tmp.write(audio_bytes)
#                 temp_audio_path = tmp.name

#             with st.spinner("Transcribing + retrieving + synthesizing..."):
#                 resp = answer_query(audio_path=temp_audio_path)

#         else:
#             # Text-only path (no audio)
#             with st.spinner("Retrieving + synthesizing..."):
#                 resp = answer_query(query=query_text)

#         # Display result
#         st.subheader("Answer")
#         st.write(resp.get("unstructured", ""))

#         st.subheader("Facts / Citations / Confidence")
#         model_json = resp.get("structured", {}).get("model_json", {})
#         st.json(model_json)

#         st.subheader("Top Citations (clean)")
#         st.json(resp.get("structured", {}).get("citations", []))

#         if show_raw:
#             st.subheader("Raw Response")
#             st.code(json.dumps(resp, indent=2, ensure_ascii=True), language="json")

#     except Exception as e:
#         st.error(f"Pipeline error: {e}")

#     finally:
#         if temp_audio_path and os.path.exists(temp_audio_path):
#             try:
#                 os.remove(temp_audio_path)
#             except Exception:
#                 pass

# st.divider()
# st.caption("Tip: if the mic doesn’t appear, ensure browser permission is granted and try another browser.")

# ===================================
# Working code 2
# ====================================

# import os
# import json
# import tempfile
# import streamlit as st

# from qa import answer_query  # uses your existing pipeline

# st.set_page_config(page_title="Voice Q&A (Local)", page_icon="🎙️", layout="centered")

# st.title("Voice Q&A (Local)")
# st.caption("Speak, upload audio, or type → Whisper → RAG → Answer")

# with st.sidebar:
#     st.header("Input")
#     use_mic = st.toggle("Use Microphone", value=True)
#     st.divider()
#     st.header("Display")
#     show_raw = st.checkbox("Show raw JSON", value=True)
#     st.caption("All processing runs locally; external APIs (OpenAI, Neo4j Aura) use your .env.")

# query_text = st.text_input("Or type a question:", value="")
# audio_bytes = None
# temp_audio_path = None

# # --- Microphone (streamlit-mic-recorder) ---
# if use_mic:
#     try:
#         from streamlit_mic_recorder import mic_recorder
#         st.write("### Microphone")
#         audio_dict = mic_recorder(
#             start_prompt="Start recording",
#             stop_prompt="Stop recording",
#             just_once=False,
#             use_container_width=True,
#             format="webm",  # wav also works; webm is smaller
#         )
#         if audio_dict and audio_dict.get("bytes"):
#             audio_bytes = audio_dict["bytes"]
#             st.success("Captured audio from microphone.")
#             st.audio(audio_bytes, format="audio/webm")
#     except Exception as e:
#         st.warning(f"Microphone component unavailable ({e}). Use file upload or text.")
#         use_mic = False

# # --- File upload (fallback/alternative) ---
# st.write("### Upload audio file (MP3/WAV/M4A/WEBM)")
# up = st.file_uploader(
#     "Choose an audio file",
#     type=["mp3", "wav", "m4a", "webm"],
#     accept_multiple_files=False
# )
# if up is not None:
#     audio_bytes = up.read()
#     st.audio(audio_bytes, format=up.type or "audio")

# run_btn = st.button("Run Q&A", type="primary", use_container_width=True)

# # --- Execute pipeline ---
# if run_btn:
#     if not query_text and not audio_bytes:
#         st.error("Provide either a typed question OR an audio recording/file.")
#         st.stop()

#     try:
#         # Audio path → answer_query(audio_path=...)
#         if audio_bytes and not query_text:
#             suffix = ".webm"
#             if up is not None:
#                 _, ext = os.path.splitext(up.name)
#                 if ext:
#                     suffix = ext
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 tmp.write(audio_bytes)
#                 temp_audio_path = tmp.name

#             with st.spinner("Transcribing + retrieving + synthesizing..."):
#                 resp = answer_query(audio_path=temp_audio_path)
#         else:
#             # Text-only path
#             with st.spinner("Retrieving + synthesizing..."):
#                 resp = answer_query(query=query_text)

#         # Display results
#         st.subheader("Answer")
#         st.write(resp.get("unstructured", ""))

#         st.subheader("Facts / Citations / Confidence")
#         st.json(resp.get("structured", {}).get("model_json", {}))

#         st.subheader("Top Citations")
#         st.json(resp.get("structured", {}).get("citations", []))

#         if show_raw:
#             # ASCII-only dump (qa.py already normalizes, but keep ensure_ascii=True)
#             st.subheader("Raw Response")
#             st.code(json.dumps(resp, indent=2, ensure_ascii=True), language="json")

#     except Exception as e:
#         st.error(f"Pipeline error: {e}")

#     finally:
#         if temp_audio_path and os.path.exists(temp_audio_path):
#             try:
#                 os.remove(temp_audio_path)
#             except Exception:
#                 pass

# st.divider()
# st.caption("Tip: if mic doesn’t appear, try Chrome/Edge and allow microphone permission.")

# ======================================
# Code 3
# ======================================

# import os
# import json
# import tempfile
# import streamlit as st

# from qa import answer_query  # uses your existing pipeline

# st.set_page_config(page_title="Voice Q&A (Local)", page_icon="🎙️", layout="centered")

# st.title("Voice Q&A (Local)")
# st.caption("Speak, upload audio, or type → Whisper → RAG → Answer")

# with st.sidebar:
#     st.header("Input")
#     use_mic = st.toggle("Use Microphone", value=True)
#     st.divider()
#     st.header("Display")
#     show_raw = st.checkbox("Show raw JSON", value=True)
#     st.caption("All processing runs locally; external APIs (OpenAI, Neo4j Aura) use your .env.")

# query_text = st.text_input("Or type a question:", value="")
# audio_bytes = None
# temp_audio_path = None

# # --- Microphone (streamlit-mic-recorder) ---
# if use_mic:
#     try:
#         from streamlit_mic_recorder import mic_recorder
#         st.write("### Microphone")
#         audio_dict = mic_recorder(
#             start_prompt="Start recording",
#             stop_prompt="Stop recording",
#             just_once=False,
#             use_container_width=True,
#             format="webm",  # wav also works; webm is smaller
#         )
#         if audio_dict and audio_dict.get("bytes"):
#             audio_bytes = audio_dict["bytes"]
#             st.success("Captured audio from microphone.")
#             st.audio(audio_bytes, format="audio/webm")
#     except Exception as e:
#         st.warning(f"Microphone component unavailable ({e}). Use file upload or text.")
#         use_mic = False

# # --- File upload (fallback/alternative) ---
# # st.write("### Upload audio file (MP3/WAV/M4A/WEBM)")
# # up = st.file_uploader(
# #     "Choose an audio file",
# #     type=["mp3", "wav", "m4a", "webm"],
# #     accept_multiple_files=False
# # )
# # if up is not None:
# #     audio_bytes = up.read()
# #     st.audio(audio_bytes, format=up.type or "audio")

# run_btn = st.button("Run Q&A", type="primary", use_container_width=True)

# # --- Execute pipeline ---
# if run_btn:
#     if not query_text and not audio_bytes:
#         st.error("Provide either a typed question OR an audio recording/file.")
#         st.stop()

#     try:
#         # Audio path → answer_query(audio_path=...)
#         if audio_bytes and not query_text:
#             suffix = ".webm"
#             if up is not None:
#                 _, ext = os.path.splitext(up.name)
#                 if ext:
#                     suffix = ext
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 tmp.write(audio_bytes)
#                 temp_audio_path = tmp.name

#             with st.spinner("Transcribing + retrieving + synthesizing..."):
#                 resp = answer_query(audio_path=temp_audio_path)
#         else:
#             # Text-only path
#             with st.spinner("Retrieving + synthesizing..."):
#                 resp = answer_query(query=query_text)

#         # Display results
#         st.subheader("Answer")
#         st.write(resp.get("unstructured", ""))

#         st.subheader("Facts / Citations / Confidence")
#         st.json(resp.get("structured", {}).get("model_json", {}))

#         st.subheader("Top Citations")
#         st.json(resp.get("structured", {}).get("citations", []))

#         if show_raw:
#             # ASCII-only dump (qa.py already normalizes, but keep ensure_ascii=True)
#             st.subheader("Raw Response")
#             st.code(json.dumps(resp, indent=2, ensure_ascii=True), language="json")

#     except Exception as e:
#         st.error(f"Pipeline error: {e}")

#     finally:
#         if temp_audio_path and os.path.exists(temp_audio_path):
#             try:
#                 os.remove(temp_audio_path)
#             except Exception:
#                 pass

# st.divider()
# st.caption("Tip: if mic doesn’t appear, try Chrome/Edge and allow microphone permission.")

# ======================================
# code 4
# ======================================
# streamlit_app.py
# import os
# import json
# import tempfile
# import hashlib
# from typing import Dict, Any

# import streamlit as st
# from qa import answer_query  # your existing pipeline (FAISS + Neo4j + OpenAI)

# st.set_page_config(page_title="Voice Q&A (Agentic)", page_icon="🎙️", layout="centered")

# # ----------------------------
# # Session state init
# # ----------------------------
# if "messages" not in st.session_state:
#     # Each item: {"role": "user"|"assistant", "content": str, "meta": Dict[str,Any]}
#     st.session_state.messages = []
# if "last_audio_sig" not in st.session_state:
#     st.session_state.last_audio_sig = None  # md5 signature of last audio bytes
# if "show_raw" not in st.session_state:
#     st.session_state.show_raw = False

# st.title("🎙️ Voice Q&A (Agentic)")
# st.caption("Stop recording → auto-transcribe as **User** → answer as **Assistant**")

# with st.sidebar:
#     st.header("Settings")
#     use_mic = st.toggle("Use Microphone", value=True)
#     enable_text = st.toggle("Enable text chat", value=True)
#     st.session_state.show_raw = st.toggle("Show raw JSON", value=st.session_state.show_raw)
#     st.caption("Tip: When using the mic, allow microphone permission in your browser.")

# # ----------------------------
# # Chat history (top to bottom)
# # ----------------------------
# for m in st.session_state.messages:
#     with st.chat_message(m["role"]):
#         st.write(m["content"])
#         # Show compact citations/facts on assistant turns if available
#         meta = m.get("meta") or {}
#         if m["role"] == "assistant":
#             if "model_json" in meta or "citations" in meta:
#                 with st.expander("Citations / Facts / Confidence"):
#                     if "model_json" in meta:
#                         st.json(meta["model_json"])
#                     if "citations" in meta:
#                         st.json(meta["citations"])

# # ----------------------------
# # Microphone recorder (agentic)
# # ----------------------------
# def _audio_md5(b: bytes) -> str:
#     return hashlib.md5(b).hexdigest()

# temp_audio_path = None
# audio_bytes = None
# audio_source = None  # "mic" or "upload"

# if use_mic:
#     try:
#         from streamlit_mic_recorder import mic_recorder
#         st.write("### 🎤 Microphone")
#         audio_dict = mic_recorder(
#             start_prompt="Start recording",
#             stop_prompt="Stop recording",
#             just_once=False,                # allow multiple takes
#             use_container_width=True,
#             format="webm",                  # small; whisper handles it
#             key="mic_recorder_component",
#         )
#         # When user stops, the component returns bytes once.
#         if audio_dict and audio_dict.get("bytes"):
#             audio_bytes = audio_dict["bytes"]
#             audio_source = "mic"
#             st.audio(audio_bytes, format="audio/webm")
#     except Exception as e:
#         st.warning(f"Microphone not available ({e}). Use file upload or text.")

# # ----------------------------
# # File upload (alternate input)
# # ----------------------------
# st.write("### 📁 Upload audio (MP3/WAV/M4A/WEBM)")
# up = st.file_uploader(
#     "Choose an audio file",
#     type=["mp3", "wav", "m4a", "webm"],
#     accept_multiple_files=False,
#     key="uploader",
# )
# if up is not None:
#     audio_bytes = up.read()
#     audio_source = "upload"
#     st.audio(audio_bytes, format=up.type or "audio")

# # ----------------------------
# # Agentic: auto-run on new audio
# # ----------------------------
# def run_audio_pipeline(bytes_blob: bytes, uploaded_name: str | None = None) -> None:
#     """Transcribe + answer; append to chat as user + assistant."""
#     global temp_audio_path
#     if not bytes_blob:
#         return

#     # dedupe: only run for new audio
#     sig = _audio_md5(bytes_blob)
#     if st.session_state.last_audio_sig == sig:
#         return
#     st.session_state.last_audio_sig = sig

#     # Save to a temp file for qa.answer_query(audio_path=...)
#     suffix = ".webm"
#     if uploaded_name:
#         _, ext = os.path.splitext(uploaded_name)
#         if ext:
#             suffix = ext
#     with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#         tmp.write(bytes_blob)
#         temp_audio_path = tmp.name

#     with st.spinner("Transcribing + retrieving + synthesizing..."):
#         try:
#             resp = answer_query(audio_path=temp_audio_path)
#         except Exception as e:
#             st.error(f"Pipeline error: {e}")
#             return
#         finally:
#             # cleanup temp
#             if temp_audio_path and os.path.exists(temp_audio_path):
#                 try:
#                     os.remove(temp_audio_path)
#                 except Exception:
#                     pass

#     # Add user (transcript) + assistant (answer) messages
#     user_text = resp.get("query", "(unrecognized speech)")
#     asst_text = resp.get("unstructured", "")
#     meta = {
#         "model_json": resp.get("structured", {}).get("model_json", {}),
#         "citations": resp.get("structured", {}).get("citations", []),
#     }

#     st.session_state.messages.append({"role": "user", "content": user_text, "meta": {}})
#     st.session_state.messages.append({"role": "assistant", "content": asst_text, "meta": meta})

#     # Re-render chat immediately
#     st.rerun()

# # Trigger auto pipeline if new audio arrived
# if audio_bytes:
#     run_audio_pipeline(audio_bytes, up.name if audio_source == "upload" and up else None)

# # ----------------------------
# # Text chat (optional)
# # ----------------------------
# if enable_text:
#     prompt = st.chat_input("Type your question…")
#     if prompt:
#         # Show user message immediately
#         st.session_state.messages.append({"role": "user", "content": prompt, "meta": {}})
#         with st.spinner("Retrieving + synthesizing..."):
#             try:
#                 resp = answer_query(query=prompt)
#             except Exception as e:
#                 st.error(f"Pipeline error: {e}")
#                 st.stop()

#         asst_text = resp.get("unstructured", "")
#         meta = {
#             "model_json": resp.get("structured", {}).get("model_json", {}),
#             "citations": resp.get("structured", {}).get("citations", []),
#         }
#         st.session_state.messages.append({"role": "assistant", "content": asst_text, "meta": meta})
#         st.rerun()

# # ----------------------------
# # Optional: show raw JSON for the *last* assistant turn
# # ----------------------------
# if st.session_state.show_raw:
#     # Find last assistant turn with meta
#     for m in reversed(st.session_state.messages):
#         if m["role"] == "assistant":
#             st.divider()
#             st.subheader("Raw (last assistant response)")
#             # Safe ASCII dump
#             st.code(json.dumps(m.get("meta", {}), indent=2, ensure_ascii=True), language="json")
#             break

# st.divider()
# st.caption("Agentic mode: end recording → auto transcribe & answer. No button needed.")
# ==================================
# code 5 --> this is best only thing is no transcription shown
# ==================================    
# import os
# import json
# import tempfile
# import streamlit as st

# from qa import answer_query  # uses your existing pipeline

# st.set_page_config(page_title="Voice Q&A (Local)", page_icon="🎙️", layout="centered")

# st.title("Voice Q&A (Local)")
# st.caption("Speak, upload audio, or type → Whisper → RAG → Answer")

# with st.sidebar:
#     st.header("Input")
#     use_mic = st.toggle("Use Microphone", value=True)
#     st.divider()
#     st.header("Display")
#     show_raw = st.checkbox("Show raw JSON", value=True)
#     st.caption("All processing runs locally; external APIs (OpenAI, Neo4j Aura) use your .env.")

# query_text = st.text_input("Or type a question:", value="")
# audio_bytes = None
# temp_audio_path = None

# # --- Microphone (streamlit-mic-recorder) ---
# if use_mic:
#     try:
#         from streamlit_mic_recorder import mic_recorder
#         st.write("### Microphone")
#         audio_dict = mic_recorder(
#             start_prompt="Start recording",
#             stop_prompt="Stop recording",
#             just_once=False,
#             use_container_width=True,
#             format="webm",  # wav also works; webm is smaller
#         )
#         if audio_dict and audio_dict.get("bytes"):
#             audio_bytes = audio_dict["bytes"]
#             st.success("Captured audio from microphone.")
#             st.audio(audio_bytes, format="audio/webm")
#     except Exception as e:
#         st.warning(f"Microphone component unavailable ({e}). Use file upload or text.")
#         use_mic = False

# # --- File upload (fallback/alternative) ---
# st.write("### Upload audio file (MP3/WAV/M4A/WEBM)")
# up = st.file_uploader(
#     "Choose an audio file",
#     type=["mp3", "wav", "m4a", "webm"],
#     accept_multiple_files=False
# )
# if up is not None:
#     audio_bytes = up.read()
#     st.audio(audio_bytes, format=up.type or "audio")

# run_btn = st.button("Run Q&A", type="primary", use_container_width=True)

# # --- Execute pipeline ---
# if run_btn:
#     if not query_text and not audio_bytes:
#         st.error("Provide either a typed question OR an audio recording/file.")
#         st.stop()

#     try:
#         # Audio path → answer_query(audio_path=...)
#         if audio_bytes and not query_text:
#             suffix = ".webm"
#             if up is not None:
#                 _, ext = os.path.splitext(up.name)
#                 if ext:
#                     suffix = ext
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#                 tmp.write(audio_bytes)
#                 temp_audio_path = tmp.name

#             with st.spinner("Transcribing + retrieving + synthesizing..."):
#                 resp = answer_query(audio_path=temp_audio_path)
#         else:
#             # Text-only path
#             with st.spinner("Retrieving + synthesizing..."):
#                 resp = answer_query(query=query_text)

#         # Display results
#         st.subheader("Answer")
#         st.write(resp.get("unstructured", ""))

#         st.subheader("Facts / Citations / Confidence")
#         st.json(resp.get("structured", {}).get("model_json", {}))

#         st.subheader("Top Citations")
#         st.json(resp.get("structured", {}).get("citations", []))

#         if show_raw:
#             # ASCII-only dump (qa.py already normalizes, but keep ensure_ascii=True)
#             st.subheader("Raw Response")
#             st.code(json.dumps(resp, indent=2, ensure_ascii=True), language="json")

#     except Exception as e:
#         st.error(f"Pipeline error: {e}")

#     finally:
#         if temp_audio_path and os.path.exists(temp_audio_path):
#             try:
#                 os.remove(temp_audio_path)
#             except Exception:
#                 pass

# st.divider()
# st.caption("Tip: if mic doesn’t appear, try Chrome/Edge and allow microphone permission.")

# ==================================
# code 6 --> added transcription
# ==================================
# streamlit_app.py — centered, agentic voice+text chat with once-only auto-transcribe
import os
import json
import hashlib
import tempfile
import streamlit as st
from qa import answer_query  # your existing pipeline (Whisper when audio_path is provided)
# --- tiny in-memory avatars so the OS can't change them ---
from io import BytesIO
from PIL import Image, ImageDraw  # pip install pillow

def _png_bytes_user(accent="#7CFFCB", bg="#1A2233") -> bytes:
    """Simple person silhouette in a rounded square."""
    W = H = 64
    r = 14
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # card background
    d.rounded_rectangle([2, 2, W-2, H-2], radius=r, fill=bg, outline=(90, 100, 120, 120), width=2)

    # head
    d.ellipse([22, 14, 42, 34], fill=accent)
    # shoulders
    d.rounded_rectangle([16, 30, 48, 54], radius=12, fill=accent)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def _png_bytes_bot(accent="#7CFFCB", bg="#1A2233") -> bytes:
    """Cute robot head in a rounded square."""
    W = H = 64
    r = 14
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # card background
    d.rounded_rectangle([2, 2, W-2, H-2], radius=r, fill=bg, outline=(90, 100, 120, 120), width=2)

    # head
    d.rounded_rectangle([14, 16, 50, 48], radius=10, fill=accent)
    # eyes
    d.ellipse([23, 28, 29, 34], fill=bg)
    d.ellipse([35, 28, 41, 34], fill=bg)
    # mouth slit
    d.rounded_rectangle([26, 38, 38, 40], radius=2, fill=bg)
    # tiny antenna
    d.line([32, 10, 32, 16], fill=accent, width=3)
    d.ellipse([30, 6, 34, 10], fill=accent)

    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# choose your accent once (matches your theme)
ACCENT = "#7CFFCB"
USER_AVATAR = _png_bytes_user(ACCENT)
BOT_AVATAR  = _png_bytes_bot(ACCENT)
# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Voice Q&A — Agentic", page_icon="🎙️", layout="wide")

# # ---- Avatars ----
# USER_AVATAR = "🧑"   #  for user
# BOT_AVATAR  = "🤖" 

# # top of file
# USER_AVATAR = "👤"   # bust in silhouette (neutral)
# BOT_AVATAR  = "🤖"   # robot

def render_chat():
    for m in st.session_state.messages:
        role = "user" if m["role"] == "user" else "assistant"
        avatar = USER_AVATAR if role == "user" else BOT_AVATAR
        with st.chat_message(role, avatar=avatar):
            st.markdown(m["text"])
# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def rerun():
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if fn:
        fn()

def fp(b: bytes) -> str:
    """Fingerprint audio bytes to avoid re-processing the same buffer."""
    return hashlib.sha1(b).hexdigest() if b else ""

def process_query(*, text: str | None = None, audio_path: str | None = None):
    """Call your pipeline and push messages into chat."""
    if audio_path:
        result = answer_query(audio_path=audio_path)     # qa.py will Whisper here
        user_text = result.get("query", "")
    else:
        result = answer_query(query=text or "")

    st.session_state.messages.append({"role": "user", "text": result.get("query", text or "")})
    st.session_state.messages.append({"role": "assistant", "text": result.get("unstructured", "")})

    st.session_state.last_structured = result.get("structured", {})
    st.session_state.last_citations = st.session_state.last_structured.get("citations", [])

def process_pending_audio():
    """Run once for captured/uploaded audio if present."""
    pending = st.session_state.get("pending_audio")
    if not pending:
        return
    audio_bytes, suffix = pending
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(audio_bytes)
            tmp = f.name
        with st.spinner("Transcribing with Whisper → Retrieving → Answering…"):
            process_query(audio_path=tmp)
    finally:
        st.session_state.pending_audio = None
        if tmp and os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass

# def render_chat():
#     for m in st.session_state.messages:
#         with st.chat_message("user" if m["role"] == "user" else "assistant",
#                              avatar="🧑" if m["role"] == "user" else "🤖"):
#             st.markdown(m["text"])

# def render_chat():
#     for m in st.session_state.messages:
#         avatar = USER_AVATAR if m["role"] == "user" else BOT_AVATAR
#         with st.chat_message(m["role"], avatar=avatar):
#             st.markdown(m["text"])

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_structured" not in st.session_state:
    st.session_state.last_structured = {}
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None         # (bytes, suffix)
if "last_audio_fp" not in st.session_state:
    st.session_state.last_audio_fp = ""           # to avoid duplicate runs

# -----------------------------------------------------------------------------
# Layout (single centered column)
# -----------------------------------------------------------------------------
pad, main, pad2 = st.columns([1, 2.4, 1])
with main:
    st.title("🎙️ Voice Q&A — Agentic")
    st.caption("Stop recording → auto-transcribe as **User** → answer as **Assistant**.")

    with st.sidebar:
        st.header("Controls")
        mic_on = st.toggle("Enable microphone", value=True)
        text_on = st.toggle("Enable text input", value=True)
        show_debug = st.toggle("Show model JSON / citations", value=False)
        if st.button("Clear chat"):
            st.session_state.messages.clear()
            st.session_state.last_structured = {}
            st.session_state.last_citations = []
            st.session_state.pending_audio = None
            st.session_state.last_audio_fp = ""
            rerun()

    # -------------------- Microphone & Upload --------------------
    with st.container(border=True):
        st.subheader("Input")
        c1, c2 = st.columns(2)

        # Microphone
        with c1:
            st.markdown("**🎤 Microphone**  \nAgentic: on stop, it auto-transcribes & answers.")
            audio_bytes = None
            if mic_on:
                try:
                    from streamlit_mic_recorder import mic_recorder
                    mic = mic_recorder(
                        start_prompt="Tap to record",
                        stop_prompt="Stop",
                        just_once=False,       # allow multiple takes in a session
                        use_container_width=True,
                        format="webm",
                        key="mic",
                    )
                    # On stop we get bytes and recording becomes False
                    if mic and mic.get("bytes") and not mic.get("recording", False):
                        audio_bytes = mic["bytes"]
                        st.audio(audio_bytes, format="audio/webm")

                        current_fp = fp(audio_bytes)
                        if current_fp != st.session_state.last_audio_fp:
                            st.session_state.last_audio_fp = current_fp
                            st.session_state.pending_audio = (audio_bytes, ".webm")
                            # No heavy work in the same frame; process on next run.
                            rerun()
                except Exception as e:
                    st.warning(f"Mic unavailable: {e}")

        # Upload
        # with c2:
        #     st.markdown("**📁 Upload audio**  \nMP3 / WAV / M4A / WEBM")
        #     up = st.file_uploader(
        #         "Choose an audio file",
        #         type=["mp3", "wav", "m4a", "webm"],
        #         accept_multiple_files=False,
        #         label_visibility="collapsed",
        #         key="uploader",
        #     )
        #     if up is not None:
        #         b = up.read()
        #         st.audio(b, format=up.type or "audio")
        #         fprint = fp(b)
        #         if fprint != st.session_state.last_audio_fp:
        #             st.session_state.last_audio_fp = fprint
        #             suffix = os.path.splitext(up.name)[1] or ".webm"
        #             st.session_state.pending_audio = (b, suffix)
        #             rerun()

    # If we have something to process, do it now (exactly once)
    process_pending_audio()

    # -------------------- Chat Transcript --------------------
    st.divider()
    render_chat()

    # -------------------- Text box --------------------
    if text_on:
        st.divider()
        tcol, bcol = st.columns([5, 1])
        with tcol:
            txt = st.text_input("Ask a question…", value="", label_visibility="collapsed", key="text_input")
        with bcol:
            send = st.button("Send", use_container_width=True)

        if (txt and txt.strip()) or send:
            if txt.strip():
                with st.spinner("Retrieving → Answering…"):
                    process_query(text=txt.strip())
                st.session_state.text_input = ""
                rerun()

    # -------------------- Debug --------------------
    if show_debug:
        st.divider()
        st.subheader("Facts / Citations / Confidence")
        st.json(st.session_state.last_structured or {})
        st.subheader("Top Citations")
        st.json(st.session_state.last_citations or {})

    st.divider()
    st.caption("Built with Whisper STT + retrieval. All calls use your local .env.")