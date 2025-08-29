# ================================
# code: streamlit_app.py
# ================================
from doctest import debug
import os
import json
import hashlib
import html
import tempfile
from io import BytesIO
import streamlit as st
from PIL import Image, ImageDraw  # pip install pillow
from aipc.qa import (
    answer_query,
    clean_text_output,
)  # your existing pipeline (Whisper when audio_path is provided)



# ------------------------------------------------------------
# Tiny in-memory PNG avatars (so they don’t depend on the OS)
# ------------------------------------------------------------
def _png_bytes_user(accent="#7CFFCB", bg="#1A2233") -> bytes:
    W = H = 64
    r = 14
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [2, 2, W - 2, H - 2], radius=r, fill=bg, outline=(90, 100, 120, 120), width=2
    )
    d.ellipse([22, 14, 42, 34], fill=accent)  # head
    d.rounded_rectangle([16, 30, 48, 54], radius=12, fill=accent)  # shoulders
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_bytes_bot(accent="#7CFFCB", bg="#1A2233") -> bytes:
    W = H = 64
    r = 14
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(
        [2, 2, W - 2, H - 2], radius=r, fill=bg, outline=(90, 100, 120, 120), width=2
    )
    d.rounded_rectangle([14, 16, 50, 48], radius=10, fill=accent)  # head
    d.ellipse([23, 28, 29, 34], fill=bg)
    d.ellipse([35, 28, 41, 34], fill=bg)  # eyes
    d.rounded_rectangle([26, 38, 38, 40], radius=2, fill=bg)  # mouth
    d.line([32, 10, 32, 16], fill=accent, width=3)
    d.ellipse([30, 6, 34, 10], fill=accent)  # antenna
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


ACCENT = "#7CFFCB"
USER_AVATAR = _png_bytes_user(ACCENT)
BOT_AVATAR = _png_bytes_bot(ACCENT)

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Voice Q&A — Agentic", page_icon="🎙️", layout="wide")


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def rerun():
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", lambda: None))()


def fp(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest() if b else ""


def process_query(*, text: str | None = None, audio_path: str | None = None):
    """Calls your pipeline and pushes messages into chat."""
    if audio_path:
        result = answer_query(audio_path=audio_path)  # Whisper path inside qa.py
        user_text = result.get("query", "")
    else:
        result = answer_query(query=text or "")
        user_text = result.get("query", text or "")

    st.session_state.messages.append({"role": "user", "text": user_text})
    st.session_state.messages.append(
        {"role": "assistant", "text": result.get("unstructured", "")}
    )
    st.session_state.last_structured = result.get("structured", {}) or {}
    st.session_state.last_citations = st.session_state.last_structured.get(
        "citations", []
    )


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
#         role = "user" if m["role"] == "user" else "assistant"
#         avatar = USER_AVATAR if role == "user" else BOT_AVATAR
#         with st.chat_message(role, avatar=avatar):
#             # st.markdown(m["text"])
#             st.markdown(clean_text_output(m["text"]))

def render_chat():
    for m in st.session_state.messages:
        role = "user" if m["role"] == "user" else "assistant"
        avatar = USER_AVATAR if role == "user" else BOT_AVATAR
        with st.chat_message(role, avatar=avatar):
            # m["text"] was already sanitized in qa.py, but defensive escape is harmless
            txt = m["text"]
            st.markdown(f"<div style='white-space:pre-wrap'>{txt}</div>", unsafe_allow_html=True)
# ------------------------------------------------------------
# State init
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_structured" not in st.session_state:
    st.session_state.last_structured = {}
if "last_citations" not in st.session_state:
    st.session_state.last_citations = []
if "pending_audio" not in st.session_state:
    st.session_state.pending_audio = None  # (bytes, suffix)
if "last_audio_fp" not in st.session_state:
    st.session_state.last_audio_fp = ""  # avoid reprocessing same buffer

# ------------------------------------------------------------
# Layout (centered content column)
# ------------------------------------------------------------
pad, main, pad2 = st.columns([1, 2.6, 1])

with st.sidebar:
    st.header("Controls")
    mic_on = st.toggle("Enable microphone", value=True)
    show_dbg = st.toggle("Show model JSON / citations", value=False)
    if st.button("Clear chat"):
        st.session_state.messages.clear()
        st.session_state.last_structured = {}
        st.session_state.last_citations = []
        st.session_state.pending_audio = None
        st.session_state.last_audio_fp = ""
        rerun()

with main:
    st.title("Voice Q&A — Agentic")
    st.caption(
        "Stop recording → auto-transcribe as **User** → answer as **Assistant**."
    )

    # -------------------- Input card --------------------
    with st.container(border=True):
        st.subheader("Input")

        st.markdown(
            "**🎤 Microphone**  \nAgentic: on stop, it auto-transcribes & answers."
        )
        audio_bytes = None

        if mic_on:
            try:
                from streamlit_mic_recorder import mic_recorder

                mic = mic_recorder(
                    start_prompt="Tap to record",
                    stop_prompt="Stop",
                    just_once=False,
                    use_container_width=True,
                    format="webm",
                    key="mic",
                )
                # When user stops, we get bytes and recording becomes False
                if mic and mic.get("bytes") and not mic.get("recording", False):
                    audio_bytes = mic["bytes"]
                    st.audio(audio_bytes, format="audio/webm")

                    current_fp = fp(audio_bytes)
                    if current_fp != st.session_state.last_audio_fp:
                        st.session_state.last_audio_fp = current_fp
                        st.session_state.pending_audio = (audio_bytes, ".webm")
                        # Schedule processing on the next run so the UI never blocks
                        rerun()
            except Exception as e:
                st.warning(f"Microphone unavailable: {e}")

    # If we have pending audio, process it once
    process_pending_audio()

    st.divider()
    render_chat()

    # -------------------- Text input (simple, auto-clearing) --------------------
    # Use chat_input so we never mutate widget state manually.
    prompt = st.chat_input("Ask a question…")
    if prompt:
        with st.spinner("Retrieving → Answering…"):
            process_query(text=prompt)
        rerun()

    # -------------------- Debug panel --------------------
    if show_dbg:
        st.divider()
        # st.subheader("Facts / Citations / Confidence")
        # st.json(st.session_state.last_structured or {})
        st.subheader("Top Citations")
        for i, c in enumerate(st.session_state.last_citations or []):
            with st.expander(
                f"{i+1}. {c.get('file','?')} — score {c.get('score',0.0):.4f}",
                expanded=False,
            ):
                st.write(c.get("excerpt", ""))
        # st.json(st.session_state.last_citations or {})

    st.divider()
    st.caption("Built with Whisper STT + retrieval. All calls use your local .env.")
