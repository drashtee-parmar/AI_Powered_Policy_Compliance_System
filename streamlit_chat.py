#!/usr/bin/env python3
import os
import json
import html
import tempfile
from typing import Dict, Any

import streamlit as st
from qa import answer_query  # your existing pipeline

st.set_page_config(page_title="Policy Q&A — Chat", page_icon="💬", layout="wide")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("Input")
    use_mic = st.toggle("Use Microphone", value=True)
    st.markdown("---")
    if st.button("Clear Chat", type="secondary", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.caption("Local voice or text → Whisper → RAG → Answer")

# ---------------------------
# Session state
# ---------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # list of turns
if "pending_audio_bytes" not in st.session_state:
    st.session_state.pending_audio_bytes = None

# ---------------------------
# Header
# ---------------------------
st.title("Policy Q&A — Chat")
st.caption("Ask by voice or text. Answers include citations and confidence. Expand a turn to view details.")

# ---------------------------
# Microphone (no upload mode)
# ---------------------------
if use_mic:
    try:
        from streamlit_mic_recorder import mic_recorder
        st.write("### Microphone")
        mic = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            use_container_width=True,
            format="webm",
            key="mic1",
        )
        if mic and mic.get("bytes"):
            st.success("Captured audio.")
            st.audio(mic["bytes"], format="audio/webm")
            st.session_state.pending_audio_bytes = mic["bytes"]
    except Exception as e:
        st.warning(f"Microphone unavailable ({e}).")

# ---------------------------
# Safe plain-text rendering (no Markdown interpretation)
# ---------------------------
def render_plain_text(s: str):
    st.markdown(f"<div style='white-space:normal'>{html.escape(s)}</div>", unsafe_allow_html=True)

# ---------------------------
# Chat history renderer
# ---------------------------
def render_turn(turn: Dict[str, Any]):
    with st.chat_message("user"):
        render_plain_text(turn["user_text"])
    with st.chat_message("assistant"):
        render_plain_text(turn["assistant_text"])
        with st.expander("Details", expanded=False):
            model_json = turn.get("assistant_struct", {}).get("model_json", {})
            st.write("**Facts / Citations / Confidence**")
            st.json(model_json)
            st.write("**Top Citations**")
            st.json(turn.get("assistant_struct", {}).get("citations", []))

for t in st.session_state.history:
    render_turn(t)

# ---------------------------
# Helpers
# ---------------------------
def run_audio_query(audio_bytes: bytes) -> Dict[str, Any]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        with st.spinner("Transcribing (Whisper) → Retrieving → Synthesizing…"):
            return answer_query(audio_path=tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def run_text_query(text: str) -> Dict[str, Any]:
    with st.spinner("Retrieving → Synthesizing…"):
        return answer_query(query=text)

# ---------------------------
# Dispatch: audio first, then text
# ---------------------------
if st.session_state.pending_audio_bytes:
    try:
        resp = run_audio_query(st.session_state.pending_audio_bytes)
        user_text = resp.get("query", "_(voice message)_")  # transcribed question
        turn = {
            "user_text": user_text,
            "assistant_text": resp.get("unstructured", ""),
            "assistant_struct": resp.get("structured", {}),
        }
        st.session_state.history.append(turn)
        render_turn(turn)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
    finally:
        st.session_state.pending_audio_bytes = None

prompt = st.chat_input("Type your question and press Enter…")
if prompt:
    try:
        resp = run_text_query(prompt)
        turn = {
            "user_text": prompt,
            "assistant_text": resp.get("unstructured", ""),
            "assistant_struct": resp.get("structured", {}),
        }
        st.session_state.history.append(turn)
        render_turn(turn)
    except Exception as e:
        st.error(f"Pipeline error: {e}")