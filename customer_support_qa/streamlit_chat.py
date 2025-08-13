#!/usr/bin/env python3
import os
import html
import tempfile
from typing import Dict, Any, List, Optional

import streamlit as st
from qa import answer_query

st.set_page_config(page_title="Policy Q&A — Chat", page_icon="💬", layout="wide")

# Sidebar
with st.sidebar:
    st.title("Microphone")
    use_mic = st.toggle("Use Microphone", value=True)
    st.markdown("---")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.clear(); st.rerun()
    st.caption("Local voice or text → Whisper → RAG → Answer")

# State
if "history" not in st.session_state:
    st.session_state.history: List[Dict[str, Any]] = []
if "pending_audio_bytes" not in st.session_state:
    st.session_state.pending_audio_bytes: Optional[bytes] = None

# Header
st.title("Policy Q&A — Chat")
st.caption("Ask by voice or text. Expand a turn to view facts, confidence, and citation excerpts.")

# CSS – prevent per-character wrapping; keep normal paragraph flow
st.markdown(
    """
    <style>
      html, body, [class^="css"]  { font-size: 16px; }
      .stChatMessage { max-width: 1180px; margin-left:auto; margin-right:auto; }
      .stChatMessage [data-testid="stChatMessageContent"] { padding: 0.95rem 1.05rem; }
      .stChatMessage [data-testid="stChatMessageAvatar"] { transform: scale(1.25); }
      .stTextInput>div>div>input { font-size: 1.05rem; height: 3rem; }

      .ans-paragraph {
        margin: 0 0 0.7rem 0;
        line-height: 1.6;
        font-size: 1.06rem;
        white-space: normal;         /* allow normal wrapping */
        word-break: normal;          /* do not break every character */
        overflow-wrap: break-word;   /* break long tokens if needed */
      }
      .ans-list { margin: 0.25rem 0 0.7rem 1.2rem; }
      .ans-list li { margin: 0.2rem 0; line-height: 1.6; font-size: 1.04rem; }

      .citation-excerpt { white-space: pre-wrap; font-size: 0.95rem; line-height: 1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Microphone
if use_mic:
    try:
        from streamlit_mic_recorder import mic_recorder
        st.write("### Microphone")
        mic = mic_recorder(
            start_prompt="Start recording", stop_prompt="Stop recording",
            just_once=False, use_container_width=True, format="webm", key="mic1",
        )
        if mic and mic.get("bytes"):
            st.success("Captured audio.")
            st.audio(mic["bytes"], format="audio/webm")
            st.session_state.pending_audio_bytes = mic["bytes"]
    except Exception as e:
        st.warning(f"Microphone unavailable ({e}).")

# Backend callers
def _run_audio(audio_bytes: bytes) -> Dict[str, Any]:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
            tmp.write(audio_bytes); tmp_path = tmp.name
        with st.spinner("Transcribing (Whisper) → Retrieving → Synthesizing…"):
            return answer_query(audio_path=tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass

def _run_text(text: str) -> Dict[str, Any]:
    with st.spinner("Retrieving → Synthesizing…"):
        return answer_query(query=text)

# Plain HTML answer renderer (no Markdown interpretation)
def render_plain_answer(text: str):
    if not text:
        return
    lines = [ln.rstrip() for ln in text.splitlines()]
    blocks: List[str] = []
    buf: List[str] = []
    in_list = False
    list_items: List[str] = []

    def flush_para():
        nonlocal buf, blocks
        if buf:
            p = html.escape(" ".join(buf).strip())
            if p:
                blocks.append(f"<p class='ans-paragraph'>{p}</p>")
            buf = []

    def flush_list(items: List[str]):
        if not items:
            return ""
        lis = "".join(f"<li>{html.escape(x.strip())}</li>" for x in items if x.strip())
        return f"<ul class='ans-list'>{lis}</ul>"

    for ln in lines:
        if ln.strip().startswith("- "):
            if not in_list:
                flush_para()
                in_list = True
            list_items.append(ln.strip()[2:])
        elif ln.strip() == "":
            if in_list:
                blocks.append(flush_list(list_items)); list_items = []; in_list = False
            else:
                flush_para()
        else:
            if in_list:
                blocks.append(flush_list(list_items)); list_items = []; in_list = False
            buf.append(ln)

    if in_list:
        blocks.append(flush_list(list_items))
    else:
        flush_para()

    st.markdown("".join(blocks), unsafe_allow_html=True)

def show_sources(citations: List[Dict[str, Any]], model_json: Any):
    if not isinstance(model_json, dict):
        model_json = {}
    facts = model_json.get("facts") or []
    conf  = model_json.get("confidence", None)
    src_count = len(citations)

    with st.expander(f"Details (sources: {src_count})", expanded=False):
        if facts:
            st.markdown("**Facts**")
            for f in facts:
                st.markdown(f"- {html.escape(str(f))}", unsafe_allow_html=True)
        if conf is not None:
            st.markdown(f"**Confidence:** `{conf}`")
        if citations:
            st.markdown("**Top Citations**")
            for i, c in enumerate(citations, 1):
                file = c.get("file", "?"); score = c.get("score", 0.0)
                excerpt = c.get("excerpt", "")
                with st.expander(f"{i}. {file} — score {score}", expanded=False):
                    st.markdown("**Excerpt**")
                    st.markdown(f"<div class='citation-excerpt'>{html.escape(excerpt)}</div>", unsafe_allow_html=True)

# Render history
for t in st.session_state.history:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(t["user_text"])
    with st.chat_message("assistant", avatar="🧠"):
        render_plain_answer(t["assistant_text"])
        show_sources(t.get("citations", []), t.get("facts_block", {}))

# Handle pending audio
if st.session_state.pending_audio_bytes:
    try:
        with st.chat_message("assistant", avatar="🧠"):
            ph = st.empty(); ph.markdown("_(listening & thinking…)_")
        resp = _run_audio(st.session_state.pending_audio_bytes)
        ph.empty()

        turn = {
            "user_text": resp.get("query", "(voice message)"),
            "assistant_text": resp.get("unstructured", ""),
            "citations": resp.get("structured", {}).get("citations", []),
            "facts_block": resp.get("structured", {}).get("model_json", {}) or {},
        }
        if not isinstance(turn["facts_block"], dict):
            turn["facts_block"] = {}
        st.session_state.history.append(turn)

        with st.chat_message("user", avatar="🧑‍💻"): st.markdown(turn["user_text"])
        with st.chat_message("assistant", avatar="🧠"):
            render_plain_answer(turn["assistant_text"])
            show_sources(turn["citations"], turn["facts_block"])
    finally:
        st.session_state.pending_audio_bytes = None

# Text input
prompt = st.chat_input("Type your question and press Enter…")
if prompt:
    with st.chat_message("user", avatar="🧑‍💻"): st.markdown(prompt)
    with st.chat_message("assistant", avatar="🧠"):
        ph = st.empty(); ph.markdown("_(thinking…)_")
        try:
            resp = _run_text(prompt)
        except Exception as e:
            ph.empty(); st.error(f"Pipeline error: {e}")
        else:
            ph.empty()
            turn = {
                "user_text": prompt,
                "assistant_text": resp.get("unstructured", ""),
                "citations": resp.get("structured", {}).get("citations", []),
                "facts_block": resp.get("structured", {}).get("model_json", {}) or {},
            }
            if not isinstance(turn["facts_block"], dict):
                turn["facts_block"] = {}
            st.session_state.history.append(turn)
            render_plain_answer(turn["assistant_text"])
            show_sources(turn["citations"], turn["facts_block"])