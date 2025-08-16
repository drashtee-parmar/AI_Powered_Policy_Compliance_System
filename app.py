# #!/usr/bin/env python3
# import html
# import os
# import tempfile
# from typing import Any, Dict, List, Optional

# import streamlit as st
# from qa import answer_query

# # ---------- Page setup ----------
# st.set_page_config(page_title="Policy Q&A", page_icon="📄", layout="wide")

# # ---------- Styles ----------
# st.markdown(
#     """
#     <style>
#       .app-title { text-align:center; margin: 10px 0 2px 0; }
#       .app-sub   { text-align:center; color:#888; margin: 0 0 24px 0; }
#       .stChatMessage { max-width: 1100px; margin-left:auto; margin-right:auto; }
#       .stChatMessage [data-testid="stChatMessageContent"] { padding: 0.9rem 1.0rem; }
#       .stChatMessage [data-testid="stChatMessageAvatar"] { transform: scale(1.1); }
#       .stTextInput>div>div>input { font-size: 1.05rem; height: 3rem; }
#       .ans-paragraph {
#         margin: 0 0 0.65rem 0; line-height: 1.55; font-size: 1.06rem;
#         white-space: normal; word-break: normal; overflow-wrap: break-word;
#       }
#       .ans-list { margin: 0.25rem 0 0.7rem 1.25rem; }
#       .ans-list li { margin: 0.2rem 0; line-height: 1.55; font-size: 1.04rem; }
#       .citation-excerpt { white-space: pre-wrap; font-size: 0.95rem; line-height: 1.45; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # ---------- Header ----------
# st.markdown("<h1 class='app-title'>Policy Q&A</h1>", unsafe_allow_html=True)
# st.markdown("<div class='app-sub'>Ask by voice or text. The system retrieves, grounds, and answers.</div>", unsafe_allow_html=True)

# # ---------- Sidebar ----------
# with st.sidebar:
#     st.header("Session")
#     use_mic = st.toggle("Use Microphone", value=True)
#     if st.button("Clear Chat", use_container_width=True):
#         st.session_state.clear()
#         st.rerun()
#     st.caption("Tip: ensure your `.env` has clean `KEY=value` lines only.")

# # ---------- Session state ----------
# if "history" not in st.session_state:
#     st.session_state.history: List[Dict[str, Any]] = []
# if "pending_audio_bytes" not in st.session_state:
#     st.session_state.pending_audio_bytes: Optional[bytes] = None

# # ---------- Helpers ----------
# def _render_plain_answer(text: str) -> None:
#     if not text:
#         return
#     lines = [ln.rstrip() for ln in text.splitlines()]
#     blocks: List[str] = []
#     buf: List[str] = []
#     in_list = False
#     list_items: List[str] = []

#     def flush_para():
#         nonlocal buf, blocks
#         if buf:
#             p = html.escape(" ".join(buf).strip())
#             if p:
#                 blocks.append(f"<p class='ans-paragraph'>{p}</p>")
#             buf = []

#     def flush_list(items: List[str]) -> str:
#         if not items:
#             return ""
#         lis = "".join(f"<li>{html.escape(x.strip())}</li>" for x in items if x.strip())
#         return f"<ul class='ans-list'>{lis}</ul>"

#     for ln in lines:
#         if ln.strip().startswith("- "):
#             if not in_list:
#                 flush_para()
#                 in_list = True
#             list_items.append(ln.strip()[2:])
#         elif ln.strip() == "":
#             if in_list:
#                 blocks.append(flush_list(list_items)); list_items = []; in_list = False
#             else:
#                 flush_para()
#         else:
#             if in_list:
#                 blocks.append(flush_list(list_items)); list_items = []; in_list = False
#             buf.append(ln)

#     if in_list:
#         blocks.append(flush_list(list_items))
#     else:
#         flush_para()

#     st.markdown("".join(blocks), unsafe_allow_html=True)

# def _show_sources(citations: List[Dict[str, Any]], model_json: Any) -> None:
#     if not isinstance(model_json, dict):
#         model_json = {}
#     facts = model_json.get("facts") or []
#     conf = model_json.get("confidence", None)

#     with st.expander(f"Details (sources: {len(citations)})", expanded=False):
#         if facts:
#             st.markdown("**Facts**")
#             for f in facts:
#                 st.markdown(f"- {html.escape(str(f))}", unsafe_allow_html=True)
#         if conf is not None:
#             st.markdown(f"**Confidence:** `{conf}`")
#         if citations:
#             st.markdown("**Top Citations**")
#             for i, c in enumerate(citations, 1):
#                 file = c.get("file", "?")
#                 score = c.get("score", 0.0)
#                 excerpt = c.get("excerpt", "")
#                 with st.expander(f"{i}. {file} — score {score}", expanded=False):
#                     st.markdown("**Excerpt**")
#                     st.markdown(
#                         f"<div class='citation-excerpt'>{html.escape(excerpt)}</div>",
#                         unsafe_allow_html=True,
#                     )

# def _run_audio(audio_bytes: bytes) -> Dict[str, Any]:
#     """Transcribe + RAG + LLM for audio input."""
#     tmp_path = None
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
#             tmp.write(audio_bytes)
#             tmp_path = tmp.name
#         with st.spinner("Transcribing (Whisper) → Retrieving → Synthesizing…"):
#             return answer_query(audio_path=tmp_path)
#     finally:
#         if tmp_path and os.path.exists(tmp_path):
#             try: os.remove(tmp_path)
#             except Exception: pass

# # ---------- Render prior turns ----------
# for t in st.session_state.history:
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(t["user_text"])
#     with st.chat_message("assistant", avatar="🅐"):
#         _render_plain_answer(t["assistant_text"])
#         _show_sources(t.get("citations", []), t.get("facts_block", {}))

# # ---------- Microphone (optional) ----------
# if use_mic:
#     try:
#         # pip name: streamlit-mic-recorder ; import name: st_mic_recorder
#         from st_mic_recorder import mic_recorder

#         st.subheader("Microphone")
#         mic = mic_recorder(
#             start_prompt="Start recording",
#             stop_prompt="Stop recording",
#             just_once=False,
#             use_container_width=True,
#             format="webm",
#             key="mic1",
#         )
#         if mic and mic.get("bytes"):
#             st.success("Captured audio.")
#             st.audio(mic["bytes"], format="audio/webm")
#             st.session_state.pending_audio_bytes = mic["bytes"]
#     except Exception as e:
#         st.warning(
#             f"Microphone unavailable ({e}). "
#             "Serve via HTTPS or localhost and ensure `streamlit-mic-recorder` is installed."
#         )

# # If we have fresh audio, process it and show a turn
# if st.session_state.pending_audio_bytes:
#     audio_bytes = st.session_state.pending_audio_bytes
#     st.session_state.pending_audio_bytes = None
#     try:
#         resp = _run_audio(audio_bytes)
#         user_text = resp.get("query", "").strip() or "(voice message)"
#         with st.chat_message("user", avatar="👤"):
#             st.markdown(user_text)

#         assistant_text = resp.get("unstructured", "")
#         citations = resp.get("structured", {}).get("citations", [])
#         facts_block = resp.get("structured", {}).get("model_json", {}) or {}
#         if not isinstance(facts_block, dict):
#             facts_block = {}

#         with st.chat_message("assistant", avatar="🅐"):
#             _render_plain_answer(assistant_text)
#             _show_sources(citations, facts_block)

#         st.session_state.history.append({
#             "user_text": user_text,
#             "assistant_text": assistant_text,
#             "citations": citations,
#             "facts_block": facts_block,
#         })
#     except Exception as e:
#         st.error(f"Pipeline error (audio): {e}")

# # ---------- Text input ----------
# prompt = st.chat_input("Type your question and press Enter…")
# if prompt:
#     with st.chat_message("user", avatar="👤"):
#         st.markdown(prompt)

#     try:
#         with st.spinner("Retrieving → Synthesizing…"):
#             resp = answer_query(query=prompt)  # only passes query
#     except Exception as e:
#         st.error(f"Pipeline error: {e}")
#     else:
#         assistant_text = resp.get("unstructured", "")
#         citations = resp.get("structured", {}).get("citations", [])
#         facts_block = resp.get("structured", {}).get("model_json", {}) or {}
#         if not isinstance(facts_block, dict):
#             facts_block = {}

#         with st.chat_message("assistant", avatar="🅐"):
#             _render_plain_answer(assistant_text)
#             _show_sources(citations, facts_block)

#         st.session_state.history.append({
#             "user_text": prompt,
#             "assistant_text": assistant_text,
#             "citations": citations,
#             "facts_block": facts_block,
#         })

# app.py
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