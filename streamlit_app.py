#!/usr/bin/env python3
import os
import tempfile
import json
import streamlit as st

# Your pipeline entrypoint
from qa import answer_query

st.set_page_config(page_title="Voice Q&A (Local)", page_icon="🎙️", layout="centered")

st.title("Voice Q&A (Local)")
st.caption("Speak or upload audio → Whisper → RAG → Answer")

# --- Sidebar ---
with st.sidebar:
    st.header("Input")
    use_mic = st.toggle("Use Microphone", value=True, help="Disable to use file upload only")
    st.divider()
    st.header("Options")
    show_raw = st.checkbox("Show raw JSON", value=True)
    st.caption("All processing runs locally; your APIs (OpenAI, Neo4j Aura) are used as configured in .env.")

# --- Main UI ---
query_text = st.text_input("Or type a question (bypasses mic):", value="")

audio_bytes = None
temp_audio_path = None

if use_mic:
    # Mic component: returns a button that records and returns bytes
    try:
        from streamlit_mic_recorder import mic_recorder

        st.write("### Microphone")
        audio_dict = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            use_container_width=True,
            format="webm",  # 'wav' also supported; webm is typically smaller
        )
        if audio_dict and audio_dict.get("bytes"):
            audio_bytes = audio_dict["bytes"]
            st.success("Captured audio from microphone.")
            st.audio(audio_bytes, format="audio/webm")

    except Exception as e:
        st.warning(f"Microphone component unavailable ({e}). Use file upload below.")
        use_mic = False

# Fallback / alternative: file upload
st.write("### Upload audio file (MP3/WAV/M4A/WEBM)")
up = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "webm"], accept_multiple_files=False)
if up is not None:
    audio_bytes = up.read()
    st.audio(audio_bytes, format=f"audio/{(up.type or 'audio')}")
    st.info(f"Loaded file: {up.name}")

run_btn = st.button("Run Q&A", type="primary", use_container_width=True)

# --- Execute pipeline ---
if run_btn:
    if not query_text and not audio_bytes:
        st.error("Provide either a typed question OR an audio recording/file.")
        st.stop()

    try:
        # If we have audio bytes, persist to a temp file and call qa.answer_query(audio_path=...)
        if audio_bytes and not query_text:
            suffix = ".webm"
            if up is not None:
                # preserve file extension when uploaded
                _, ext = os.path.splitext(up.name)
                if ext:
                    suffix = ext
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                temp_audio_path = tmp.name

            with st.spinner("Transcribing + retrieving + synthesizing..."):
                resp = answer_query(audio_path=temp_audio_path)

        else:
            # Text-only path (no audio)
            with st.spinner("Retrieving + synthesizing..."):
                resp = answer_query(query=query_text)

        # Display result
        st.subheader("Answer")
        st.write(resp.get("unstructured", ""))

        st.subheader("Facts / Citations / Confidence")
        model_json = resp.get("structured", {}).get("model_json", {})
        st.json(model_json)

        st.subheader("Top Citations (clean)")
        st.json(resp.get("structured", {}).get("citations", []))

        if show_raw:
            st.subheader("Raw Response")
            st.code(json.dumps(resp, indent=2, ensure_ascii=True), language="json")

    except Exception as e:
        st.error(f"Pipeline error: {e}")

    finally:
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass

st.divider()
st.caption("Tip: if the mic doesn’t appear, ensure browser permission is granted and try another browser.")