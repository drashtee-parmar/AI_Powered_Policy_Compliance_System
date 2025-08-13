# #!/usr/bin/env python3
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