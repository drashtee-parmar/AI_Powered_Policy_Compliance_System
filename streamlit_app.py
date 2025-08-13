import streamlit as st
import tempfile
from graph_rag_voice.ingestion import ingest_folder
from graph_rag_voice.rag import answer_question, answer_question_hybrid
from graph_rag_voice.stt import transcribe
from graph_rag_voice.config import settings

st.set_page_config(page_title='Graph RAG Voice Q&A', layout='centered')
st.title('Graph RAG • Voice Q&A')

with st.expander('Ingest documents (.txt/.md/.docx/.pdf)'):
    folder = st.text_input('Folder path', './sample_docs')
    if st.button('Ingest Now'):
        with st.spinner('Ingesting...'):
            ingest_folder(folder)
        st.success('Ingestion complete.')

st.markdown('---')
mode = st.radio('Ask mode', ['Text', 'Audio'], horizontal=True)

question = ''
if mode == 'Text':
    question = st.text_input('Your question', value='What did Sarah say about the compliance deadline?')
else:
    audio = st.file_uploader('Upload audio file (.wav/.mp3/.m4a)', type=['wav','mp3','m4a'])
    if st.button('Transcribe'):
        if audio is None:
            st.warning('Upload an audio file first.')
        else:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(audio.read()); tmp.flush()
                text = transcribe(tmp.name)
            st.session_state['last_transcript'] = text
    question = st.text_area('Transcript (editable)', value=st.session_state.get('last_transcript',''))

use_hybrid = st.checkbox('Use Hybrid Retrieval (Graph + FAISS)', value=True)

if st.button('Ask'):
    if not question.strip():
        st.warning('Enter a question or transcribe audio first.')
    else:
        with st.spinner('Retrieving...'):
            res = answer_question_hybrid(question) if (use_hybrid and settings.enable_hybrid) else answer_question(question)
        st.subheader('Answer')
        st.write(res['answer'])
        st.caption(f"Entities: {res.get('entities')} • Chunks: {res.get('num_chunks')} • Hybrid: {res.get('hybrid', False)}")
