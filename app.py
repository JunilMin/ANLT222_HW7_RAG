import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import answer_question, load_vectorstore, OFF_TOPIC_RESPONSE

load_dotenv()

st.set_page_config(page_title="Yosemite RAG App", layout="wide")
st.title("Yosemite Guide RAG App")
st.caption("Ask questions about the provided PDF only.")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY was not found.")
    st.stop()

index_dir = Path("faiss_index")
index_file = index_dir / "index.faiss"
meta_file = index_dir / "index.pkl"

if not index_file.exists() or not meta_file.exists():
    st.error("Preprocessed FAISS files are missing. Run `python preprocess_pdf.py` first.")
    st.stop()

vectorstore = load_vectorstore()

question = st.text_input("Enter your question about the document:")

if question:
    answer, sources = answer_question(question, vectorstore)

    st.subheader("Assistant")
    st.write(answer)

    if answer != OFF_TOPIC_RESPONSE and sources:
        st.subheader("Sources")
        for i, source in enumerate(sources, start=1):
            st.markdown(f"**Source {i} ({source['page']})**")
            st.write(source["content"])