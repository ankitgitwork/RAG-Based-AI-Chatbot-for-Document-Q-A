"""
Live demo of the RAG-Based AI Chatbot for Document Q&A — free/local-model
version for public deployment (no API key, no cost).

Swaps the production OpenAI embeddings/chat model (see app/rag_chain.py,
app/ingestion.py) for a fully local stack so anyone can try it without
credentials:
  - Embeddings: sentence-transformers/all-MiniLM-L6-v2
  - Vector search: FAISS (in-memory, per-session)
  - Generation: google/flan-t5-base (local, CPU)

Same RAG pattern as the production app: chunk -> embed -> retrieve ->
ground the answer in retrieved context -> generate.
"""

import io

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄", layout="wide")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4


@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=256)
    return embedder, generator


def chunk_text(text: str, source: str):
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def extract_text(uploaded_file) -> str:
    if uploaded_file.name.lower().endswith(".pdf"):
        reader = PdfReader(io.BytesIO(uploaded_file.read()))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return uploaded_file.read().decode("utf-8", errors="ignore")


st.title("📄 RAG-Based AI Chatbot for Document Q&A")
st.caption(
    "Upload one or more documents (PDF/TXT), then ask questions. Answers are "
    "grounded only in retrieved passages from what you upload — this demo runs "
    "fully on free, local, open-source models (no API key, no cost). "
    "The production version ([GitHub repo](https://github.com/ankitgitwork/RAG-Based-AI-Chatbot-for-Document-Q-A)) "
    "swaps in OpenAI embeddings + GPT-4o-mini via a FastAPI backend for higher answer quality."
)

embedder, generator = load_models()

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = []
    st.session_state.history = []

with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader(
        "PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Build index", type="primary", disabled=not files):
        with st.spinner("Chunking and embedding documents..."):
            all_chunks = []
            for f in files:
                text = extract_text(f)
                all_chunks.extend(chunk_text(text, f.name))
            embeddings = embedder.encode(
                [c["text"] for c in all_chunks], show_progress_bar=False
            )
            embeddings = np.array(embeddings).astype("float32")
            faiss.normalize_L2(embeddings)
            index = faiss.IndexFlatIP(embeddings.shape[1])
            index.add(embeddings)
            st.session_state.index = index
            st.session_state.chunks = all_chunks
            st.session_state.history = []
        st.success(f"Indexed {len(all_chunks)} chunks from {len(files)} document(s).")

    if st.session_state.chunks:
        st.caption(f"✅ {len(st.session_state.chunks)} chunks indexed and ready.")

st.header("2. Ask a question")

if st.session_state.index is None:
    st.info("Upload documents and click **Build index** in the sidebar to get started.")
else:
    question = st.text_input("Your question")
    if st.button("Ask") and question:
        with st.spinner("Retrieving context and generating answer..."):
            q_emb = embedder.encode([question]).astype("float32")
            faiss.normalize_L2(q_emb)
            scores, idxs = st.session_state.index.search(q_emb, TOP_K)
            retrieved = [st.session_state.chunks[i] for i in idxs[0] if i != -1]
            context = "\n\n".join(c["text"] for c in retrieved)
            sources = sorted({c["source"] for c in retrieved})

            prompt = (
                "Answer the question using ONLY the context below. "
                "If the answer is not contained in the context, say you "
                "don't have enough information rather than guessing.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            answer = generator(prompt)[0]["generated_text"]

            st.session_state.history.insert(0, {
                "question": question, "answer": answer, "sources": sources,
                "scores": [float(s) for s in scores[0]],
            })

    for turn in st.session_state.history:
        st.markdown(f"**Q: {turn['question']}**")
        st.write(turn["answer"])
        st.caption(f"Sources: {', '.join(turn['sources'])}")
        st.divider()

st.caption(
    "Built by Ankit Singh · [GitHub repo](https://github.com/ankitgitwork/RAG-Based-AI-Chatbot-for-Document-Q-A)"
)
