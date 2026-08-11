"""
FastAPI backend for the RAG document Q&A chatbot.

Endpoints:
  POST /ingest        - upload documents and (re)build the vector store
  POST /chat           - ask a question, grounded in ingested documents
  POST /chat/reset      - clear conversation memory
  GET  /health          - liveness check
"""
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import settings
from app.ingestion import build_vector_store

app = FastAPI(title="RAG Document Q&A Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_chatbot = None  # lazily initialized after first ingest


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        suffix = Path(f.filename).suffix.lower()
        if suffix not in settings.allowed_extensions:
            raise HTTPException(400, f"Unsupported file type: {f.filename}")
        dest = upload_dir / f.filename
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(f.filename)

    build_vector_store(str(upload_dir))

    global _chatbot
    _chatbot = None  # force re-init with fresh store on next /chat call

    return {"ingested_files": saved}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    global _chatbot
    if _chatbot is None:
        from app.rag_chain import RAGChatbot
        try:
            _chatbot = RAGChatbot()
        except Exception as e:
            raise HTTPException(400, f"No documents ingested yet: {e}")

    answer, sources = _chatbot.ask(req.question)
    return ChatResponse(answer=answer, sources=sources)


@app.post("/chat/reset")
def reset_chat():
    global _chatbot
    if _chatbot is not None:
        _chatbot.reset_memory()
    return {"status": "memory cleared"}
