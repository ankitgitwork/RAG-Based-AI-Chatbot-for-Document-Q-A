"""
Central configuration for the RAG chatbot.
All tunable retrieval/generation parameters live here so they can be
adjusted without touching pipeline code.
"""
import os
from dataclasses import dataclass


@dataclass
class Settings:
    # --- OpenAI ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # --- Vector store ---
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./data/chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")

    # --- Chunking ---
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # --- Retrieval ---
    top_k: int = int(os.getenv("TOP_K", "4"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))

    # --- Conversation memory ---
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "6"))

    # --- Upload limits ---
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/sample_docs")
    allowed_extensions: tuple = (".pdf", ".txt", ".md", ".docx")


settings = Settings()
