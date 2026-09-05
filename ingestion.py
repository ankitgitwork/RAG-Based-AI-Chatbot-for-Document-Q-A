"""
Ingestion pipeline: loads raw documents, splits them into overlapping
chunks, embeds them, and writes them into a persistent Chroma vector store.
"""
import re
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

from app.config import settings

LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".docx": Docx2txtLoader,
}

_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def load_documents(source_dir: str) -> List[Document]:
    """Load every supported file in source_dir into LangChain Documents."""
    docs: List[Document] = []
    for path in Path(source_dir).glob("**/*"):
        if path.suffix.lower() not in settings.allowed_extensions:
            continue
        loader_cls = LOADER_MAP.get(path.suffix.lower())
        if loader_cls is None:
            continue
        if path.suffix.lower() == ".pdf":
            # "layout" mode reconstructs spacing from each glyph's actual
            # coordinates instead of the raw content-stream order. The
            # default "plain" mode concatenates text runs with zero spacing
            # whenever two elements (e.g. a name and a location field placed
            # side by side in a resume header) don't contain an explicit
            # space character between them — that's what turns "ANKIT SINGH"
            # into "ANKITSINGH". Layout mode fixes that at the source instead
            # of guessing at word boundaries after the fact.
            loader = loader_cls(str(path), extraction_mode="layout")
        else:
            loader = loader_cls(str(path))
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = path.name
            if path.suffix.lower() == ".pdf":
                # Layout mode pads with runs of spaces to preserve columns;
                # collapse those back down for embedding/prompting.
                d.page_content = _MULTI_SPACE.sub(" ", d.page_content)
        docs.extend(loaded)
    return docs


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks sized for retrieval quality."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vector_store(source_dir: str = None) -> Chroma:
    """Load, chunk, embed, and persist documents into the vector store."""
    source_dir = source_dir or settings.upload_dir
    raw_docs = load_documents(source_dir)
    if not raw_docs:
        raise ValueError(f"No supported documents found in {source_dir}")

    chunks = chunk_documents(raw_docs)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.vector_db_path,
    )
    return vector_store


def get_existing_vector_store() -> Chroma:
    """Reconnect to an already-populated vector store without re-embedding."""
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=settings.vector_db_path,
    )


if __name__ == "__main__":
    store = build_vector_store()
    print(f"Ingested and persisted vector store at {settings.vector_db_path}")
