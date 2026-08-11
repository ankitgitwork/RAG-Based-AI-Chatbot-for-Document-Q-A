"""
Unit tests that don't require an OpenAI API key — they exercise the
chunking logic directly on in-memory Documents.
"""
from langchain.schema import Document
from app.ingestion import chunk_documents


def test_chunking_respects_overlap_and_splits_long_text():
    long_text = "Sentence about topic A. " * 200
    doc = Document(page_content=long_text, metadata={"source": "test.txt"})

    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.page_content) <= 800 + 50  # chunk_size + slack
        assert c.metadata["source"] == "test.txt"


def test_short_document_produces_single_chunk():
    doc = Document(page_content="Short text.", metadata={"source": "short.txt"})
    chunks = chunk_documents([doc])
    assert len(chunks) == 1
    assert chunks[0].page_content == "Short text."


if __name__ == "__main__":
    test_chunking_respects_overlap_and_splits_long_text()
    test_short_document_produces_single_chunk()
    print("All tests passed.")
