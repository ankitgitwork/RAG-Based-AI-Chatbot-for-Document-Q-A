# RAG-Based AI Chatbot for Document Q&A

A Retrieval-Augmented Generation (RAG) pipeline that indexes documents into a
vector database and grounds LLM responses in retrieved context — reducing
hallucinated answers compared to a baseline LLM-only chatbot. Built with
LangChain, OpenAI, Chroma, and FastAPI.

## Why RAG

A plain LLM chatbot answers from parametric memory alone, so it can
confidently invent facts about documents it has never seen. This project
instead retrieves the most relevant chunks of your own documents at query
time and forces the model to answer from that context — so answers are
traceable back to a source file.

## Architecture

```
                 ┌─────────────┐
   documents ──▶ │  Ingestion  │  load → chunk → embed
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  Chroma DB  │  persistent vector store
                 └──────┬──────┘
                        ▼
   question ──▶ ┌─────────────┐   top-k similarity search
                 │  Retriever  │   (score-thresholded)
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐
                 │  Chat LLM   │  answers ONLY from retrieved context
                 │ + memory    │  (windowed multi-turn history)
                 └──────┬──────┘
                        ▼
                     answer + sources
```

- **Chunking**: `RecursiveCharacterTextSplitter` with configurable
  `chunk_size` / `chunk_overlap` (defaults: 800 / 120 characters) so context
  windows stay coherent without cutting sentences awkwardly.
- **Embeddings**: OpenAI `text-embedding-3-small`, persisted in a local
  Chroma collection — no external vector DB service required.
- **Retrieval**: similarity-score-threshold search (`top_k=4`,
  `similarity_threshold=0.25` by default) so irrelevant chunks are dropped
  rather than padding the prompt.
- **Conversation memory**: a windowed buffer (last 6 turns by default) keeps
  multi-turn follow-up questions coherent without unbounded context growth.
- **Serving**: a FastAPI backend exposes `/ingest` and `/chat`, plus a
  terminal CLI for local testing without standing up a frontend.

## Project structure

```
rag-chatbot/
├── app/
│   ├── config.py       # all tunable parameters in one place
│   ├── ingestion.py    # load → chunk → embed → persist
│   ├── rag_chain.py    # retriever + LLM + memory
│   └── main.py         # FastAPI app (/ingest, /chat, /health)
├── cli.py              # terminal chat client
├── tests/
│   └── test_ingestion.py
├── data/sample_docs/    # drop source PDFs/txt/md/docx here
├── requirements.txt
└── .env.example
```

## Setup

```bash
git clone https://github.com/ankitgitwork/rag-chatbot.git
cd rag-chatbot
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env   # then add your OPENAI_API_KEY
```

## Usage

**1. Add documents** to `data/sample_docs/` (PDF, TXT, MD, or DOCX).

**2. Build the vector store:**
```bash
python -m app.ingestion
```

**3a. Chat from the terminal:**
```bash
python cli.py
```

**3b. Or run the API:**
```bash
uvicorn app.main:app --reload
```
Then:
```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@data/sample_docs/your_doc.pdf"

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What does the document say about X?"}'
```

## Tuning retrieval quality

All of these are environment variables (see `.env.example`) so they can be
tuned per document set without touching code:

| Parameter               | Default | Effect                                      |
|--------------------------|---------|----------------------------------------------|
| `CHUNK_SIZE`             | 800     | Larger = more context per chunk, less precise |
| `CHUNK_OVERLAP`          | 120     | Higher = less chance of splitting an idea mid-chunk |
| `TOP_K`                  | 4       | How many chunks are retrieved per question    |
| `SIMILARITY_THRESHOLD`   | 0.25    | Higher = stricter relevance filtering         |
| `MAX_HISTORY_TURNS`      | 6       | How much prior conversation is kept in context |

## Running tests

```bash
pytest tests/
```
(`test_ingestion.py` covers chunking logic and needs no API key.)

## Tech stack

Python · LangChain · OpenAI API · Chroma (Vector DB) · FastAPI · pytest

## License

MIT
