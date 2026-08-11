"""
Retrieval-Augmented Generation chain.

Wraps the vector store retriever + chat model into a conversational
chain that grounds every answer in retrieved document context and
keeps a bounded window of chat history for multi-turn coherence.
"""
from typing import List, Tuple

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from app.config import settings
from app.ingestion import get_existing_vector_store

SYSTEM_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a document Q&A assistant. Answer the question using ONLY "
        "the context below. If the answer is not contained in the context, "
        "say you don't have enough information rather than guessing.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


class RAGChatbot:
    def __init__(self):
        self.vector_store = get_existing_vector_store()

        self.retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": settings.top_k,
                "score_threshold": settings.similarity_threshold,
            },
        )

        self.llm = ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
        )

        self.memory = ConversationBufferWindowMemory(
            k=settings.max_history_turns,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": SYSTEM_PROMPT},
        )

    def ask(self, question: str) -> Tuple[str, List[str]]:
        """Return (answer, list of source filenames used)."""
        result = self.chain.invoke({"question": question})
        sources = sorted({
            doc.metadata.get("source", "unknown")
            for doc in result.get("source_documents", [])
        })
        return result["answer"], sources

    def reset_memory(self):
        self.memory.clear()
