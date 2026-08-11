"""
Simple terminal chat interface for the RAG chatbot.

Usage:
    python -m app.ingestion            # build the vector store once
    python cli.py                      # then chat from the terminal
"""
from app.rag_chain import RAGChatbot


def main():
    print("RAG Document Q&A Chatbot (type 'exit' to quit, 'reset' to clear memory)\n")
    bot = RAGChatbot()

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        if question.lower() == "reset":
            bot.reset_memory()
            print("(conversation memory cleared)\n")
            continue

        answer, sources = bot.ask(question)
        print(f"\nBot: {answer}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()


if __name__ == "__main__":
    main()
