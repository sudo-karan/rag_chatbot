#!/usr/bin/env python3
import sys
from app.ingestion import ingest_pdfs
from app.session import create_session
from app.chat import process_message, get_disclaimer
from app.llm import is_ollama_available


def run_terminal():
    if not is_ollama_available():
        print("ERROR: Ollama is not running or the model is not available.")
        print("Please run: ollama serve")
        print("Then pull the model: ollama pull llama3.1:8b")
        sys.exit(1)

    print("\nInitialising chatbot - loading PDFs into vector store...")
    ingest_pdfs()
    print("Done.\n")

    session = create_session()
    session_id = session.session_id

    print(get_disclaimer())
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye"):
            print("Bot: Thank you for using the Government Open Data Portal assistant. Goodbye!")
            sys.exit(0)

        response = process_message(session_id, user_input)
        print(f"\nBot: {response}\n")


if __name__ == "__main__":
    run_terminal()
