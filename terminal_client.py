#!/usr/bin/env python3
import os
import sys

from dotenv import load_dotenv

# Load .env before any profile detection so PROFILE_OVERRIDE / VRAM thresholds
# set there are visible to the interactive picker below.
load_dotenv()


def _select_profile_interactively() -> None:
    """Show detected specs + the recommended profile and let the operator accept
    or override it BEFORE the app modules (which freeze config from the profile)
    are imported. Sets PROFILE_OVERRIDE in the environment when a choice is made.

    No-op (keeps the recommendation) when PROFILE_OVERRIDE is already set, when
    stdin is not a TTY (piped / CI), or on empty input / EOF. Only app.profile is
    imported here — it has no import-time side effects and does not pull in config."""
    from app.profile import detect_profile, format_profile_banner, PROFILES

    _, info = detect_profile()
    print(format_profile_banner(info))

    if os.getenv("PROFILE_OVERRIDE"):
        return  # already pinned by env — respect it
    if not sys.stdin.isatty():
        return  # non-interactive — accept the recommendation

    choices = "|".join(PROFILES.keys())
    try:
        choice = input(
            f"Profile [{info['recommended']}] — Enter to accept, or type {choices}: "
        ).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return

    if not choice:
        return
    if choice in PROFILES:
        os.environ["PROFILE_OVERRIDE"] = choice
        print(f"Using profile: {choice}")
    else:
        print(f"Unknown profile '{choice}'; using recommended '{info['recommended']}'.")


def run_terminal():
    _select_profile_interactively()

    # Deferred imports: app.config resolves the profile on first import, so the
    # interactive choice above must already be applied to the environment.
    from app.ingestion import ingest_pdfs
    from app.session import create_session
    from app.chat import process_message_stream, get_disclaimer
    from app.llm import is_ollama_available
    from app.health import run_startup_checks

    if not is_ollama_available():
        print("ERROR: Ollama is not running or the model is not available.")
        print("Please run: ollama serve")
        print("Then pull the model: ollama pull llama3.1:8b")
        sys.exit(1)

    print("\nInitialising chatbot - loading PDFs into vector store...")
    ingest_pdfs()
    run_startup_checks()
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

        print("\nBot: ", end="", flush=True)
        for chunk in process_message_stream(session_id, user_input):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    run_terminal()
