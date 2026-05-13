"""
Ollama wrapper — every LLM call goes through here.

Two role-based entry points:
- chat()    : multi-turn conversation, uses the MAIN model from the active profile.
- helper()  : single-shot prompt, uses the HELPER model (small/fast) for
              moderation, intent classification, and grounding verification.

Both honour the profile's keep_alive, num_thread, and num_ctx, unless the
caller overrides per-call.
"""
import ollama
from app.config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_HELPER_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_NUM_THREAD,
    OLLAMA_NUM_CTX,
)


def _get_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_BASE_URL)


def _options(temperature: float, max_tokens: int, num_ctx: int | None = None) -> dict:
    opts: dict = {"temperature": temperature, "num_predict": max_tokens}
    if OLLAMA_NUM_THREAD > 0:
        opts["num_thread"] = OLLAMA_NUM_THREAD
    if num_ctx is not None and num_ctx > 0:
        opts["num_ctx"] = num_ctx
    return opts


def chat(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.1,
    model: str | None = None,
    keep_alive: str | None = None,
) -> str:
    """Multi-turn chat against the MAIN model (override with `model=`)."""
    client = _get_client()
    response = client.chat(
        model=model or OLLAMA_MODEL,
        messages=[{"role": "system", "content": system}] + messages,
        options=_options(temperature, max_tokens, num_ctx=OLLAMA_NUM_CTX),
        keep_alive=keep_alive or OLLAMA_KEEP_ALIVE,
    )
    return response["message"]["content"].strip()


def complete(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    model: str | None = None,
    keep_alive: str | None = None,
) -> str:
    """Generic single-shot completion (defaults to the MAIN model)."""
    client = _get_client()
    response = client.generate(
        model=model or OLLAMA_MODEL,
        prompt=prompt,
        options=_options(temperature, max_tokens),
        keep_alive=keep_alive or OLLAMA_KEEP_ALIVE,
    )
    return response["response"].strip()


def helper(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
    keep_alive: str | None = None,
) -> str:
    """Single-shot via the HELPER model (cheap, fast). Used by moderator, intent
    classifier, and grounding verifier."""
    return complete(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        model=OLLAMA_HELPER_MODEL,
        keep_alive=keep_alive,
    )


def is_ollama_available() -> bool:
    """True iff Ollama is reachable AND every configured model is present."""
    try:
        client = _get_client()
        listing = client.list()
        available = {m["name"] for m in listing.get("models", [])}
        # Match by base name to be lenient about :tag variations.
        wanted_bases = {OLLAMA_MODEL.split(":")[0], OLLAMA_HELPER_MODEL.split(":")[0]}
        present_bases = {name.split(":")[0] for name in available}
        return wanted_bases.issubset(present_bases)
    except Exception:
        return False
