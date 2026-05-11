"""
Ollama wrapper — all LLM calls go through here.
Ollama must be running: `ollama serve`
Model must be pulled: `ollama pull llama3.1:8b`
"""
import ollama
from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL


def _get_client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_BASE_URL)


def chat(
    system: str,
    messages: list[dict],
    max_tokens: int = 1024,
    temperature: float = 0.1,
) -> str:
    """
    Multi-turn chat with system prompt.
    messages: list of {"role": "user"|"assistant", "content": "..."}
    Returns the assistant reply as a string.
    Low temperature (0.1) for factual, consistent government responses.
    """
    client = _get_client()
    ollama_messages = [{"role": "system", "content": system}] + messages
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=ollama_messages,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )
    return response["message"]["content"].strip()


def complete(
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.0,
) -> str:
    """
    Single-shot completion. Used for intent classification.
    Temperature 0.0 for deterministic output.
    """
    client = _get_client()
    response = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    )
    return response["response"].strip()


def is_ollama_available() -> bool:
    """Check if Ollama server is running and the configured model is available."""
    try:
        client = _get_client()
        models = client.list()
        available = [m["name"] for m in models.get("models", [])]
        model_base = OLLAMA_MODEL.split(":")[0]
        return any(model_base in m for m in available)
    except Exception:
        return False
