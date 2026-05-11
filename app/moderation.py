from app.config import RAG_RELEVANCE_THRESHOLD

POLITICAL_KEYWORDS = [
    "election", "vote", "party", "parliament", "bjp", "congress", "aap", "modi",
    "opposition", "government policy", "protest", "strike", "riot", "religion",
    "caste", "reservation", "communal", "terrorism", "war", "military", "army",
    "nuclear", "sanctions", "judiciary", "court verdict", "supreme court ruling",
    "political", "politician",
]


def is_politically_sensitive(message: str) -> bool:
    msg_lower = message.lower()
    return any(kw in msg_lower for kw in POLITICAL_KEYWORDS)


def has_relevant_context(rag_chunks: list[dict]) -> bool:
    return any((1.0 - chunk["distance"]) >= RAG_RELEVANCE_THRESHOLD for chunk in rag_chunks)


def filter_relevant_chunks(rag_chunks: list[dict]) -> list[dict]:
    return [c for c in rag_chunks if (1.0 - c["distance"]) >= RAG_RELEVANCE_THRESHOLD]
