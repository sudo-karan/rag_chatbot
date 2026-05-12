from app.config import RAG_RELEVANCE_THRESHOLD, SCOPE_TOPICS, SCOPE_THRESHOLD

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


SENSITIVE_PROMPT = """Classify whether the following user message is about any of these sensitive topics:
- Politics, elections, political parties, politicians, political ideology
- Geopolitics, foreign relations, international conflicts, territorial disputes, wars
- Religion, religious groups, communal matters, sectarian issues
- Caste, reservation policy, ethnic groups
- Military, armed forces, defense operations, security forces
- Legal advice, court cases, judicial rulings, ongoing litigation
- Personal medical, financial, or legal advice
- Violence, terrorism, extremism, illegal activities
- Public figures' personal lives or controversies

Reply with ONLY "yes" or "no". No explanation. No punctuation.

User message: {message}

Answer:"""


def is_sensitive_llm(message: str) -> bool:
    """LLM-based sensitivity check. Returns False on any error (safe-open: other gates catch it)."""
    from app.llm import complete
    try:
        raw = complete(SENSITIVE_PROMPT.format(message=message), max_tokens=4, temperature=0.0)
        return raw.strip().lower().startswith("yes")
    except Exception as e:
        print(f"LLM moderation error: {e}")
        return False


_scope_embeddings: list[list[float]] | None = None


def _ensure_scope_embeddings():
    global _scope_embeddings
    if _scope_embeddings is None:
        from app.embedder import embed
        _scope_embeddings = embed(SCOPE_TOPICS) if SCOPE_TOPICS else []


def is_in_scope(message: str) -> tuple[bool, float, int]:
    """
    Semantic check against SCOPE_TOPICS. Returns (in_scope, best_score, best_index).
    in_scope = best_score >= SCOPE_THRESHOLD.
    """
    if not SCOPE_TOPICS:
        return True, 1.0, -1
    from app.embedder import embed_one, cosine_similarity
    _ensure_scope_embeddings()
    q = embed_one(message)
    best_score, best_idx = -1.0, -1
    for i, topic_emb in enumerate(_scope_embeddings):
        score = cosine_similarity(q, topic_emb)
        if score > best_score:
            best_score, best_idx = score, i
    return best_score >= SCOPE_THRESHOLD, best_score, best_idx


def has_relevant_context(rag_chunks: list[dict]) -> bool:
    return any((1.0 - chunk["distance"]) >= RAG_RELEVANCE_THRESHOLD for chunk in rag_chunks)


def filter_relevant_chunks(rag_chunks: list[dict]) -> list[dict]:
    return [c for c in rag_chunks if (1.0 - c["distance"]) >= RAG_RELEVANCE_THRESHOLD]
