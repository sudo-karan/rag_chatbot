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


SENSITIVE_PROMPT = """Decide whether the user message is about any of these sensitive topics.

Sensitive topics (answer "yes"):
- Partisan politics: elections, political parties, politicians, political ideology, government-policy debates
- Geopolitics: foreign relations, international conflicts between countries, territorial disputes, wars
- Religion, religious groups, communal matters, sectarian issues
- Caste, reservation policy, ethnic groups
- Military, armed forces, defense operations, security forces
- Legal advice, court cases, judicial rulings, ongoing litigation
- Personal medical, financial, or legal advice
- Violence, terrorism, extremism, illegal activities

NOT sensitive (answer "no"):
- Sports of any kind, including international tournaments and world cups
- Science, geography, technology, mathematics
- Factual history (events, dates, biographies) — but NOT ongoing political controversies
- Entertainment: movies, books, music, games, food, travel
- General knowledge, trivia, definitions
- Government data portals, datasets, public administration logistics
- Greetings, small talk, asking what the assistant can do

Examples:
Message: who won the cricket world cup
Answer: no
Message: how is FIFA organised
Answer: no
Message: olympic games 2024 medal table
Answer: no
Message: tell me about the south china sea dispute
Answer: yes
Message: what is photosynthesis
Answer: no
Message: BJP election results
Answer: yes
Message: recommend a movie
Answer: no
Message: kashmir issue
Answer: yes
Message: nutritional value of oranges
Answer: no
Message: how do I download a dataset
Answer: no
Message: what is the army's role in disaster relief
Answer: yes
Message: history of the roman empire
Answer: no

Reply with ONLY "yes" or "no". No explanation. No punctuation.

Message: {message}
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


def prototype_scope_score(query_emb: list[float]) -> float:
    """Max cosine similarity between the query embedding and any SCOPE_TOPICS example."""
    if not SCOPE_TOPICS:
        return 1.0
    from app.embedder import cosine_similarity
    _ensure_scope_embeddings()
    best = -1.0
    for topic_emb in _scope_embeddings:
        score = cosine_similarity(query_emb, topic_emb)
        if score > best:
            best = score
    return best


def prototype_scope_pass(query_emb: list[float]) -> bool:
    return prototype_scope_score(query_emb) >= SCOPE_THRESHOLD


def has_relevant_context(rag_chunks: list[dict]) -> bool:
    return any((1.0 - chunk["distance"]) >= RAG_RELEVANCE_THRESHOLD for chunk in rag_chunks)


def filter_relevant_chunks(rag_chunks: list[dict]) -> list[dict]:
    return [c for c in rag_chunks if (1.0 - c["distance"]) >= RAG_RELEVANCE_THRESHOLD]
