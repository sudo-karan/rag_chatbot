from app.config import load_predefined_qa, QA_MATCH_THRESHOLD
from app.embedder import embed, embed_one, cosine_similarity

_qa_pairs = None
_qa_embeddings = None


def _ensure_loaded():
    global _qa_pairs, _qa_embeddings
    if _qa_pairs is None:
        _qa_pairs = load_predefined_qa()
        _qa_embeddings = embed([p["q"] for p in _qa_pairs]) if _qa_pairs else []


def find_best_predefined_answer(query: str) -> str | None:
    _ensure_loaded()
    if not _qa_pairs:
        return None
    query_emb = embed_one(query)
    best_score, best_answer = -1.0, None
    for pair, qa_emb in zip(_qa_pairs, _qa_embeddings):
        score = cosine_similarity(query_emb, qa_emb)
        if score > best_score:
            best_score, best_answer = score, pair["a"]
    return best_answer if best_score >= QA_MATCH_THRESHOLD else None
