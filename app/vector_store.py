import chromadb
from app.config import CHROMA_DB_PATH

_client = None
_collection = None
COLLECTION_NAME = "gov_portal_docs"


def get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return _client


def get_collection():
    global _collection
    if _collection is None:
        client = get_client()
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_documents(ids, embeddings, documents, metadatas):
    get_collection().add(
        ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
    )


def query(query_embedding: list[float], n_results: int = 5) -> list[dict]:
    col = get_collection()
    count = col.count()
    if count == 0:
        return []
    n = min(n_results, count)
    results = col.query(
        query_embeddings=[query_embedding],
        n_results=n,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]


def collection_count() -> int:
    return get_collection().count()


def clear_collection():
    global _collection
    get_client().delete_collection(COLLECTION_NAME)
    _collection = None
