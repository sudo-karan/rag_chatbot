"""Local dataset directory backing the mocked API intents.

Loaded once at startup from a CSV (or JSON) on disk; powers the chatbot's
search, URL→contributor lookup, and contributor lookup for the intents
`search`, `cdo_details`, `dataset_cdo_link`, and `contact_cdo`.

Required CSV columns: title, url, contributor_name, contributor_email.
Optional columns: ministry, sector, last_updated. Unknown columns are
silently ignored, so the schema can grow without code changes.
"""
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import DATASETS_FILE


@dataclass
class DatasetRecord:
    title: str
    url: str
    contributor_name: str
    contributor_email: str
    ministry: str = ""
    sector: str = ""
    last_updated: str = ""


_records: list[DatasetRecord] | None = None
_url_index: dict[str, DatasetRecord] | None = None
_title_embeddings: list[list[float]] | None = None


def _load() -> list[DatasetRecord]:
    path = Path(DATASETS_FILE)
    if not path.exists():
        print(
            f"WARNING: Dataset directory not found at {path}. The three "
            "read-backed API intents (search / cdo_details / dataset_cdo_link) "
            "will return a graceful 'not configured' response. The feedback "
            "intents (portal_feedback / contact_cdo) are unaffected — they log "
            "to FEEDBACK_LOG_FILE regardless. Drop a CSV at this path or set "
            "DATASETS_FILE in .env."
        )
        return []
    try:
        if path.suffix.lower() == ".json":
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            return [_record_from_row(r) for r in raw if r.get("url")]
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [_record_from_row(row) for row in reader if row.get("url")]
    except Exception as e:
        print(f"WARNING: Failed to load dataset directory {path}: {e}")
        return []


def _record_from_row(row: dict) -> DatasetRecord:
    return DatasetRecord(
        title=(row.get("title") or "").strip(),
        url=(row.get("url") or "").strip(),
        contributor_name=(row.get("contributor_name") or "").strip(),
        contributor_email=(row.get("contributor_email") or "").strip(),
        ministry=(row.get("ministry") or "").strip(),
        sector=(row.get("sector") or "").strip(),
        last_updated=(row.get("last_updated") or "").strip(),
    )


def _ensure_loaded():
    global _records, _url_index
    if _records is None:
        _records = _load()
        _url_index = {r.url: r for r in _records if r.url}


def all_records() -> list[DatasetRecord]:
    _ensure_loaded()
    return _records or []


def is_configured() -> bool:
    return bool(all_records())


def get_by_url(url: str) -> DatasetRecord | None:
    """Exact-match first, then substring fallback (handles trailing slashes,
    UTM tags, the user pasting a resource URL vs a catalog URL, etc.)."""
    _ensure_loaded()
    if not _records:
        return None
    url = url.strip()
    if url in _url_index:
        return _url_index[url]
    for r in _records:
        if r.url and (url in r.url or r.url in url):
            return r
    return None


def get_by_contributor(query: str, n: int = 10) -> list[DatasetRecord]:
    """Case-insensitive substring match on name, email, or ministry."""
    _ensure_loaded()
    if not _records:
        return []
    q = query.strip().lower()
    if not q:
        return []
    return [
        r for r in _records
        if q in r.contributor_name.lower()
        or q in r.contributor_email.lower()
        or q in r.ministry.lower()
    ][:n]


def search(keyword: str, n: int = 5) -> list[DatasetRecord]:
    """Semantic ranking of dataset titles against the query.

    Title embeddings are computed once at first call and cached. Per-query
    cost is one embed + a dot product per record — trivially fast up to
    a few thousand records."""
    global _title_embeddings
    _ensure_loaded()
    if not _records:
        return []
    keyword = keyword.strip()
    if not keyword:
        return []

    if _title_embeddings is None:
        from app.embedder import embed
        _title_embeddings = embed([r.title for r in _records])

    from app.embedder import embed_one, cosine_similarity
    q = embed_one(keyword)
    scored = [
        (cosine_similarity(q, emb), rec)
        for emb, rec in zip(_title_embeddings, _records)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:n]]
