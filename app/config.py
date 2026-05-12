import os, json
from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

SUPPORT_EMAIL = os.getenv("SUPPORT_EMAIL", "support@portal.gov")
SUPPORT_PHONE = os.getenv("SUPPORT_PHONE", "1800-XXX-XXXX")
SUPPORT_URL = os.getenv("SUPPORT_URL", "")

SEARCH_API_URL = os.getenv("SEARCH_API_URL", "https://api.example.gov/search")
CDO_DETAILS_API_URL = os.getenv("CDO_DETAILS_API_URL", "https://api.example.gov/cdo/details")
DATASET_CDO_API_URL = os.getenv("DATASET_CDO_API_URL", "https://api.example.gov/dataset/cdo")
FEEDBACK_API_URL = os.getenv("FEEDBACK_API_URL", "https://api.example.gov/feedback")
CONTACT_CDO_API_URL = os.getenv("CONTACT_CDO_API_URL", "https://api.example.gov/contact-cdo")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
PDF_FOLDER = os.getenv("PDF_FOLDER", "./pdfs")

RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.45"))
QA_MATCH_THRESHOLD = float(os.getenv("QA_MATCH_THRESHOLD", "0.75"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

LLM_MODERATION_ENABLED = os.getenv("LLM_MODERATION_ENABLED", "true").lower() == "true"
SCOPE_THRESHOLD = float(os.getenv("SCOPE_THRESHOLD", "0.25"))

DEFAULT_SCOPE_TOPICS = [
    "Greeting, small talk, or asking what this assistant can do",
    "Government Open Data Portal — what it is, how it works, accessing it",
    "Datasets on the portal — searching, downloading, browsing, file formats",
    "Chief Data Officers (CDOs) — contact details, ministries, roles, responsibilities",
    "Feedback, complaints, or reporting issues about the portal or a dataset",
    "Topics described in the portal documentation PDFs",
]


def _load_scope_topics() -> list[str]:
    raw = os.getenv("SCOPE_TOPICS")
    if not raw:
        return DEFAULT_SCOPE_TOPICS
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed) and parsed:
            return parsed
    except json.JSONDecodeError:
        print("Warning: SCOPE_TOPICS is not valid JSON, falling back to defaults.")
    return DEFAULT_SCOPE_TOPICS


SCOPE_TOPICS = _load_scope_topics()


def load_predefined_qa() -> list[dict]:
    """Load all PREDEFINED_QA_N env vars and return as list of {q, a} dicts."""
    qa_pairs = []
    i = 1
    while True:
        raw = os.getenv(f"PREDEFINED_QA_{i}")
        if raw is None:
            break
        try:
            pair = json.loads(raw)
            if "q" in pair and "a" in pair:
                qa_pairs.append(pair)
        except json.JSONDecodeError:
            print(f"Warning: PREDEFINED_QA_{i} is not valid JSON, skipping.")
        i += 1
    return qa_pairs
