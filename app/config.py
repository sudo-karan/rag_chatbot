import os, json
from dotenv import load_dotenv

load_dotenv()

from app.profile import ACTIVE_PROFILE, PROFILE_INFO

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Per-role models. Env wins, profile is the spec-aware fallback.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL") or ACTIVE_PROFILE.main_model
OLLAMA_HELPER_MODEL = os.getenv("OLLAMA_HELPER_MODEL") or ACTIVE_PROFILE.helper_model

# Ollama runtime tuning, also profile-driven with env override.
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE") or ACTIVE_PROFILE.keep_alive
OLLAMA_NUM_THREAD = int(os.getenv("OLLAMA_NUM_THREAD", ACTIVE_PROFILE.num_thread))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", ACTIVE_PROFILE.num_ctx))
# Helper-model context window. The moderation / intent prompts are long
# (~1.5k tokens plus the recent-context block), so we set an explicit window
# well above Ollama's 2048 default to avoid silent front-truncation of the
# instructions. Cheap for the small helper model on every profile.
OLLAMA_HELPER_NUM_CTX = int(os.getenv("OLLAMA_HELPER_NUM_CTX", "4096"))
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", ACTIVE_PROFILE.embed_batch_size))

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

DATASETS_FILE = os.getenv("DATASETS_FILE", "./datasets.csv")
FEEDBACK_LOG_FILE = os.getenv("FEEDBACK_LOG_FILE", "./feedback_log.jsonl")

RAG_RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.45"))
QA_MATCH_THRESHOLD = float(os.getenv("QA_MATCH_THRESHOLD", "0.85"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))

LLM_MODERATION_ENABLED = os.getenv("LLM_MODERATION_ENABLED", "true").lower() == "true"
# ENABLE_OUTPUT_VERIFICATION default comes from the profile; env wins.
_verification_env = os.getenv("ENABLE_OUTPUT_VERIFICATION")
ENABLE_OUTPUT_VERIFICATION = (
    _verification_env.lower() == "true" if _verification_env is not None
    else ACTIVE_PROFILE.enable_verification
)
SCOPE_THRESHOLD = float(os.getenv("SCOPE_THRESHOLD", "0.45"))

DEFAULT_SCOPE_TOPICS = [
    "hi",
    "hello",
    "namaste",
    "what can you do",
    "what can you help me with",
    "help",
    "what is data.gov.in",
    "what is the OGD Platform India",
    "who runs data.gov.in",
    "what is NDSAP",
    "what is the scope of NDSAP",
    "guiding principles of NDSAP",
    "how do I search for a dataset",
    "how do I download a dataset",
    "what formats are datasets available in",
    "is data.gov.in free",
    "how do I get an API key",
    "how do I register on data.gov.in",
    "what services do registered users get",
    "what is a catalog vs a resource",
    "what is a Chief Data Officer",
    "responsibilities of a CDO",
    "who can be nominated as CDO",
    "what is a Data Contributor",
    "who is the CDO of Ministry of Agriculture",
    "find CDO for my ministry",
    "what is an NDSAP Cell",
    "what is the Negative List",
    "what is a High-Value Dataset",
    "how do I contribute a dataset",
    "how do I suggest a new dataset",
    "how do I give feedback on a dataset",
    "how to contact the NDSAP PMU",
    "what is the Government Open Data Licence India",
    "what are the terms of use",
    "what is the privacy policy",
    "accessibility features of data.gov.in",
    "screen readers supported by the portal",
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

# Curated, human-readable summary of the in-scope themes, shown in the
# out-of-scope redirect hint. Kept as a clean prose list (not derived from raw
# chunk text) so the suggestion reads well and is deterministic. Override via
# env if your corpus covers different ground.
KNOWN_TOPICS_SUMMARY = os.getenv(
    "KNOWN_TOPICS_SUMMARY",
    "data.gov.in and the Open Government Data Platform, the NDSAP policy, "
    "Chief Data Officers and Nodal Officers, searching and downloading datasets, "
    "data formats and APIs, contributing or suggesting datasets, accessibility "
    "features, and the portal's terms of use",
)


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
