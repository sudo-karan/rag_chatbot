# RAG Government Chatbot (Fully Offline / Local)

A production-ready, fully offline RAG-based chatbot for a Government Open Data Portal. All LLM inference runs locally via Ollama, embeddings are computed locally with `sentence-transformers`, and the vector store (ChromaDB) is persisted to disk. There are **zero** external API calls at runtime.

---

## 1. Overview

The chatbot:

- Answers questions strictly from the PDFs you place in `pdfs/` and from predefined Q&A pairs in `.env`.
- Refuses any out-of-scope or politically sensitive question with a clear explanation and support contact.
- Requires disclaimer acceptance before every chat session.
- Routes user messages to one of 6 intents: `search`, `cdo_details`, `dataset_cdo_link`, `portal_feedback`, `contact_cdo`, or `rag_chat` (default).
- Maintains per-session conversation history (in-memory).
- Exposes both a FastAPI REST API (multi-user, session-based) and a terminal client.

---

## 2. Architecture

| Layer            | Component                                |
|------------------|------------------------------------------|
| LLM inference    | Ollama (local) — default `llama3.1:8b`   |
| Embeddings       | `sentence-transformers` `all-MiniLM-L6-v2` |
| Vector store     | ChromaDB (persistent, on disk)           |
| PDF parsing      | `pdfplumber`                             |
| API framework    | FastAPI + uvicorn                        |
| Session store    | In-memory Python dict                    |
| Intent routing   | Ollama single-shot JSON classification   |

Request flow for a chat message:

1. Disclaimer gate — refuses to process until the user accepts.
2. Intent classification via Ollama.
3. If intent is an API intent (search, CDO, feedback, etc.) — call the mocked API.
4. Otherwise (`rag_chat`):
   - Check predefined Q&A (semantic similarity ≥ `QA_MATCH_THRESHOLD`).
   - Otherwise embed the query, retrieve top-K chunks from ChromaDB.
   - Filter chunks by relevance (`RAG_RELEVANCE_THRESHOLD`).
   - If nothing relevant → refuse with redirect hint and support contact.
   - Else build a strict system prompt with the context and call Ollama.

---

## 3. Prerequisites

- Python **3.13.5** (3.11+ also works).
- [Ollama](https://ollama.com/download) installed and running locally.
- ~8 GB free RAM (for `llama3.1:8b`) or ~4 GB for the lighter `llama3.2:3b`.

---

## 4. Setup

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env if needed (model, support contact, API URLs, predefined Q&A)

# Start Ollama (keep running in background)
ollama serve

# Pull the default model (~4.7 GB, one-time)
ollama pull llama3.1:8b

# Generate the sample PDF
python create_sample_pdf.py
```

Verify Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

---

## 5. Running the terminal client

```bash
python terminal_client.py
```

The client will:
1. Check Ollama is reachable.
2. Ingest PDFs into ChromaDB (idempotent; skips if already populated).
3. Show the disclaimer and wait for `I agree`.
4. Loop on `You:` prompt. Type `exit`/`quit`/`bye` to leave.

---

## 6. Running the API server

```bash
uvicorn main:app --reload --port 8000
```

Open the auto-generated docs at `http://localhost:8000/docs`.

---

## 7. API endpoints

| Method | Path                       | Description                                |
|--------|----------------------------|--------------------------------------------|
| GET    | `/health`                  | Status, Ollama availability, chunk count.  |
| POST   | `/session/create`          | Create a new session; returns disclaimer.  |
| POST   | `/chat/{session_id}`       | Send a chat message; returns reply.        |
| DELETE | `/session/{session_id}`    | End a session.                             |
| GET    | `/sessions`                | List active session IDs.                   |

Example:

```bash
SID=$(curl -s -X POST localhost:8000/session/create | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -s -X POST localhost:8000/chat/$SID -H 'Content-Type: application/json' -d '{"message":"I agree"}'
curl -s -X POST localhost:8000/chat/$SID -H 'Content-Type: application/json' -d '{"message":"Tell me about oranges"}'
```

---

## 8. Adding PDFs

Drop any `.pdf` files into the `pdfs/` folder, then either:

- Restart the API server / terminal client (auto-ingests on first run), or
- Run `python ingest.py --force` to wipe and re-ingest everything.

Chunking rules: sentence-aligned, ~400 chars target / 600 max, with the last sentence of each chunk overlapped into the next.

---

## 9. Adding Predefined Q&A

In `.env`, add entries of the form:

```dotenv
PREDEFINED_QA_6={"q": "Your question here?", "a": "Your answer here."}
PREDEFINED_QA_7={"q": "...", "a": "..."}
```

The loader scans `PREDEFINED_QA_1`, `PREDEFINED_QA_2`, … and stops at the first missing index. Matching uses semantic similarity (`all-MiniLM-L6-v2`) with a configurable threshold.

---

## 10. Changing the model

```bash
ollama pull llama3.2:3b          # lighter / faster (~2 GB)
echo 'OLLAMA_MODEL=llama3.2:3b' >> .env
```

Any Ollama-compatible chat model can be used. `llama3.2:3b` is good for laptops; `llama3.1:8b` gives noticeably better answers.

---

## 11. Environment variables

| Variable                    | Default                                  | Description                                  |
|-----------------------------|------------------------------------------|----------------------------------------------|
| `OLLAMA_BASE_URL`           | `http://localhost:11434`                 | Ollama server URL.                           |
| `OLLAMA_MODEL`              | `llama3.1:8b`                            | Local LLM model name.                        |
| `SUPPORT_EMAIL`             | `support@portal.gov`                     | Shown in fallback messages.                  |
| `SUPPORT_PHONE`             | `1800-XXX-XXXX`                          | Shown in fallback messages.                  |
| `SUPPORT_URL`               | *(empty)*                                | Shown in fallback messages.                  |
| `SEARCH_API_URL`            | `https://api.example.gov/search`         | Placeholder (mocked).                        |
| `CDO_DETAILS_API_URL`       | `https://api.example.gov/cdo/details`    | Placeholder (mocked).                        |
| `DATASET_CDO_API_URL`       | `https://api.example.gov/dataset/cdo`    | Placeholder (mocked).                        |
| `FEEDBACK_API_URL`          | `https://api.example.gov/feedback`       | Placeholder (mocked).                        |
| `CONTACT_CDO_API_URL`       | `https://api.example.gov/contact-cdo`    | Placeholder (mocked).                        |
| `PREDEFINED_QA_N`           | 5 sample pairs                           | JSON `{"q": ..., "a": ...}`.                 |
| `CHROMA_DB_PATH`            | `./chroma_db`                            | Persistent ChromaDB folder.                  |
| `PDF_FOLDER`                | `./pdfs`                                 | Folder scanned for PDFs at startup.          |
| `RAG_RELEVANCE_THRESHOLD`   | `0.30`                                   | Min cosine similarity to keep a chunk.       |
| `QA_MATCH_THRESHOLD`        | `0.75`                                   | Min similarity to match a predefined Q.      |
| `RAG_TOP_K`                 | `5`                                      | Top-K chunks retrieved from ChromaDB.        |

---

## 12. Offline deployment

After the first run (which downloads the `all-MiniLM-L6-v2` embedding model from Hugging Face into the local cache and pulls the Ollama model), the system needs **no internet at runtime**. All embeddings and inference happen locally.

For an air-gapped deployment, pre-cache the embedding model on a machine with internet:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

Then copy `~/.cache/huggingface` and the `ollama` models directory to the target machine.

---

## 13. Limitations

- Session store is in-memory — sessions are lost on restart. Swap for Redis for production.
- External APIs (`search`, `cdo_details`, …) are mocked. Replace the bodies in `app/apis.py` with real `httpx` calls when endpoints are available.
- Political-topic filtering is keyword-based — it is intentionally aggressive but not exhaustive.
- The RAG pipeline trusts the local LLM. Despite low temperature (0.1) and a strict system prompt, occasional hallucinations are possible; the disclaimer warns users of this.
- ChromaDB is single-process; concurrent writers may conflict.
