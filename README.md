# data.gov.in RAG Chatbot (Fully Offline / Local)

A production-ready, fully offline RAG chatbot for **data.gov.in**, the Open Government Data (OGD) Platform India. All LLM inference runs locally via Ollama, embeddings are computed locally with `sentence-transformers`, and the vector store (ChromaDB) is persisted to disk. **Zero external API calls at runtime.**

The same codebase runs on a laptop, on a 32-core / 128 GB workstation, and on a GPU cloud VM — it auto-detects the host and picks an appropriate model + Ollama runtime configuration via the hardware-profile system in `app/profile.py`.

---

## Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Prerequisites](#3-prerequisites)
4. [Local setup (bare metal)](#4-local-setup-bare-metal)
5. [Running the terminal client (streams by default)](#5-running-the-terminal-client)
6. [Running the REST API](#6-running-the-rest-api)
6a. [Running with Docker](#6a-running-with-docker)
6b. [Hardware profiles](#6b-hardware-profiles)
7. [API endpoints](#7-api-endpoints)
8. [Streaming responses (SSE)](#8-streaming-responses-sse)
9. [Adding PDFs to the corpus](#9-adding-pdfs-to-the-corpus)
10. [Adding predefined Q&A](#10-adding-predefined-qa)
11. [Changing models](#11-changing-models)
12. [Environment variables](#12-environment-variables)
13. [Offline / air-gapped deployment](#13-offline--air-gapped-deployment)
14. [Troubleshooting](#14-troubleshooting)
15. [Limitations](#15-limitations)

---

## 1. Overview

The chatbot:

- Answers questions strictly from the seven data.gov.in PDFs in `pdfs/` (About, Help, FAQ, NDSAP Implementation Guidelines, Terms of Use, Miscellaneous Policies, Accessibility Statement) and from the 12 predefined Q&A pairs in `.env`.
- Routes user messages to one of seven intents — `search`, `cdo_details`, `dataset_cdo_link`, `portal_feedback`, `contact_cdo`, `retry`, or `rag_chat` — via a small-model JSON classifier; the five non-RAG intents call mocked `app/apis.py` functions, and `retry` re-runs the user's clarified question through RAG with the predefined-Q&A fast-path disabled (no keyword-matching for dissatisfaction — the classifier learns it from few-shot examples).
- Runs a four-stage moderation pipeline before generation: keyword pre-filter, LLM categorical moderator (`SAFE` / `POLITICAL` / `INJECTION` / `OOS`), semantic scope allowlist, and optional post-generation grounding verification. Defense is positional and structural, not enumerative.
- **Streams** the RAG answer token-by-token over HTTP SSE and to the terminal — first words appear in ~1 s instead of the user staring at a blank prompt for 30 s.
- Requires disclaimer acceptance before every chat session and maintains per-session conversation history (in-memory).

---

## 2. Architecture

| Layer            | Component                                  |
|------------------|--------------------------------------------|
| LLM inference    | Ollama (local). Profile-driven: main + helper models. |
| Default main     | `llama3.1:8b` (8 GB RAM, ~10–15 tok/s on CPU). |
| Default helper   | `llama3.2:1b` (600 MB, used for moderator / intent / verifier). |
| Embeddings       | `sentence-transformers` `all-MiniLM-L6-v2` |
| Vector store     | ChromaDB (persistent, cosine, on disk)     |
| PDF parsing      | `pdfplumber` (sentence-aligned chunking, ~400 char target). |
| API framework    | FastAPI + uvicorn + SSE streaming.         |
| Session store    | In-memory Python dict (swap for Redis in prod). |
| Intent routing   | Ollama single-shot JSON classification (helper model). |

### Per-turn flow

```
┌────────────┐   ┌────────────────────┐   ┌──────────────────────┐
│ Disclaimer │ → │ Intent classifier  │ → │ Non-RAG: mocked API  │
│  gate      │   │  (helper model)    │   │  → emit & history    │
└────────────┘   └────────────────────┘   └──────────────────────┘
                  │
                  ├─ retry (with prior turn) → RAG path with QA fast-path
                  │                            disabled; prefix the answer
                  │                            with "Apologies — let me
                  │                            try that again."
                  │
                  ▼ (rag_chat)
            ┌─────────────────────────────┐
            │ 1. Categorical moderator    │  SAFE → continue
            │   (helper, max 4 tokens)    │  POLITICAL → POL refusal
            │                             │  INJECTION → INJ refusal
            │                             │  OOS → OOS refusal
            ├─────────────────────────────┤
            │ 2. Predefined Q&A           │  semantic match ≥ 0.85 → emit
            │   (skipped on retry)        │
            ├─────────────────────────────┤
            │ 3. Retrieve top-K chunks    │
            │   + filter by relevance     │
            ├─────────────────────────────┤
            │ 4. Scope gate (corpus +     │  neither passes → OOS refusal
            │    prototype allowlist)     │  only prototype → CONV reply
            ├─────────────────────────────┤
            │ 5. Stream answer from main  │  ← SSE / terminal stream
            │    model with context       │
            ├─────────────────────────────┤
            │ 6. (optional) Grounding     │  non-streaming: replace w/ OOS
            │    verifier (helper)        │  streaming: append warning
            └─────────────────────────────┘
```

### Retry — handling user dissatisfaction

When a user follows up with a contrastive correction (`no no I am asking about responsibilities of CDO`, `actually I meant who enforces it`, `that's not what I meant`), the intent classifier emits `retry` with `extracted` containing the clarified question. The chat handler then:

1. Bypasses the predefined-Q&A fast-path (`skip_predefined_qa=True`) so the same canned answer doesn't fire again.
2. Re-runs the **clarified** query through RAG so the bot grounds in the actual documentation chunks.
3. Prefixes the reply with `"Apologies for the previous reply — let me try that again."` so the user sees their dissatisfaction was registered.

Detection is **not keyword-based**. The helper model recognises the *shape* of dissatisfaction (contrastive markers tied to a re-stated question, in the presence of a previous assistant turn) via four few-shot examples — three positive (`no no`, `actually`, `that's not what I meant`) and two negative (`got it, thanks`, `and what about Y` as a satisfied topic shift). New phrasings the model has never seen get caught structurally the same way the categorical moderator catches new injection patterns.

`retry` only fires when there's a prior assistant turn in the session history. On the first turn, the classifier defaults to `rag_chat`.

### Injection defense

User input is wrapped in `<USER_INPUT>...</USER_INPUT>` tags in every LLM-facing prompt (moderator, intent classifier). The system prompt's Rule 10 instructs the main model to treat user messages as data, not as instructions. The categorical moderator has explicit `INJECTION` few-shots so override attempts (`ignore previous instructions`, `pretend you are`, `disregard your rules`) are caught structurally without enumerating bad words. Each layer is independent; the bot can only emit what's in the corpus, so even total moderation failure cannot make it answer outside scope.

---

## 3. Prerequisites

- Python **3.11+** (project tested on 3.13).
- [Ollama](https://ollama.com/download) installed and running locally **or** Docker / Docker Compose.
- At least 8 GB free RAM for the default `llama3.1:8b`, plus ~700 MB for the `llama3.2:1b` helper. The `xl` profile needs ~50 GB RAM (CPU) or a 40 GB+ GPU for `llama3.1:70b`.
- ~5 GB free disk for the Ollama models, ~90 MB for the sentence-transformer, plus your `chroma_db/` (small — a few MB per dozen PDF pages).

---

## 4. Local setup (bare metal)

```bash
git clone <repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                   # edit if you have custom values

ollama serve &                         # background: keep running
ollama pull llama3.1:8b                # main model (~4.7 GB)
ollama pull llama3.2:1b                # helper model (~700 MB)
```

Verify Ollama:

```bash
curl http://localhost:11434/api/tags
```

PDFs in `pdfs/` are ingested at first startup. To force re-ingest after dropping new PDFs:

```bash
python ingest.py --force
```

---

## 5. Running the terminal client

```bash
python terminal_client.py
```

The client:

1. Verifies Ollama is reachable and both configured models are pulled.
2. Ingests PDFs into ChromaDB (idempotent — skips when already populated).
3. Runs startup health checks and prints the active hardware profile.
4. Shows the disclaimer; waits for `I agree`.
5. Loops on `You: ` prompt. **Bot output streams token-by-token** so the first words appear in ~1 second.
6. Type `exit`, `quit`, or `bye` to leave.

Example streaming output:

```
You: what is NDSAP?
Bot: NDSAP is the National Data Sharing and Accessibility Policy of the
Government of India. It applies to all data and information…
```

(Each phrase appears as the model produces it.)

---

## 6. Running the REST API

```bash
uvicorn main:app --reload --port 8000
```

Auto-generated docs at `http://localhost:8000/docs`.

---

## 6a. Running with Docker

Two services in `docker-compose.yml`: `ollama` (official image) + `app` (built from local `Dockerfile`). The app reaches Ollama at `http://ollama:11434` over the internal compose network.

### One-time setup

```bash
cp .env.example .env                   # adjust if you have custom values
docker compose build                   # build the app image (~2–3 GB; torch is the bulk)
docker compose up -d ollama            # start Ollama; wait for healthcheck (~30 s)
docker exec datagovin-ollama ollama pull llama3.1:8b    # main model
docker exec datagovin-ollama ollama pull llama3.2:1b    # helper model
docker compose up -d app               # start the chatbot API on :8000
```

Verify:

```bash
curl http://localhost:8000/health
curl http://localhost:11434/api/tags
```

### Daily commands

```bash
docker compose up -d                   # bring everything up
docker compose logs -f app             # follow chatbot logs (incl. profile line)
docker compose down                    # stop, preserve volumes
docker compose down -v                 # stop AND wipe named volumes (Ollama models, HF cache)
```

### Volumes

| Mount                  | Type   | Purpose                                                                                  |
|------------------------|--------|------------------------------------------------------------------------------------------|
| `ollama_models`        | named  | Ollama model files (large; survives rebuilds).                                           |
| `hf_cache`             | named  | `sentence-transformers` model cache (~90 MB).                                            |
| `./pdfs`               | bind   | Drop PDFs here from the host; restart `app` or `docker exec datagovin-chatbot python ingest.py --force`. |
| `./chroma_db`          | bind   | Persistent vector store; visible from the host for inspection.                           |

### Swap the LLM main model (from the host)

```bash
docker exec datagovin-ollama ollama pull qwen2.5:14b
# in .env: OLLAMA_MODEL=qwen2.5:14b
docker compose restart app
```

The active hardware profile auto-picks both models, but explicit `OLLAMA_MODEL` / `OLLAMA_HELPER_MODEL` in `.env` always win.

### Pre-pinning models in RAM

`docker-compose.yml` passes `OLLAMA_KEEP_ALIVE` (default `-1`) to the Ollama daemon, so models stay loaded forever once first used. With 128 GB of host RAM this is free; if RAM is tight, set `OLLAMA_KEEP_ALIVE=5m` to age them out.

### Notes

- The app container needs internet on first boot to download the sentence-transformer (~90 MB). Cached in `hf_cache` thereafter.
- Bind-mounted `pdfs/` and `chroma_db/` are root-owned inside the container by default. If host-side permissions are awkward, change them to named volumes.
- **GPU passthrough**: add `deploy.resources.reservations.devices` to the `ollama` service when you wire up an NVIDIA card. Worth it for the `xl` profile / 70b model.

---

## 6b. Hardware profiles

`app/profile.py` detects RAM, CPU cores, and NVIDIA GPU at boot, then picks one of five tiered profiles. Each profile bundles `{main_model, helper_model, keep_alive, num_thread, num_ctx, enable_verification, embed_batch_size}`.

| Profile  | Trigger                  | Main model        | Helper model    | `keep_alive`   | Verifier |
|----------|--------------------------|-------------------|-----------------|----------------|----------|
| `tiny`   | < 16 GB RAM, no GPU      | `llama3.2:3b`     | `llama3.2:1b`   | 5 m            | off      |
| `small`  | 16–32 GB RAM, no GPU     | `llama3.1:8b`     | `llama3.2:1b`   | 5 m            | off      |
| `medium` | 32–64 GB RAM, no GPU     | `llama3.1:8b`     | `llama3.2:1b`   | 30 m           | on       |
| `large`  | 64+ GB RAM, no GPU       | `llama3.1:8b`     | `llama3.2:1b`   | forever        | on       |
| `xl`     | any NVIDIA GPU detected  | `llama3.1:70b`    | `llama3.1:8b`   | forever        | on       |

At startup you'll see (after env overrides are applied):

```
[profile] active=large (auto) | main=llama3.1:8b helper=llama3.2:1b | keep_alive=-1 | verify=True | ram=128.0GB cores=32 gpu=False
```

### Forcing a profile

Auto-detection misreads container limits (a 128 GB host running an 8 GB container looks like `tiny` from inside). Force a tier:

```bash
export PROFILE_OVERRIDE=large          # or tiny / small / medium / xl
```

In Docker:

```bash
PROFILE_OVERRIDE=large docker compose up -d
```

### Overriding individual knobs

Per-knob env wins over the profile:

```dotenv
OLLAMA_MODEL=qwen2.5:14b              # main RAG model
OLLAMA_HELPER_MODEL=llama3.2:3b       # moderator / intent / verifier
OLLAMA_KEEP_ALIVE=-1                  # how long Ollama holds models in RAM
OLLAMA_NUM_THREAD=0                   # 0 = let Ollama decide
OLLAMA_NUM_CTX=8192                   # context window for the main model
EMBED_BATCH_SIZE=128                  # sentence-transformer batch size
ENABLE_OUTPUT_VERIFICATION=true       # post-generation grounding check
```

### When to nudge things

| Symptom                                            | Suggested change                                                           |
|----------------------------------------------------|----------------------------------------------------------------------------|
| Answers feel slow on a big box                     | `PROFILE_OVERRIDE=large` — pins models, raises ctx and embed batch.        |
| Bot occasionally hallucinates a fact               | `ENABLE_OUTPUT_VERIFICATION=true` or jump a tier.                          |
| Helper/moderator latency too high                  | Already on helper model; you can't go smaller without quality loss.        |
| You added a GPU                                    | `PROFILE_OVERRIDE=xl` — or just leave auto, `nvidia-smi` check picks it up. |
| Container shows `tiny` profile but host is bigger  | `PROFILE_OVERRIDE=<tier>` — cgroup limits hid the host specs from psutil.  |

Add or edit profiles in `app/profile.py:PROFILES` for custom tiers.

---

## 7. API endpoints

| Method | Path                            | Description                                                 |
|--------|---------------------------------|-------------------------------------------------------------|
| GET    | `/health`                       | Status, Ollama availability, vector store chunk count.      |
| POST   | `/session/create`               | Create a new session; returns `session_id` + disclaimer text. |
| POST   | `/chat/{session_id}`            | Send a chat message; returns the **full** reply (blocking). |
| POST   | `/chat/{session_id}/stream`     | Send a chat message; **stream** the reply via SSE.          |
| DELETE | `/session/{session_id}`         | End a session and free its history.                         |
| GET    | `/sessions`                     | List active session IDs.                                    |

### Non-streaming example

```bash
SID=$(curl -s -X POST localhost:8000/session/create \
  | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -s -X POST localhost:8000/chat/$SID \
  -H 'Content-Type: application/json' \
  -d '{"message":"I agree"}'
curl -s -X POST localhost:8000/chat/$SID \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is NDSAP?"}'
```

---

## 8. Streaming responses (SSE)

`POST /chat/{session_id}/stream` returns `text/event-stream`. Each event is a JSON line. The stream ends with `data: {"done": true}`. On error, the last event is `data: {"error": "..."}`.

```bash
SID=$(curl -s -X POST localhost:8000/session/create \
  | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
curl -s -X POST localhost:8000/chat/$SID \
  -H 'Content-Type: application/json' \
  -d '{"message":"I agree"}'
curl -N -X POST localhost:8000/chat/$SID/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is NDSAP?"}'
```

Sample output:

```
data: {"chunk": "NDSAP is the National "}
data: {"chunk": "Data Sharing and "}
data: {"chunk": "Accessibility Policy..."}
data: {"done": true}
```

### What streams vs. what doesn't

| Path                                           | Stream behaviour                                                     |
|------------------------------------------------|----------------------------------------------------------------------|
| Disclaimer / empty input                       | Single chunk + `done`.                                               |
| Moderator → `INJECTION` / `POLITICAL` / `OOS`  | Single chunk + `done`.                                               |
| Predefined Q&A match                           | Single chunk + `done`.                                               |
| Scope → `CONVERSATIONAL_HELP`                  | Single chunk + `done`.                                               |
| Mocked API intents (search / CDO / …)          | Single chunk + `done`.                                               |
| Grounded RAG answer                            | Many chunks, one per model emit, then `done`.                        |

### Output verifier in streaming mode

When `ENABLE_OUTPUT_VERIFICATION=true` and a RAG answer fails grounding:

- **Non-streaming `/chat`**: the answer is *replaced* with the out-of-scope refusal (strict mode).
- **Streaming `/chat/.../stream`**: already-streamed tokens cannot be retracted, so a warning footer is *appended*:
  > [Note: I'm not fully confident this answer is grounded in the official data.gov.in documentation. Please verify on the portal before relying on it.]

If you absolutely need verification to gate the answer, use the blocking endpoint. Otherwise the streamed UX is the better default.

### JS / SSE client snippet

```js
const sid = (await (await fetch('/session/create', { method: 'POST' })).json()).session_id;
await fetch(`/chat/${sid}`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'I agree' }),
});

const res = await fetch(`/chat/${sid}/stream`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'What is NDSAP?' }),
});
const reader = res.body.getReader();
const decoder = new TextDecoder();
let buf = '';
while (true) {
  const { value, done } = await reader.read();
  if (done) break;
  buf += decoder.decode(value, { stream: true });
  for (const line of buf.split('\n\n')) {
    if (!line.startsWith('data: ')) continue;
    const ev = JSON.parse(line.slice(6));
    if (ev.chunk) process.stdout.write(ev.chunk);
    if (ev.done)  return;
    if (ev.error) throw new Error(ev.error);
  }
  buf = buf.endsWith('\n\n') ? '' : buf.split('\n\n').pop();
}
```

---

## 9. Adding PDFs to the corpus

Drop `.pdf` files into `pdfs/`. Then either:

- Restart the server / terminal client (auto-ingests if `chroma_db/` is empty), or
- Run `python ingest.py --force` to wipe and re-ingest everything.

Chunking rules: sentence-aligned, ~400 char target / 600 char max, last sentence overlapped into the next chunk. Source file recorded in metadata for citation.

---

## 10. Adding predefined Q&A

In `.env`, add entries:

```dotenv
PREDEFINED_QA_13={"q": "Your question here?", "a": "Your answer here."}
PREDEFINED_QA_14={"q": "...", "a": "..."}
```

The loader scans `PREDEFINED_QA_1`, `PREDEFINED_QA_2`, … and stops at the first missing index. Matching uses semantic similarity (`all-MiniLM-L6-v2`); a question hits if `QA_MATCH_THRESHOLD` is cleared.

The startup health check warns if your Q&A look semantically misaligned with `SCOPE_TOPICS` — useful guard against a stale `.env` carried over from a previous corpus.

---

## 11. Changing models

```bash
ollama pull qwen2.5:14b                # better grounding compliance than 8b, manageable speed
echo 'OLLAMA_MODEL=qwen2.5:14b' >> .env
# restart server / docker compose restart app
```

Any Ollama-compatible chat model can be used. Suggested tiers:

| Model                  | RAM     | Tok/s (32-core CPU) | Use case                                |
|------------------------|---------|---------------------|-----------------------------------------|
| `llama3.2:3b`          | ~3 GB   | ~25–35              | Tiny boxes, fastest, lowest quality.    |
| `llama3.1:8b`          | ~6 GB   | ~10–15              | Default sweet spot.                     |
| `qwen2.5:14b-q4_K_M`   | ~10 GB  | ~6–9                | Better refusal compliance / grounding.  |
| `llama3.1:70b-q4_K_M`  | ~42 GB  | ~1–2                | Best quality, viable on CPU but slow.   |

With a 24 GB GPU (RTX 3090 / 4090), `llama3.1:8b` runs at ~60 tok/s, `70b` at ~30 tok/s — both interactive.

---

## 12. Environment variables

Profile-driven defaults: per-knob `.env` always wins; otherwise the active hardware profile decides.

### Required for the corpus / domain

| Variable                     | Default                                  | Purpose                                            |
|------------------------------|------------------------------------------|----------------------------------------------------|
| `SUPPORT_EMAIL`              | `ndsap@gov.in`                           | Shown in refusal / out-of-scope responses.         |
| `SUPPORT_PHONE`              | `011-24363692`                           | Same.                                              |
| `SUPPORT_URL`                | `https://data.gov.in/contact`            | Same.                                              |
| `PREDEFINED_QA_N`            | 12 pairs                                 | Per-pair JSON `{"q": ..., "a": ...}`.              |

### Ollama / models

| Variable                     | Default (effective)                      | Purpose                                            |
|------------------------------|------------------------------------------|----------------------------------------------------|
| `OLLAMA_BASE_URL`            | `http://localhost:11434`                 | Ollama server URL.                                 |
| `OLLAMA_MODEL`               | profile main model                       | Main RAG model.                                    |
| `OLLAMA_HELPER_MODEL`        | profile helper model                     | Used by moderator / intent / verifier.             |
| `OLLAMA_KEEP_ALIVE`          | profile keep_alive (`5m` … `-1`)         | How long Ollama keeps a model loaded after a call. |
| `OLLAMA_NUM_THREAD`          | `0` (auto)                               | Ollama `num_thread` option.                        |
| `OLLAMA_NUM_CTX`             | profile num_ctx                          | Context window of the main model.                  |
| `PROFILE_OVERRIDE`           | *(unset)*                                | Force a profile: `tiny`/`small`/`medium`/`large`/`xl`. |

### Vector store / RAG

| Variable                     | Default                                  | Purpose                                            |
|------------------------------|------------------------------------------|----------------------------------------------------|
| `CHROMA_DB_PATH`             | `./chroma_db`                            | Persistent ChromaDB folder.                        |
| `PDF_FOLDER`                 | `./pdfs`                                 | Folder scanned at startup.                         |
| `RAG_RELEVANCE_THRESHOLD`    | `0.45`                                   | Min cosine similarity to keep a chunk.             |
| `RAG_TOP_K`                  | `5`                                      | Top-K chunks per retrieval.                        |
| `QA_MATCH_THRESHOLD`         | `0.85`                                   | Min cosine similarity to match a predefined Q. Raised from 0.75 to 0.85 so loose paraphrases fall through to RAG instead of returning an over-eager canned answer. |
| `EMBED_BATCH_SIZE`           | profile embed_batch_size                 | sentence-transformer batch size.                   |

### Moderation & scope

| Variable                     | Default                                  | Purpose                                            |
|------------------------------|------------------------------------------|----------------------------------------------------|
| `LLM_MODERATION_ENABLED`     | `true`                                   | Run the 4-way categorical moderator.               |
| `ENABLE_OUTPUT_VERIFICATION` | profile value (off for tiny/small)       | Post-generation grounding check.                   |
| `SCOPE_THRESHOLD`            | `0.45`                                   | Min cosine similarity vs `SCOPE_TOPICS`.           |
| `SCOPE_TOPICS`               | 38 phrases                               | JSON array of example utterances the bot will handle. |

### Mocked APIs (placeholders until real endpoints exist)

| Variable                     | Default                                  |
|------------------------------|------------------------------------------|
| `SEARCH_API_URL`             | `https://api.example.gov/search`         |
| `CDO_DETAILS_API_URL`        | `https://api.example.gov/cdo/details`    |
| `DATASET_CDO_API_URL`        | `https://api.example.gov/dataset/cdo`    |
| `FEEDBACK_API_URL`           | `https://api.example.gov/feedback`       |
| `CONTACT_CDO_API_URL`        | `https://api.example.gov/contact-cdo`    |

---

## 13. Offline / air-gapped deployment

After the first run (which downloads `all-MiniLM-L6-v2` from Hugging Face into the local cache and pulls the Ollama models), the system needs **no internet at runtime**.

Pre-cache on a machine with internet:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ollama pull llama3.1:8b
ollama pull llama3.2:1b
```

Then copy `~/.cache/huggingface/` and the Ollama models directory (`~/.ollama/models/`) to the target machine.

For air-gapped Docker, do the same on a machine with internet, then `docker save` / `docker load` the `ollama/ollama:latest` image and the locally built `app` image, and rsync the volume contents over.

---

## 14. Troubleshooting

| Symptom                                                                       | Probable cause / fix                                                                                                                  |
|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `WARNING: No predefined Q&A pairs loaded`                                      | `.env` is empty or stale. `cp .env.example .env`.                                                                                     |
| `WARNING: Predefined Q&A look misaligned with SCOPE_TOPICS`                    | `.env` is from a different domain (e.g. the generic baseline). Sync from `.env.example`.                                              |
| `WARNING: Vector store is empty (0 chunks)`                                    | No PDFs ingested. Drop PDFs into `pdfs/` and run `python ingest.py --force`.                                                          |
| Bot replies with `CONVERSATIONAL_HELP_RESPONSE` to a real question             | Predefined Q&A missed AND no chunks above `RAG_RELEVANCE_THRESHOLD`. Re-ingest with the right corpus, or lower the threshold.         |
| Loose paraphrase returns a wrong canned QA answer                              | `QA_MATCH_THRESHOLD` is too low. Default is 0.85; bump higher if false matches persist. Or send a follow-up like `no, I meant <clarification>` — the `retry` intent re-runs through RAG. |
| Retry doesn't fire on `no no, I meant X`                                       | Confirm there's a prior assistant turn in the session (retry needs context). If yes, add another `retry` few-shot example to `INTENT_PROMPT_TEMPLATE`. |
| Bot answers a political question                                              | `LLM_MODERATION_ENABLED=false`? Set it back to `true`. Otherwise add a few-shot example to `SENSITIVITY_PROMPT`.                       |
| Cricket / sports question wrongly hits `POLITICAL_REFUSAL`                    | Moderator over-flagging. Strengthen the OOS examples in `SENSITIVITY_PROMPT`.                                                         |
| Cold start of ~10 s on each turn                                              | `OLLAMA_KEEP_ALIVE` is too short. Set to `-1` (or jump to `large` profile).                                                          |
| First RAG turn slow then subsequent turns fast                                | Normal — Ollama loads the model into RAM on the first call. With `keep_alive=-1` it stays loaded.                                     |
| Streaming endpoint returns 200 but no data                                    | Reverse proxy is buffering. Set `proxy_buffering off` (Nginx) or `X-Accel-Buffering: no` (already set in this server).                |
| Auto-detected `tiny` profile inside Docker but host is huge                   | cgroup limits hide host specs. Use `PROFILE_OVERRIDE=large` (or whichever tier matches).                                              |
| `is_ollama_available()` returns False but Ollama is running                   | The configured helper model isn't pulled. `ollama pull llama3.2:1b`.                                                                  |

---

## 15. Limitations

- **Session store is in-memory.** Sessions are lost on restart. Swap for Redis for multi-instance / production.
- **External APIs are mocked.** The five `app/apis.py` functions return canned responses. Replace bodies with real `httpx` calls when endpoints exist.
- **Single-process ChromaDB.** Concurrent writers may conflict. Multiple workers behind a load balancer need a shared store (Postgres-backed Chroma, pgvector, etc.).
- **Moderator is small-model-driven.** `llama3.2:1b` is well-suited to the constrained yes/no/label tasks but can be fooled by sufficiently clever inputs. The four-gate defense (input delimiters + categorical moderator + scope allowlist + corpus-as-floor) raises the bar a lot, but no single layer is bulletproof.
- **Grounding verifier is opt-in.** Off by default on the smaller profiles. On `medium`/`large`/`xl` it's on by default; in streaming mode it appends a warning footer rather than retracting tokens.
- **No GPU detection beyond NVIDIA.** Apple Silicon (Metal) and AMD ROCm aren't auto-detected — they still work via Ollama, but the profile auto-picker won't promote to `xl`. Use `PROFILE_OVERRIDE=xl` manually.
