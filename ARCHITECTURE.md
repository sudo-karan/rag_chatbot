# Architecture

How the data.gov.in RAG chatbot is put together, and why. The [README](README.md)
covers setup and usage; this document is the design and internals. Current gaps
and caveats live in the README's [Limitations](README.md#15-limitations).

## Goals and constraints

- **Fully offline at runtime.** Inference is local (Ollama), embeddings are local
  (`sentence-transformers`), and the vector store (ChromaDB) is on disk. Once the
  models are pulled, nothing calls out to the network.
- **Strict, grounded answers.** The assistant answers from a fixed corpus of
  data.gov.in PDFs plus a small set of predefined Q&A. It refuses politics,
  injection attempts, and out-of-scope questions instead of guessing.
- **Configuration over hardcoding.** Models, thresholds, paths, support contacts,
  and the scope allowlist all come from the environment.
- **Runs on varied hardware.** A hardware profile picks the models and Ollama
  runtime settings for the host, from a laptop to a multi-GPU server.

## Layout

```
main.py              FastAPI app: /session, /chat, /chat/.../stream, /health, /sessions
terminal_client.py   Local REPL client (interactive profile pick, then streams)
ingest.py            One-shot PDF ingestion CLI

app/
  config.py          Loads .env, resolves the active profile, exposes settings
  profile.py         Hardware detection + the tiered Profile table
  llm.py             The only module that talks to Ollama (chat / stream / helper)
  embedder.py        sentence-transformers wrapper + cosine helper (singleton model)
  vector_store.py    ChromaDB persistent collection (cosine space)
  ingestion.py       PDF -> text -> sentence-aligned chunks -> embeddings -> store
  predefined_qa.py   Loads .env Q&A pairs; semantic match against the question bank
  intent.py          Single-shot JSON intent classifier (helper model)
  moderation.py      Categorical moderator, scope allowlist, grounding verifier
  datasets.py        Local CSV/JSON dataset directory behind the mocked APIs
  apis.py            The five intent-routed functions (search, CDO lookup, feedback)
  rag.py             The gate plus grounded generation (blocking and streaming)
  chat.py            Per-turn orchestration: disclaimer, intent routing, history
  session.py         In-memory session store and conversation history
  health.py          Startup checks and the profile banner
```

## Two models per deployment

Every profile names two Ollama models:

- a **main** model for the user-facing answer (default `llama3.1:8b`), and
- a small **helper** model (default `llama3.2:1b`) for the cheap, structured jobs:
  intent classification, moderation, and the optional grounding check.

`app/llm.py` is the single choke point for Ollama. `chat()` / `chat_stream()` use
the main model; `helper()` uses the helper model and runs at an explicit context
window (`OLLAMA_HELPER_NUM_CTX`, default 4096) so the long classification prompts
aren't silently truncated at Ollama's 2048 default.

## Per-turn flow

Each message runs through `app/chat.py`:

1. **Disclaimer gate.** Until the session has accepted the disclaimer, every
   message except an affirmative ("I agree", "yes", ...) is turned away.
2. **Intent classification.** The helper model labels the message as one of
   `search`, `cdo_details`, `dataset_cdo_link`, `portal_feedback`, `contact_cdo`,
   `retry`, or `rag_chat`.
3. **Routing.**
   - The five non-RAG intents call the matching function in `app/apis.py`, which
     reads from the local dataset directory (`app/datasets.py`) or appends a
     feedback ticket to a JSONL log.
   - `retry` re-runs the user's clarified question through the RAG path with the
     Q&A fast-path disabled, so a "no, I meant X" correction reaches the corpus
     instead of repeating the same canned answer.
   - `rag_chat` (the default) goes through the gate below.

## The RAG gate

`app/rag.py` runs a sequence of gates before any generation:

1. **Moderation** (`app/moderation.py`, helper model). The message is labelled
   `SAFE` / `POLITICAL` / `INJECTION` / `OOS`. `POLITICAL` and `INJECTION` are
   hard refusals. An `OOS` label is *held* rather than acted on immediately.
2. **Predefined Q&A.** A semantic match at or above `QA_MATCH_THRESHOLD` (0.85)
   returns the operator-authored answer verbatim. A match here also overrides a
   held `OOS` label — the Q&A bank is in scope by construction, so an exact match
   is strong evidence the message is in scope. With no match, a held `OOS` label
   becomes a refusal.
3. **Retrieval + scope.** The query is embedded, the top-K chunks are pulled from
   ChromaDB and filtered by `RAG_RELEVANCE_THRESHOLD` (0.45). Scope is judged two
   ways: a relevant corpus hit, or similarity to one of the `SCOPE_TOPICS` example
   utterances above `SCOPE_THRESHOLD`. Neither passes -> out-of-scope refusal;
   only the allowlist passes -> a conversational help reply.
4. **Generation.** The main model answers from the retrieved chunks under a system
   prompt that forbids outside knowledge and fabrication.
5. **Grounding check (optional).** When enabled, the helper model judges whether
   the answer is supported by the context. Blocking responses are replaced with an
   out-of-scope refusal on failure; streamed responses (which can't be retracted)
   get a warning footer appended.

The same gate backs both the blocking and streaming paths, so they can't drift
apart. One consequence worth knowing: because moderation runs first and its `OOS`
verdict is only rescued by a Q&A match, a question the small model mislabels as
out-of-scope is refused before retrieval ever runs (see Limitations).

## Scope defense

Scope is enforced at several points rather than by a single keyword list:

- user input is wrapped in `<USER_INPUT>...</USER_INPUT>` tags in every LLM-facing
  prompt, and the main system prompt treats the message as data, not instructions;
- the categorical moderator classifies injection / political / out-of-scope intent
  from few-shot examples, not a blocklist;
- the scope allowlist and corpus relevance gate retrieval;
- generation is constrained to the retrieved chunks.

These are independent layers. They raise the bar against both letting a bad
message through and refusing a good one, but none is individually bulletproof.

## Hardware profiles

`app/profile.py` detects RAM, CPU cores, and NVIDIA GPU + VRAM, then recommends one
of six tiers (`tiny`, `small`, `medium`, `large`, `gpu`, `xl`). Each tier bundles
the main/helper models and Ollama runtime settings (keep-alive, context window,
thread count, embed batch size) plus whether the grounding verifier is on by
default. The `gpu` and `xl` tiers are gated on detected VRAM, so a small card never
auto-selects the 70B model.

Detection is lazy and resolved once in `config.py`. The recommendation is logged at
startup and surfaced on `/health`; the terminal client offers an interactive
accept/override prompt. Any per-knob environment variable (or `PROFILE_OVERRIDE`)
takes precedence over the auto choice.

## Ingestion and chunking

`app/ingestion.py` extracts text with `pdfplumber`, splits it into sentences, and
groups them into roughly 400-character chunks (600 max), carrying the last sentence
of each chunk into the next for continuity. The splitter re-joins fragments that
break on honorifics or abbreviations ("Dr.", "Shri.", "Sec.") so names and
references aren't fractured. Chunks are embedded in one batch and stored in ChromaDB
with their source filename for citation.

## Configuration

`app/config.py` loads `.env` and exposes every setting. The knobs that matter most:

- `OLLAMA_MODEL` / `OLLAMA_HELPER_MODEL` — override the profile's models.
- `QA_MATCH_THRESHOLD` (0.85), `RAG_RELEVANCE_THRESHOLD` (0.45),
  `SCOPE_THRESHOLD` (0.45), `RAG_TOP_K` (5) — retrieval and match tuning.
- `LLM_MODERATION_ENABLED`, `ENABLE_OUTPUT_VERIFICATION` — toggle the gates.
- `SCOPE_TOPICS`, `KNOWN_TOPICS_SUMMARY` — the in-scope allowlist and the redirect
  hint shown on an out-of-scope refusal.
- `PREDEFINED_QA_N` — the Q&A bank.

The full table is in the README's [Environment variables](README.md#12-environment-variables).
