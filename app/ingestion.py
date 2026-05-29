import pdfplumber
import re
import uuid
from pathlib import Path
from app.embedder import embed
from app.vector_store import add_documents, clear_collection, collection_count
from app.config import PDF_FOLDER


def extract_text_from_pdf(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(layout=False)
            if page_text:
                text_parts.append(page_text.strip())
    return "\n\n".join(text_parts)


# Tokens that end in a period but usually do NOT end a sentence. After the naive
# split we re-join any fragment ending in one of these, so honorifics and
# references ("Dr. Sharma", "Shri. Kumar", "Sec. A", "etc. The ...") are not
# fractured. Stored lowercase with internal dots removed.
_ABBREVIATIONS = {
    # honorifics / titles
    "dr", "mr", "mrs", "ms", "smt", "shri", "sri", "kum", "km", "prof", "hon",
    "capt", "col", "gen", "lt", "sgt", "rev", "st",
    # structural references
    "no", "nos", "sec", "secs", "art", "arts", "fig", "figs", "vol", "vols",
    "para", "paras", "pp", "cl", "reg", "regs", "rule", "ch", "pt", "ann",
    # organisations / business
    "govt", "dept", "deptt", "ltd", "pvt", "co", "corp", "inst", "assn", "bros",
    # latin / misc
    "etc", "viz", "vs", "ie", "eg", "al", "cf", "ibid", "approx", "est",
    "incl", "min", "max", "avg",
}

_LAST_TOKEN_RE = re.compile(r"""([A-Za-z][A-Za-z.&]*)\.["')\]]*\s*$""")


def _ends_with_abbreviation(text: str) -> bool:
    """True if `text` ends in a known abbreviation or a single-letter initial
    (e.g. the "A." in "Dr. A. Sharma"), meaning the next fragment should be
    rejoined rather than treated as a new sentence."""
    m = _LAST_TOKEN_RE.search(text)
    if not m:
        return False
    token = m.group(1).lower().replace(".", "")
    return token in _ABBREVIATIONS or len(token) == 1


def split_into_sentences(text: str) -> list[str]:
    text = re.sub(r"\n{3,}", "\n\n", text)
    paragraphs = text.split("\n\n")
    sentences = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Zऀ-ॿ])', para)
        # Re-merge fragments split on an abbreviation/initial instead of a real
        # sentence boundary. Over-merging is harmless for size-based chunking;
        # over-splitting fractures retrieval, which CLAUDE.md forbids.
        merged: list[str] = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if merged and _ends_with_abbreviation(merged[-1]):
                merged[-1] = f"{merged[-1]} {part}"
            else:
                merged.append(part)
        sentences.extend(merged)
    return sentences


def chunk_sentences(sentences: list[str], target_size: int = 400, max_size: int = 600) -> list[str]:
    chunks = []
    current_chunk = []
    current_len = 0
    last_sentence = None

    for sentence in sentences:
        sentence_len = len(sentence)

        if sentence_len > max_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            chunks.append(sentence)
            current_chunk = [sentence]
            current_len = sentence_len
            last_sentence = sentence
            continue

        if current_len + sentence_len + 1 > max_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            if last_sentence:
                current_chunk = [last_sentence, sentence]
                current_len = len(last_sentence) + sentence_len + 1
            else:
                current_chunk = [sentence]
                current_len = sentence_len
        else:
            current_chunk.append(sentence)
            current_len += sentence_len + 1

        last_sentence = sentence

        if current_len >= target_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [last_sentence] if last_sentence else []
            current_len = len(last_sentence) if last_sentence else 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c.strip() for c in chunks if c.strip()]


def ingest_pdfs(force_reingest: bool = False):
    if not force_reingest and collection_count() > 0:
        print(f"Vector store already has {collection_count()} chunks. Skipping ingestion.")
        return

    if force_reingest:
        print("Force re-ingesting: clearing existing collection.")
        clear_collection()

    pdf_folder = Path(PDF_FOLDER)
    pdf_files = list(pdf_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: No PDF files found in {PDF_FOLDER}.")
        return

    all_ids, all_embeddings, all_documents, all_metadatas = [], [], [], []

    for pdf_path in pdf_files:
        print(f"Ingesting: {pdf_path.name}")
        raw_text = extract_text_from_pdf(str(pdf_path))
        sentences = split_into_sentences(raw_text)
        chunks = chunk_sentences(sentences)
        print(f"  -> {len(chunks)} chunks extracted")

        for i, chunk in enumerate(chunks):
            all_ids.append(f"{pdf_path.stem}_chunk_{i}_{uuid.uuid4().hex[:8]}")
            all_documents.append(chunk)
            all_metadatas.append({"source_file": pdf_path.name, "chunk_index": i})

    if not all_documents:
        print("No text extracted from PDFs.")
        return

    print(f"Embedding {len(all_documents)} chunks...")
    # embed() already batches internally at EMBED_BATCH_SIZE, so a single call
    # is enough — the previous outer loop re-batched at a hardcoded 64 that also
    # ignored the configured batch size.
    all_embeddings = embed(all_documents)

    add_documents(all_ids, all_embeddings, all_documents, all_metadatas)
    print(f"Ingestion complete. {len(all_documents)} chunks stored.")
