#!/usr/bin/env python3
import argparse
from app.ingestion import ingest_pdfs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
    parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    args = parser.parse_args()
    ingest_pdfs(force_reingest=args.force)
