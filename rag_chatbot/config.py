"""Central configuration for the RAG chatbot.

Everything can be overridden with environment variables (or a .env file),
so the same code works with a different Gemini model, chunk size, etc.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load a .env file from the project root if present (never commit real keys).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Paths -----------------------------------------------------------------
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", PROJECT_ROOT / "docs"))
INDEX_PATH = Path(os.getenv("RAG_INDEX_PATH", PROJECT_ROOT / "index" / "vector_store.json"))

# --- Models (Section 2 / 3 of the workshop) --------------------------------
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "gemini-3.1-flash-lite")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "models/gemini-embedding-001")
TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.2"))  # low = factual answers
MAX_OUTPUT_TOKENS = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", "1024"))

# --- Chunking / retrieval (Section 7) ---------------------------------------
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
# Gemini's free tier allows ~100 embedding requests/minute and counts every chunk in a
# batch as one request, so embed in small batches and back off on 429 errors.
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "20"))
EMBED_MAX_RETRIES = int(os.getenv("RAG_EMBED_MAX_RETRIES", "8"))

# --- Memory (Section 5) -----------------------------------------------------
# Keep only the last N message pairs to bound prompt size on long chats.
MAX_HISTORY_TURNS = int(os.getenv("RAG_MAX_HISTORY_TURNS", "10"))

SUPPORTED_EXTENSIONS = {".ipynb", ".md", ".txt", ".pdf"}


def require_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Create a free key at "
            "https://aistudio.google.com/app/apikey and put it in a .env file "
            "(see .env.example) or export it in your shell."
        )
    return key
