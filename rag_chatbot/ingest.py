"""Ingestion pipeline (Section 6/7 of the workshop):

    Document -> Load -> Chunk -> Embedding -> Vector Database

Run once to build the index, and again whenever the docs folder changes:

    python -m rag_chatbot.ingest            # build if missing
    python -m rag_chatbot.ingest --force    # rebuild from scratch
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Callable, TypeVar

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .loaders import load_directory


T = TypeVar("T")


def is_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


def _suggested_delay(exc: BaseException) -> float | None:
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.IGNORECASE)
    return float(m.group(1)) if m else None


def with_retry(
    fn: Callable[[], T],
    retries: int = config.EMBED_MAX_RETRIES,
    log: Callable[[str], None] | None = None,
) -> T:
    """Call ``fn`` with a hard timeout; on a Gemini 429/quota error wait and retry (bounded).

    The embeddings client does not honour a per-request timeout, so the call runs in a
    worker thread and we stop waiting after ``config.REQUEST_TIMEOUT`` seconds. That
    turns a hung network call into an error the UI can show instead of an endless spinner.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

    for attempt in range(1, retries + 1):
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                try:
                    return pool.submit(fn).result(timeout=config.REQUEST_TIMEOUT)
                except FutureTimeout:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise TimeoutError(
                        f"Gemini request did not respond within {config.REQUEST_TIMEOUT:.0f}s"
                    )
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt == retries:
                raise
            # Google may suggest waiting hours for a daily quota; cap it so we fail fast.
            delay = min(max(_suggested_delay(exc) or 0.0, 5.0 * attempt), config.EMBED_MAX_DELAY)
            if log:
                log(f"Gemini rate limit hit; waiting {delay:.0f}s (attempt {attempt}/{retries})")
            time.sleep(delay)
    raise RuntimeError("unreachable")


def embed_and_add(
    store: InMemoryVectorStore,
    chunks,
    batch_size: int = config.EMBED_BATCH_SIZE,
    log: Callable[[str], None] | None = None,
) -> None:
    """Add chunks to the store in small batches, retrying on rate limits."""
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        with_retry(lambda b=batch: store.add_documents(b), log=log)
        if log:
            log(f"  embedded {min(start + batch_size, len(chunks))}/{len(chunks)} chunks")


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    config.require_api_key()
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        transport=config.TRANSPORT,
        request_options={"timeout": config.REQUEST_TIMEOUT},
    )


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_documents(docs)


def build_index(
    docs_dir: Path = config.DOCS_DIR,
    index_path: Path = config.INDEX_PATH,
    verbose: bool = True,
) -> InMemoryVectorStore:
    """Load every supported file in ``docs_dir``, chunk, embed and persist."""
    log = print if verbose else (lambda *a, **k: None)

    docs = load_directory(docs_dir)
    if not docs:
        raise RuntimeError(
            f"No supported documents found in {docs_dir}. "
            f"Add .ipynb / .md / .txt / .pdf files and re-run."
        )
    log(f"Loaded {len(docs)} document(s) from {docs_dir}")

    chunks = split_documents(docs)
    log(f"Split into {len(chunks)} chunks (chunk_size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    embeddings = get_embeddings()
    log(f"Embedding with {config.EMBEDDING_MODEL} ...")
    store = InMemoryVectorStore(embeddings)
    embed_and_add(store, chunks, log=log)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    store.dump(str(index_path))
    log(f"Saved vector store with {len(chunks)} chunks to {index_path}")
    return store


def indexed_sources(store: InMemoryVectorStore) -> set[str]:
    """File names (metadata['source']) that already have chunks in the store."""
    return {
        rec["metadata"].get("source")
        for rec in store.store.values()
        if rec.get("metadata", {}).get("source")
    }


def add_files_to_index(
    store: InMemoryVectorStore,
    paths: list[Path],
    index_path: Path = config.INDEX_PATH,
) -> dict[str, int]:
    """Incrementally load, chunk and embed ``paths`` into an existing store, then persist.

    Returns ``{file_name: number_of_chunks_added}``. A count of 0 means the file
    produced no text (for example a scanned, image-only PDF). Files whose name is
    already in the index are skipped so re-uploads don't create duplicate chunks.
    """
    from .loaders import load_file

    known = indexed_sources(store)
    added: dict[str, int] = {}
    for path in paths:
        if path.name in known:
            continue
        docs = load_file(path)
        chunks = split_documents(docs) if docs else []
        if chunks:
            embed_and_add(store, chunks)
        added[path.name] = len(chunks)
        known.add(path.name)
    if any(added.values()):
        index_path.parent.mkdir(parents=True, exist_ok=True)
        store.dump(str(index_path))
    return added


def sync_index(
    store: InMemoryVectorStore,
    docs_dir: Path = config.DOCS_DIR,
    index_path: Path = config.INDEX_PATH,
) -> dict[str, int]:
    """Index any file in ``docs_dir`` that is not yet in the store."""
    from .loaders import iter_supported_files

    known = indexed_sources(store)
    missing = [p for p in sorted(iter_supported_files(docs_dir)) if p.name not in known]
    return add_files_to_index(store, missing, index_path) if missing else {}


def load_index(index_path: Path = config.INDEX_PATH) -> InMemoryVectorStore:
    if not index_path.exists():
        raise FileNotFoundError(f"No index at {index_path}. Run `python -m rag_chatbot.ingest` first.")
    return InMemoryVectorStore.load(str(index_path), get_embeddings())


def load_or_build_index(
    docs_dir: Path = config.DOCS_DIR,
    index_path: Path = config.INDEX_PATH,
    force: bool = False,
    verbose: bool = True,
    sync: bool = True,
) -> InMemoryVectorStore:
    """Load the saved index (building it if absent).

    With ``sync=True`` any file in ``docs_dir`` missing from the index is embedded
    too; a rate-limit failure there is reported but does not prevent the already
    loaded index from being returned.
    """
    if index_path.exists() and not force:
        if verbose:
            print(f"Loading existing index from {index_path}")
        store = load_index(index_path)
        if sync:
            try:
                added = sync_index(store, docs_dir, index_path)
            except Exception as exc:
                print(f"Warning: could not index new files ({exc}); using saved index.", file=sys.stderr)
            else:
                if verbose:
                    for name, n in added.items():
                        print(f"Indexed new file {name}: {n} chunks")
        return store
    return build_index(docs_dir, index_path, verbose=verbose)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument("--docs", type=Path, default=config.DOCS_DIR, help="Folder of source documents")
    parser.add_argument("--index", type=Path, default=config.INDEX_PATH, help="Where to save the index")
    parser.add_argument("--force", action="store_true", help="Rebuild even if an index already exists")
    args = parser.parse_args(argv)

    try:
        load_or_build_index(args.docs, args.index, force=args.force)
    except Exception as exc:  # surface a clean message instead of a traceback
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
