"""Ingestion pipeline (Section 6/7 of the workshop):

    Document -> Load -> Chunk -> Embedding -> Vector Database

Run once to build the index, and again whenever the docs folder changes:

    python -m rag_chatbot.ingest            # build if missing
    python -m rag_chatbot.ingest --force    # rebuild from scratch
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config
from .loaders import load_directory


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    config.require_api_key()
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)


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
    store.add_documents(chunks)

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
    produced no text (for example a scanned, image-only PDF).
    """
    from .loaders import load_file

    added: dict[str, int] = {}
    for path in paths:
        docs = load_file(path)
        chunks = split_documents(docs) if docs else []
        if chunks:
            store.add_documents(chunks)
        added[path.name] = len(chunks)
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
) -> InMemoryVectorStore:
    if index_path.exists() and not force:
        if verbose:
            print(f"Loading existing index from {index_path}")
        store = load_index(index_path)
        # Pick up any files dropped into docs/ since the index was last saved.
        added = sync_index(store, docs_dir, index_path)
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
