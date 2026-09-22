"""Document loading (Section 7.1 of the workshop).

Turns files on disk into LangChain ``Document`` objects. Supports:

* ``.ipynb`` - Jupyter notebooks. Markdown cells are kept as-is, code cells are
  wrapped in fenced code blocks, and text outputs are included so the chatbot
  can answer "what did this cell print?" style questions.
* ``.md`` / ``.txt`` - plain text.
* ``.pdf`` - via pypdf, one Document per page.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from .config import SUPPORTED_EXTENSIONS


def _output_text(output: dict) -> str:
    """Extract printable text from a notebook cell output, if any."""
    otype = output.get("output_type")
    if otype == "stream":
        return "".join(output.get("text", []))
    if otype in ("execute_result", "display_data"):
        data = output.get("data", {})
        text = data.get("text/plain")
        if text:
            return "".join(text) if isinstance(text, list) else str(text)
    if otype == "error":
        return "\n".join(output.get("traceback", []))
    return ""


def load_notebook(path: Path, include_outputs: bool = True) -> list[Document]:
    """Convert a .ipynb into one Document per cell (with a running section title).

    Each cell becomes its own Document so that the text splitter never merges
    unrelated cells, and so the source metadata can point back to a cell index
    and the section heading it lives under.
    """
    nb = json.loads(path.read_text(encoding="utf-8"))
    docs: list[Document] = []
    current_section = path.stem

    for idx, cell in enumerate(nb.get("cells", [])):
        source = cell.get("source", "")
        source = "".join(source) if isinstance(source, list) else source
        source = source.strip()
        if not source:
            continue

        ctype = cell.get("cell_type")
        if ctype == "markdown":
            # Track the most recent heading so every chunk knows its section.
            for line in source.splitlines():
                if line.startswith("#"):
                    current_section = line.lstrip("#").strip()
                    break
            content = source
        elif ctype == "code":
            lang = nb.get("metadata", {}).get("kernelspec", {}).get("language", "python")
            content = f"```{lang}\n{source}\n```"
            if include_outputs:
                outputs = [_output_text(o) for o in cell.get("outputs", [])]
                outputs = [o.strip() for o in outputs if o and o.strip()]
                if outputs:
                    joined = "\n".join(outputs)
                    # Cap very long outputs (e.g. giant model lists) to keep chunks useful.
                    if len(joined) > 1500:
                        joined = joined[:1500] + "\n... [output truncated]"
                    content += f"\n\nOutput:\n{joined}"
        else:
            continue

        docs.append(
            Document(
                page_content=f"Section: {current_section}\n\n{content}",
                metadata={
                    "source": path.name,
                    "cell_index": idx,
                    "cell_type": ctype,
                    "section": current_section,
                },
            )
        )
    return docs


def load_text(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [Document(page_content=text, metadata={"source": path.name})]


def load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader  # imported lazily so pypdf is optional

    reader = PdfReader(str(path))
    docs = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": page_no}))
    return docs


def load_file(path: Path) -> list[Document]:
    ext = path.suffix.lower()
    if ext == ".ipynb":
        return load_notebook(path)
    if ext in (".md", ".txt"):
        return load_text(path)
    if ext == ".pdf":
        return load_pdf(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_directory(directory: Path) -> list[Document]:
    """Load every supported file under ``directory`` (recursively)."""
    if not directory.exists():
        raise FileNotFoundError(f"Docs directory not found: {directory}")
    docs: list[Document] = []
    for path in sorted(iter_supported_files(directory)):
        docs.extend(load_file(path))
    return docs


def iter_supported_files(directory: Path) -> Iterable[Path]:
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path
