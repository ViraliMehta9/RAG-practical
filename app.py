"""Streamlit web UI for the RAG chatbot.

    streamlit run app.py

Visitors see a clean assistant with a browsable, downloadable knowledge base.
Document management (upload / index / rebuild) lives in an Admin panel that can be
protected with an ADMIN_PASSWORD secret or environment variable.
"""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import streamlit as st

from rag_chatbot import config
from rag_chatbot.chatbot import RAGChatbot
from rag_chatbot.ingest import (
    add_files_to_index,
    indexed_sources,
    is_rate_limit_error,
    load_or_build_index,
    unindexed_files,
)

APP_NAME = "GenAI Workshop Assistant"
TAGLINE = "Ask questions about the Practical Generative AI Workshop and its companion documents."
SUGGESTED_QUESTIONS = [
    "What does the temperature parameter control?",
    "Explain RAG in simple terms.",
    "Show me the code that builds the vector store.",
    "How many vacation days do Acme employees get?",
]
FILE_ICONS = {".ipynb": "📓", ".pdf": "📄", ".md": "📝", ".txt": "📃"}
MIME_TYPES = {
    ".ipynb": "application/x-ipynb+json",
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
}

st.set_page_config(page_title=APP_NAME, page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# --- Styling -------------------------------------------------------------------
st.markdown(
    """
<style>
#MainMenu, footer, [data-testid="stStatusWidget"] {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 960px;}
.hero {display:flex; align-items:center; gap:14px; padding: 6px 0 4px 0;}
.hero .logo {width:48px; height:48px; border-radius:14px; background:linear-gradient(135deg,#4F46E5,#8B5CF6);
             display:flex; align-items:center; justify-content:center; font-size:26px; color:white;}
.hero h1 {margin:0; font-size:1.7rem; line-height:1.2;}
.hero p {margin:2px 0 0 0; color:#6B7280; font-size:0.95rem;}
.pill {display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.75rem; font-weight:600;
       background:#EEF2FF; color:#3730A3; margin-right:6px; margin-top:8px;}
.pill.ok {background:#ECFDF5; color:#065F46;}
.pill.warn {background:#FFFBEB; color:#92400E;}
.doc-card {border:1px solid #E5E7EB; border-radius:12px; padding:10px 12px; margin-bottom:8px; background:white;}
.doc-card .name {font-weight:600; font-size:0.86rem; word-break:break-all;}
.doc-card .meta {color:#6B7280; font-size:0.75rem; margin-top:2px;}
.welcome {border:1px dashed #C7D2FE; border-radius:16px; padding:22px 24px; background:#F8FAFF; margin:14px 0 18px 0;}
.welcome h3 {margin:0 0 6px 0;}
.welcome p {margin:0; color:#4B5563;}
.source {font-size:0.82rem; color:#374151; padding:4px 0;}
.source b {color:#111827;}
[data-testid="stChatMessage"] {border-radius:14px; padding:10px 14px;}
</style>
""",
    unsafe_allow_html=True,
)

docs_dir: Path = config.DOCS_DIR
docs_dir.mkdir(parents=True, exist_ok=True)


# --- Secrets / keys --------------------------------------------------------------
def secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name] if name in st.secrets else None
    except Exception:
        return None


if not os.getenv("GOOGLE_API_KEY") and secret("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = secret("GOOGLE_API_KEY")  # type: ignore[arg-type]
ADMIN_PASSWORD = secret("ADMIN_PASSWORD")


# --- Helpers ---------------------------------------------------------------------
def get_bot(force: bool = False) -> RAGChatbot:
    """Load the saved index (fast, no API calls). Embedding only happens on demand."""
    if force or "bot" not in st.session_state:
        with st.spinner("Preparing the knowledge base..."):
            store = load_or_build_index(force=force, verbose=False, sync=False)
        st.session_state.bot = RAGChatbot(store)
    return st.session_state.bot


def index_files(bot: RAGChatbot, paths: list[Path]) -> None:
    """Embed ``paths`` into the live index with a progress bar and friendly errors."""
    if not paths:
        return
    bar = st.progress(0.0, text="Preparing documents ...")

    def on_progress(done: int, total: int) -> None:
        bar.progress(done / total, text=f"Embedding chunks {done}/{total}")

    try:
        added = add_files_to_index(bot.vector_store, paths, progress=on_progress)
    except Exception as exc:
        bar.empty()
        if is_rate_limit_error(exc):
            st.warning("Gemini's rate limit was hit. Wait a minute and try again.")
        else:
            st.error(f"Could not index: {exc}")
        return
    bar.empty()
    for name, n in added.items():
        if n:
            st.success(f"Indexed {name} ({n} chunks).")
        else:
            st.warning(f"{name} has no extractable text (scanned PDF?). Upload a text-based version.")


def save_uploads(uploads) -> list[Path]:
    seen = st.session_state.setdefault("saved_uploads", set())
    new_paths = []
    for up in uploads:
        key = (up.name, up.size)
        if key in seen:
            continue
        path = docs_dir / up.name
        path.write_bytes(up.getbuffer())
        new_paths.append(path)
        seen.add(key)
    return new_paths


def chunk_counts(bot: RAGChatbot) -> Counter:
    return Counter(rec["metadata"].get("source") for rec in bot.vector_store.store.values())


def render_sources(result) -> str:
    lines = []
    for i, (doc, score) in enumerate(result.sources, start=1):
        meta = doc.metadata
        where = meta.get("source", "?")
        detail = meta.get("section") or (f"page {meta['page']}" if meta.get("page") else "")
        lines.append(
            f"<div class='source'><b>[{i}] {where}</b>"
            + (f" · {detail}" if detail else "")
            + f" · relevance {score:.2f}</div>"
        )
    return "\n".join(lines)


# --- Gate on API key ---------------------------------------------------------------
if not os.getenv("GOOGLE_API_KEY"):
    st.markdown(
        f"<div class='hero'><div class='logo'>🧠</div><div><h1>{APP_NAME}</h1><p>{TAGLINE}</p></div></div>",
        unsafe_allow_html=True,
    )
    key = st.text_input("Gemini API key", type="password", help="Kept only in this session's memory.")
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        st.rerun()
    st.info("Enter a Gemini API key to start. On Streamlit Cloud, set GOOGLE_API_KEY in the app Secrets.")
    st.stop()

try:
    bot = get_bot()
except Exception as exc:
    st.error(f"Could not initialise the assistant: {exc}")
    st.stop()

counts = chunk_counts(bot)
indexed = indexed_sources(bot.vector_store)
pending = unindexed_files(bot.vector_store, docs_dir)
on_disk = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in config.SUPPORTED_EXTENSIONS)

# --- Sidebar: knowledge base -----------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Knowledge base")
    st.caption("The assistant answers only from these documents. Download one to read along.")
    for path in on_disk:
        ext = path.suffix.lower()
        is_indexed = path.name in indexed
        status = f"{counts[path.name]} chunks · searchable" if is_indexed else "not indexed yet"
        st.markdown(
            f"<div class='doc-card'><div class='name'>{FILE_ICONS.get(ext, '📁')} {path.name}</div>"
            f"<div class='meta'>{path.stat().st_size / 1024:.0f} KB · {status}</div></div>",
            unsafe_allow_html=True,
        )
        st.download_button(
            "⬇ Download",
            data=path.read_bytes(),
            file_name=path.name,
            mime=MIME_TYPES.get(ext, "application/octet-stream"),
            key=f"dl-{path.name}",
            use_container_width=True,
        )

    st.divider()
    if st.button("🗑 New conversation", use_container_width=True):
        st.session_state.pop("messages", None)
        bot.reset()
        st.rerun()

    # --- Admin panel --------------------------------------------------------------------
    with st.expander("⚙️ Admin", expanded=False):
        authorised = True
        if ADMIN_PASSWORD:
            entered = st.text_input("Admin password", type="password", key="admin-pw")
            authorised = entered == ADMIN_PASSWORD
            if entered and not authorised:
                st.error("Wrong password.")
        if authorised:
            uploads = st.file_uploader(
                "Upload documents", type=["ipynb", "md", "txt", "pdf"], accept_multiple_files=True
            )
            if uploads:
                index_files(bot, save_uploads(uploads))
                st.rerun()
            if pending:
                st.warning(f"{len(pending)} file(s) not indexed yet.")
                if st.button(f"Index {len(pending)} new file(s)", use_container_width=True):
                    index_files(bot, pending)
                    st.rerun()
            if st.button("Rebuild index from scratch", use_container_width=True):
                st.session_state.pop("messages", None)
                st.session_state.pop("saved_uploads", None)
                get_bot(force=True)
                st.rerun()
            st.caption(
                f"Model {config.CHAT_MODEL} · embeddings {config.EMBEDDING_MODEL} · top-k {config.TOP_K}"
            )

# --- Header ---------------------------------------------------------------------------
st.markdown(
    f"<div class='hero'><div class='logo'>🧠</div><div><h1>{APP_NAME}</h1><p>{TAGLINE}</p></div></div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<span class='pill ok'>● Online</span>"
    f"<span class='pill'>{len(indexed)} documents · {len(bot.vector_store.store)} chunks</span>"
    + (f"<span class='pill warn'>{len(pending)} awaiting indexing</span>" if pending else ""),
    unsafe_allow_html=True,
)

# --- Chat ----------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown(
        "<div class='welcome'><h3>👋 Hello! How can I help?</h3>"
        "<p>I answer using only the documents in the knowledge base and tell you which one I used. "
        "Try one of these to get started:</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"sugg-{i}", use_container_width=True):
            st.session_state.pending_prompt = q
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({msg['n_sources']})"):
                st.markdown(msg["sources"], unsafe_allow_html=True)

prompt = st.chat_input("Ask a question about the documents...") or st.session_state.pop("pending_prompt", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Searching the documents..."):
            try:
                result = bot.ask(prompt)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    st.warning("Gemini's rate limit was hit. Please wait a minute and ask again.")
                else:
                    st.error(f"Something went wrong: {exc}")
                st.stop()
        st.markdown(result.answer)
        source_html = render_sources(result) if result.sources else ""
        if source_html:
            with st.expander(f"Sources ({len(result.sources)})"):
                st.markdown(source_html, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": result.answer, "sources": source_html, "n_sources": len(result.sources)}
    )
    st.rerun()
