"""Streamlit web UI for the RAG chatbot.

    streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from rag_chatbot import RAGChatbot, config
from rag_chatbot.ingest import load_or_build_index

st.set_page_config(page_title="Workshop RAG Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 Practical GenAI Workshop — RAG Chatbot")
st.caption(
    f"Ask anything about the workshop notebook. Model: `{config.CHAT_MODEL}` · "
    f"Embeddings: `{config.EMBEDDING_MODEL}` · Top-k: {config.TOP_K}"
)


# --- Sidebar: API key, documents, index management --------------------------
with st.sidebar:
    st.header("Setup")
    if not os.getenv("GOOGLE_API_KEY"):
        key = st.text_input("Gemini API key", type="password", help="Kept only in this session's memory.")
        if key:
            os.environ["GOOGLE_API_KEY"] = key
    else:
        st.success("Gemini API key loaded")

    st.header("Documents")
    docs_dir: Path = config.DOCS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in docs_dir.iterdir() if p.suffix.lower() in config.SUPPORTED_EXTENSIONS)
    st.write("\n".join(f"- `{n}`" for n in existing) or "_no documents yet_")

    uploads = st.file_uploader(
        "Add documents", type=["ipynb", "md", "txt", "pdf"], accept_multiple_files=True
    )
    if uploads:
        for up in uploads:
            (docs_dir / up.name).write_bytes(up.getbuffer())
        st.info(f"Saved {len(uploads)} file(s). Click **Rebuild index** to include them.")

    rebuild = st.button("Rebuild index", use_container_width=True)
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.pop("bot", None)
        st.session_state.pop("messages", None)
        st.rerun()


# --- Load / build the chatbot once per session ------------------------------
def get_bot(force: bool = False) -> RAGChatbot:
    if force or "bot" not in st.session_state:
        with st.spinner("Building vector index..." if force else "Loading vector index..."):
            store = load_or_build_index(force=force, verbose=False)
        st.session_state.bot = RAGChatbot(store)
    return st.session_state.bot


if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Enter your Gemini API key in the sidebar to start.")
    st.stop()

try:
    bot = get_bot(force=rebuild)
except Exception as exc:
    st.error(f"Could not initialise the chatbot: {exc}")
    st.stop()

if rebuild:
    st.session_state.pop("messages", None)
    st.toast("Index rebuilt")

# --- Chat transcript ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved context"):
                st.markdown(msg["sources"])

if prompt := st.chat_input("Ask about the workshop, e.g. 'What does temperature control?'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = bot.ask(prompt)
            except Exception as exc:
                st.error(f"Error talking to Gemini: {exc}")
                st.stop()
        st.markdown(result.answer)

        source_md = ""
        if result.sources:
            parts = []
            if result.retrieval_query != prompt:
                parts.append(f"_Search query used:_ `{result.retrieval_query}`\n")
            for i, (doc, score) in enumerate(result.sources, start=1):
                meta = doc.metadata
                title = meta.get("source", "?")
                if meta.get("section"):
                    title += f" › {meta['section']}"
                parts.append(f"**[{i}] {title}** · score {score:.3f}\n\n```\n{doc.page_content[:700]}\n```")
            source_md = "\n\n".join(parts)
            with st.expander("Retrieved context"):
                st.markdown(source_md)

    st.session_state.messages.append({"role": "assistant", "content": result.answer, "sources": source_md})
