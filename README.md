# Workshop RAG Chatbot

A Retrieval-Augmented Generation chatbot built from the **Practical Generative AI
Workshop** notebook, using the same stack the notebook teaches: Python, LangChain and
Google Gemini. The notebook itself is the knowledge base, so you can ask the bot things
like *"what does temperature do?"*, *"show me the code that builds the vector store"* or
*"how many vacation days does Acme give?"* (the sample handbook from Section 7 is
included too).

```
User -> Chat UI -> Conversation history -> Retriever -> Relevant chunks -> Gemini -> Answer
```

## What's inside

| Path | Workshop section | Purpose |
|---|---|---|
| `rag_chatbot/config.py` | 2, 3 | Model names, temperature, chunk size, top-k (all env-overridable) |
| `rag_chatbot/loaders.py` | 7.1 | Load `.ipynb`, `.md`, `.txt`, `.pdf` into LangChain `Document`s |
| `rag_chatbot/ingest.py` | 7.2 – 7.4 | Chunk, embed with Gemini, persist an `InMemoryVectorStore` to disk |
| `rag_chatbot/chatbot.py` | 5, 7.5, 7.6, 8 | Retrieval + short-term memory + grounded system prompt |
| `chat.py` | 8 | Terminal chat |
| `app.py` | 8 | Streamlit web chat with source citations and document upload |
| `docs/` | 7 | Knowledge base: the workshop notebook and the Acme handbook |

## Setup

Python 3.10+ is required.

```bash
cd rag-chatbot
python -m venv .venv
.venv\Scripts\activate          # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
copy .env.example .env          # then paste your Gemini key into .env
```

Get a free Gemini API key at https://aistudio.google.com/app/apikey.

## Run

Build the index (loads `docs/`, chunks, embeds, saves to `index/vector_store.json`):

```bash
python -m rag_chatbot.ingest
```

Chat in the terminal:

```bash
python chat.py
```

Or use the web UI:

```bash
streamlit run app.py
```

The web UI lets you paste the API key if it isn't in `.env`, upload more documents, and
rebuild the index with one click. Every answer shows the retrieved chunks and their
similarity scores so you can see *why* the bot said what it said.

## Adding your own documents

Drop `.ipynb`, `.md`, `.txt` or `.pdf` files into `docs/` and run
`python -m rag_chatbot.ingest --force` (or click **Rebuild index** in the web UI).

## How it differs from the notebook

The notebook code is kept almost verbatim, with a few production-minded additions:

- **Notebook loader**: each cell becomes a document tagged with its section heading and
  cell index, and code outputs are included, so retrieval can cite exact cells.
- **Persistent index**: the in-memory vector store is dumped to JSON so you embed once,
  not on every start.
- **Bounded memory**: only the last 10 turns are resent (the notebook flags this as the
  "next problem you'd hit").
- **Follow-up handling**: a follow-up like *"and the overlap?"* is first rewritten into a
  standalone question using the history, so retrieval still finds the right chunk.
- **Source citations** with similarity scores in both UIs.

## Troubleshooting

- `GOOGLE_API_KEY is not set` – create `.env` from `.env.example` or export the variable.
- A model name errors out – list the models available to your key and set
  `RAG_CHAT_MODEL` / `RAG_EMBEDDING_MODEL` in `.env` accordingly.
- Answers ignore new documents – re-run ingestion with `--force`; the saved index is
  only rebuilt when you ask for it.
