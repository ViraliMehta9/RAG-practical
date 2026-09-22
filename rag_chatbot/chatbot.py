"""The chatbot itself (Section 8 of the workshop - "Final Mini AI System").

On every turn:
    1. (optional) rewrite a follow-up question into a standalone one using history
    2. retrieve the top-k chunks relevant to the question           (Section 7)
    3. build a system prompt containing that context                (Section 7.6)
    4. send system + trimmed conversation history + new message      (Section 5)
    5. record the turn so the next one remembers it
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from . import config
from .ingest import is_rate_limit_error, load_or_build_index, with_retry  # noqa: F401

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about the documents \
provided in CONTEXT below. The knowledge base may contain several documents (for example \
the "Practical Generative AI Workshop" notebook, company handbooks, brochures or PDFs). \
Each context chunk starts with its source file name.

Rules:
- Base your answer on the CONTEXT. Quote code or figures from the context when it helps, \
and mention which document the answer came from.
- You may also use facts already established earlier in this conversation \
(for example the user's name).
- If the answer is in neither the context nor the conversation, say: \
"I don't have that information in the documents." Do not guess or use outside knowledge \
to invent details about the documents.
- Be concise and use Markdown formatting (code blocks for code).

CONTEXT:
{context}
"""

CONDENSE_PROMPT = """Given the conversation so far and a follow-up message, rewrite the \
follow-up as a single standalone question that can be understood without the conversation. \
Keep it short. If the message is already standalone or is not a question (e.g. a greeting), \
return it unchanged.

Conversation:
{history}

Follow-up message: {question}

Standalone question:"""


def get_text(response) -> str:
    """Return just the visible text of an LLM response.

    ``response.content`` is usually a string, but some Gemini models return a list of
    content blocks (text + an internal thought-signature block). Handle both.
    """
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


@dataclass
class ChatResult:
    answer: str
    sources: list[tuple[Document, float]] = field(default_factory=list)
    retrieval_query: str = ""

    def format_sources(self) -> str:
        lines = []
        for i, (doc, score) in enumerate(self.sources, start=1):
            meta = doc.metadata
            where = meta.get("source", "?")
            if "section" in meta:
                where += f" › {meta['section']}"
            if "cell_index" in meta:
                where += f" (cell {meta['cell_index']})"
            if "page" in meta:
                where += f" (page {meta['page']})"
            lines.append(f"[{i}] {where}  (score {score:.3f})")
        return "\n".join(lines)


class RAGChatbot:
    def __init__(
        self,
        vector_store: InMemoryVectorStore,
        llm: ChatGoogleGenerativeAI | None = None,
        top_k: int = config.TOP_K,
        max_history_turns: int = config.MAX_HISTORY_TURNS,
        condense_followups: bool = True,
    ):
        config.require_api_key()
        self.vector_store = vector_store
        self.llm = llm or ChatGoogleGenerativeAI(
            model=config.CHAT_MODEL,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_OUTPUT_TOKENS,
            transport=config.TRANSPORT,
            timeout=config.REQUEST_TIMEOUT,
            max_retries=2,
        )
        self.top_k = top_k
        self.max_history_turns = max_history_turns
        self.condense_followups = condense_followups
        self.history: list[BaseMessage] = []  # short-term memory (Section 5)

    # -- memory --------------------------------------------------------------
    def reset(self) -> None:
        self.history.clear()

    def _trimmed_history(self) -> list[BaseMessage]:
        # Keep only the most recent N human/AI pairs so the prompt stays bounded.
        return self.history[-2 * self.max_history_turns :]

    # -- retrieval -----------------------------------------------------------
    def _standalone_question(self, user_message: str) -> str:
        if not self.condense_followups or not self.history:
            return user_message
        history_text = "\n".join(
            f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
            for m in self._trimmed_history()
        )
        prompt = CONDENSE_PROMPT.format(history=history_text, question=user_message)
        try:
            rewritten = get_text(self.llm.invoke(prompt)).strip()
        except Exception:
            return user_message
        return rewritten or user_message

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        # Embedding the query is one Gemini request; retry briefly if rate limited.
        return with_retry(
            lambda: self.vector_store.similarity_search_with_score(query, k=self.top_k),
            retries=3,
        )

    # -- generation ----------------------------------------------------------
    def ask(self, user_message: str) -> ChatResult:
        query = self._standalone_question(user_message)
        sources = self.retrieve(query)
        context = "\n\n---\n\n".join(
            f"[Source: {doc.metadata.get('source', 'unknown')}"
            + (f", page {doc.metadata['page']}" if doc.metadata.get("page") else "")
            + f"]\n{doc.page_content}"
            for doc, _ in sources
        ) or "(no documents matched)"

        messages: list[BaseMessage] = [
            SystemMessage(content=SYSTEM_PROMPT.format(context=context)),
            *self._trimmed_history(),
            HumanMessage(content=user_message),
        ]
        answer = get_text(self.llm.invoke(messages))

        self.history.append(HumanMessage(content=user_message))
        self.history.append(AIMessage(content=answer))
        return ChatResult(answer=answer, sources=sources, retrieval_query=query)


def load_chatbot(force_rebuild: bool = False, verbose: bool = True, **kwargs) -> RAGChatbot:
    """Convenience: load (or build) the index and return a ready chatbot."""
    store = load_or_build_index(force=force_rebuild, verbose=verbose)
    return RAGChatbot(store, **kwargs)
