"""Terminal chat interface.

    python chat.py                 # uses ./docs and ./index
    python chat.py --rebuild       # re-ingest documents first
    python chat.py --no-sources    # hide the retrieved-chunk list

Commands inside the chat: /reset  /sources  /quit
"""
from __future__ import annotations

import argparse
import sys

from rag_chatbot import load_chatbot


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="RAG chatbot over the workshop notebook.")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector index before chatting")
    parser.add_argument("--no-sources", action="store_true", help="Do not print retrieved sources")
    args = parser.parse_args(argv)

    try:
        bot = load_chatbot(force_rebuild=args.rebuild)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    show_sources = not args.no_sources
    last = None
    print("\nRAG chatbot ready. Ask about the workshop notebook. Type /quit to exit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/reset":
            bot.reset()
            print("(conversation memory cleared)\n")
            continue
        if user == "/sources":
            print(last.format_sources() if last else "(no previous answer)", "\n")
            continue

        try:
            last = bot.ask(user)
        except Exception as exc:
            print(f"Error talking to Gemini: {exc}\n", file=sys.stderr)
            continue

        print(f"\nBot: {last.answer}\n")
        if show_sources and last.sources:
            print("Sources:\n" + last.format_sources() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
