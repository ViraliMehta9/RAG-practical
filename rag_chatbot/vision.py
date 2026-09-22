"""Photo understanding with Gemini's multimodal input.

Two jobs:
1. ``analyze_image`` - turn an uploaded photo into a structured "photo card"
   (scene, objects, colours, actions, mood, text, people + expressions).
2. ``describe_for_index`` - render that card as plain text so it can be embedded
   into the vector store and found by the normal RAG search.

Answering follow-up questions about a photo lives in ``RAGChatbot.ask_about_image``,
which sends the image itself along with the question for precise answers.
"""
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from . import config

MAX_SIDE = 1024  # downscale large photos: cheaper, faster, no loss for description tasks

ANALYSIS_PROMPT = """You are an expert image analyst. Describe this photo thoroughly and \
objectively, then return ONLY a JSON object with exactly these keys:

{
  "title": "short 3-6 word title",
  "summary": "2-3 sentence plain-language description of what the photo shows",
  "scene": "setting / location type / time of day / indoor-outdoor",
  "subject": "what the photo is mainly about or related to (e.g. sports, family, food, nature, work)",
  "objects": ["notable objects, animals, landmarks ..."],
  "colors": ["dominant colours in order, e.g. 'deep blue sky', 'warm orange'"],
  "actions": ["what is happening / what people or animals are doing"],
  "mood": "overall mood or atmosphere",
  "text_in_image": "any readable text, or empty string",
  "people_count": 0,
  "people": [
    {
      "position": "where in the frame (e.g. left, centre)",
      "apparent_age_group": "child / teenager / adult / older adult",
      "expression": "facial expression as seen (e.g. broad smile, furrowed brow)",
      "apparent_emotion": "likely emotion inferred from expression and body language",
      "posture_or_action": "what the person is doing",
      "clothing": "notable clothing or accessories"
    }
  ],
  "tags": ["5-10 short keywords"]
}

Rules: do not try to identify who any person is; describe only what is visible; \
use empty lists or empty strings when something does not apply."""


def prepare_image(data: bytes) -> tuple[bytes, str]:
    """Return (jpeg/png bytes, mime) downscaled to MAX_SIDE. Falls back to the original."""
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.load()
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")
        w, h = img.size
        scale = min(1.0, MAX_SIDE / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)))
        buf = io.BytesIO()
        if img.mode == "RGBA":
            img.save(buf, format="PNG")
            return buf.getvalue(), "image/png"
        img.save(buf, format="JPEG", quality=88)
        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, "image/jpeg"


def image_part(data: bytes, mime: str) -> dict:
    """A LangChain content block carrying the image inline."""
    b64 = base64.b64encode(data).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _vision_llm() -> ChatGoogleGenerativeAI:
    config.require_api_key()
    return ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        temperature=0.1,
        max_output_tokens=2048,
        transport=config.TRANSPORT,
        timeout=config.REQUEST_TIMEOUT,
        max_retries=2,
    )


def _text_of(response) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")


def _parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def analyze_image(data: bytes, mime: str) -> dict[str, Any]:
    """Run the structured analysis. Always returns a dict; ``raw`` holds the model text."""
    llm = _vision_llm()
    msg = HumanMessage(content=[{"type": "text", "text": ANALYSIS_PROMPT}, image_part(data, mime)])
    raw = _text_of(llm.invoke([msg]))
    parsed = _parse_json(raw) or {}
    parsed.setdefault("title", "Uploaded photo")
    parsed.setdefault("summary", raw if not parsed else "")
    for key in ("objects", "colors", "actions", "people", "tags"):
        if not isinstance(parsed.get(key), list):
            parsed[key] = []
    parsed.setdefault("people_count", len(parsed["people"]))
    parsed["raw"] = raw
    return parsed


def describe_for_index(analysis: dict[str, Any], filename: str) -> str:
    """Plain-text rendering of the card, used both for embedding and for the LLM context."""
    lines = [f"Photo: {filename}", f"Title: {analysis.get('title', '')}", f"Summary: {analysis.get('summary', '')}"]
    for key, label in (
        ("scene", "Scene"),
        ("subject", "Related to"),
        ("mood", "Mood"),
        ("text_in_image", "Text in image"),
    ):
        if analysis.get(key):
            lines.append(f"{label}: {analysis[key]}")
    for key, label in (("objects", "Objects"), ("colors", "Colours"), ("actions", "Actions"), ("tags", "Tags")):
        if analysis.get(key):
            lines.append(f"{label}: {', '.join(map(str, analysis[key]))}")
    people = analysis.get("people") or []
    lines.append(f"People: {analysis.get('people_count', len(people))}")
    for i, p in enumerate(people, start=1):
        if isinstance(p, dict):
            parts = [f"{k.replace('_', ' ')}: {v}" for k, v in p.items() if v]
            lines.append(f"  Person {i} - " + "; ".join(parts))
    return "\n".join(lines)


def analysis_document(analysis: dict[str, Any], filename: str) -> Document:
    return Document(
        page_content=describe_for_index(analysis, filename),
        metadata={"source": filename, "type": "image", "title": analysis.get("title", "")},
    )
