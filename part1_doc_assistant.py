"""
Grounded document assistant for MGC Developments sales staff.

Design decision: no vector DB / chunking / embedding pipeline. The corpus is
three short markdown documents (a few KB total), so on every query we just pass
the full text of all three straight into the Gemini prompt as context. A RAG
pipeline would add moving parts (chunking, an embedding store, retrieval) that
buy nothing at this size and would be harder to defend on a call than simply
"the whole corpus fits in the prompt, so we send the whole corpus."
"""

import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent / "docs"
MODEL_NAME = "gemini-3.6-flash"  # gemini-2.5-flash is blocked for this API key/tier;
# the API's own 404 error names this as the direct replacement.

SYSTEM_PROMPT = """You are a grounded document assistant for MGC Developments sales staff. \
You answer questions about MGC Aurora Heights using ONLY the documents provided below. \
Follow these rules strictly:

1. Answer ONLY using facts present in the provided documents. Never invent or guess a \
figure, date, or fact that is not stated in the documents.
2. Every fact in your answer must cite which document(s) it came from, in the form \
"[Source: <filename>]". Cite inline, next to the fact it supports.
3. If two documents state different values for the same fact, do NOT pick one. State \
BOTH values explicitly, each with its own source citation, and note that they disagree.
4. If a document explicitly says something is unconfirmed, pending, not yet decided, or \
that staff must not quote a figure verbally, say so plainly using that document's own \
framing — do not soften it into a generic "I don't know."
5. If the answer is genuinely not covered by the documents, say plainly that you don't \
have that information in the documents, and suggest a specific person or role to ask \
(e.g. "ask the marketing manager") if the documents mention who owns that topic. Never \
fabricate a number to fill the gap.
6. For price calculations (e.g. base price plus stacked location premiums), do the \
arithmetic yourself using the exact figures found in the documents, and show the \
calculation step by step in your answer.

Respond with ONLY a JSON object (no markdown fences, no extra text) of the exact form:
{"answer": "<your answer text, including inline [Source: ...] citations>", \
"sources": ["<filename1>", "<filename2>", ...]}
The "sources" list must contain the filenames of every document you actually drew on,
and nothing else.
"""


def _load_documents() -> dict[str, str]:
    if not DOCS_DIR.is_dir():
        raise RuntimeError(f"docs/ directory not found at {DOCS_DIR}")
    docs = {}
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs[path.name] = path.read_text(encoding="utf-8")
    if not docs:
        raise RuntimeError(f"No .md documents found in {DOCS_DIR}")
    return docs


def _build_context(docs: dict[str, str]) -> str:
    parts = [f"=== {name} ===\n{text}" for name, text in docs.items()]
    return "\n\n".join(parts)


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set.\n"
            "Create a .env file in the project root (copy .env.example) and set:\n"
            "  GEMINI_API_KEY=<your key>\n"
            "Get a free key (no credit card) at https://aistudio.google.com/apikey"
        )
    from google import genai

    return genai.Client(api_key=api_key)


def _parse_model_output(raw_text: str) -> dict:
    text = raw_text.strip()
    # Defensive: strip markdown code fences if the model adds them anyway.
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Last resort: pull out the first {...} block.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise RuntimeError(f"Model did not return valid JSON:\n{raw_text}")
        data = json.loads(match.group(0))
    answer = data.get("answer", "")
    sources = data.get("sources", [])
    if not isinstance(sources, list):
        sources = [sources]
    return {"answer": answer, "sources": sources}


def answer_question(question: str) -> dict:
    """Answer `question` grounded in docs/, returning {"answer": str, "sources": list[str]}."""
    docs = _load_documents()
    context = _build_context(docs)
    client = _get_client()

    prompt = f"{SYSTEM_PROMPT}\n\nDOCUMENTS:\n{context}\n\nQUESTION: {question}"

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config={"response_mime_type": "application/json"},
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API call failed: {exc}") from exc

    return _parse_model_output(response.text)


TEST_QUESTIONS = [
    "What's the base price of a 2-bed in Block B?",
    "What's the total for a Margalla-facing corner unit on floor 15, 2-bed Block B?",
    "What's the transfer fee?",
    "What's the rental yield on a 1-bed?",
    "Who is the anchor tenant?",
]


def _run_test_suite() -> None:
    for i, question in enumerate(TEST_QUESTIONS, start=1):
        print(f"\n{'=' * 70}")
        print(f"Q{i}: {question}")
        print("=" * 70)
        try:
            result = answer_question(question)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            return
        print(f"ANSWER: {result['answer']}")
        print(f"SOURCES: {result['sources']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        try:
            res = answer_question(q)
        except RuntimeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(1)
        print(f"ANSWER: {res['answer']}")
        print(f"SOURCES: {res['sources']}")
    else:
        _run_test_suite()
