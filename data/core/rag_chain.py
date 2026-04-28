"""
core/rag_chain.py — RAG pipeline (non-streaming and streaming).

Flow:
  1. Retrieve top-K docs with relevance scores from FAISS
  2. Filter below threshold; fall back to top-1 if nothing passes
  3. Always call the LLM — greetings / identity questions work naturally
  4. Return answer + sources
"""

from typing import Generator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.vectorstores import FAISS

import config
from .llm import get_llm


# ── Retrieval ──────────────────────────────────────────────────────────

def _retrieve(
    question: str,
    vector_store: Optional[FAISS],
    threshold: float,
) -> list[tuple]:
    """Return (doc, score) pairs, filtered by threshold with a top-1 fallback."""
    if vector_store is None:
        return []
    scored = vector_store.similarity_search_with_relevance_scores(
        question, k=config.TOP_K_DOCS
    )
    relevant = [(d, s) for d, s in scored if s >= threshold]
    if not relevant and scored:
        relevant = scored[:1]
    return relevant


# ── Context builder ────────────────────────────────────────────────────

def _build_context(relevant: list[tuple]) -> str:
    if not relevant:
        return ""
    parts = []
    for i, (doc, _score) in enumerate(relevant, 1):
        file_name = doc.metadata.get("source_file", "")
        page = int(doc.metadata.get("page", 0)) + 1
        parts.append(f"[{i}] Source: {file_name}, page {page}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Prompt builder ─────────────────────────────────────────────────────

def _build_messages(
    question: str,
    context: str,
    bot_name: str,
    system_prompt: str,
    history: list[dict],
) -> list:
    ctx_block = context if context else "(No relevant documents found for this question.)"
    system_content = f"""You are {bot_name}, an AI assistant.

{system_prompt}

═══════════════════════════════
RULES:
1. You are an AI. You do not have a phone number, email, address, age, body,
   feelings, or any other personal attributes. If asked for something personal
   that only a human or real company would have, politely clarify you are an AI
   and offer to help with what the user actually needs.
2. For greetings and questions about your name, role, or what you can help with:
   respond naturally and helpfully.
3. For factual or knowledge questions: base your answer ONLY on the
   "Document context" below. Never mix up "your" personal identity with
   information that happens to appear in the documents.
4. If a topic is not covered in the documents, say so specifically —
   e.g. "I don't have information about [exact topic]" — never give a vague reply.
5. Keep answers concise (1–4 sentences unless the user asks for more detail).
6. Reply in the same language the user used.
═══════════════════════════════

Document context:
{ctx_block}
"""
    messages = [SystemMessage(content=system_content)]
    for msg in history[-(config.MAX_HISTORY_PAIRS * 2):]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    return messages


# ── Source formatter ───────────────────────────────────────────────────

def _format_sources(scored_docs: list[tuple]) -> list[dict]:
    seen: set = set()
    sources = []
    for doc, score in scored_docs:
        key = (doc.metadata.get("source_file", ""), doc.metadata.get("page", 0))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("source_file", "unknown file"),
                "page": int(doc.metadata.get("page", 0)) + 1,
                "score": round(float(score), 3),
                "snippet": doc.page_content[:200].replace("\n", " "),
            })
    return sorted(sources, key=lambda x: -x["score"])


# ── LLM helper ────────────────────────────────────────────────────────

def _get_llm(session: dict):
    return get_llm(
        provider=session.get("llm_provider", config.LLM_PROVIDER),
        model=session.get("llm_model", config.OPENAI_MODEL),
    )


# ── Public API ─────────────────────────────────────────────────────────

def ask(
    question: str,
    vector_store,
    session: dict,
    history: list[dict],
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> tuple[str, list[dict]]:
    """Non-streaming RAG call. Returns (answer, sources)."""
    try:
        relevant = _retrieve(question, vector_store, threshold)
    except Exception as e:
        return f"⚠️ Vector store search error: {e}", []

    messages = _build_messages(
        question=question,
        context=_build_context(relevant),
        bot_name=session.get("bot_name", config.DEFAULT_BOT_NAME),
        system_prompt=session.get("system_prompt", config.DEFAULT_SYSTEM_PROMPT),
        history=history,
    )
    try:
        response = _get_llm(session).invoke(messages)
        answer = response.content
    except Exception as e:
        err = str(e)
        if "404" in err and session.get("llm_provider") == "ollama":
            model = session.get("llm_model", config.OLLAMA_MODEL).removesuffix(":latest")
            return (
                f"⚠️ LLM error: Ollama call failed with status code 404. "
                f"Maybe your model is not found and you should pull the model with "
                f"`ollama pull {model}`."
            ), []
        return f"⚠️ LLM error: {e}", []

    return answer, _format_sources(relevant)


def ask_stream(
    question: str,
    vector_store,
    session: dict,
    history: list[dict],
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> Generator[tuple[str, list | None], None, None]:
    """
    Streaming RAG call.
    Yields (text_chunk, None) for each streamed token.
    Final yield is ("", sources_list) with the source metadata.
    """
    try:
        relevant = _retrieve(question, vector_store, threshold)
    except Exception as e:
        yield f"⚠️ Vector store search error: {e}", None
        yield "", []
        return

    messages = _build_messages(
        question=question,
        context=_build_context(relevant),
        bot_name=session.get("bot_name", config.DEFAULT_BOT_NAME),
        system_prompt=session.get("system_prompt", config.DEFAULT_SYSTEM_PROMPT),
        history=history,
    )
    try:
        for chunk in _get_llm(session).stream(messages):
            if chunk.content:
                yield chunk.content, None
    except Exception as e:
        err = str(e)
        if "404" in err and session.get("llm_provider") == "ollama":
            model = session.get("llm_model", config.OLLAMA_MODEL).removesuffix(":latest")
            yield (
                f"⚠️ LLM error: Ollama call failed with status code 404. "
                f"Maybe your model is not found and you should pull the model with "
                f"`ollama pull {model}`."
            ), None
        else:
            yield f"⚠️ LLM error: {e}", None

    yield "", _format_sources(relevant)
