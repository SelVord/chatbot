"""
core/rag_chain.py — RAG pipeline.

Flow:
  1. Retrieve top-K docs with relevance scores from FAISS
  2. Filter out docs below the relevance threshold
  3. If nothing passes → return NO_CONTEXT_REPLY (bot says "I don't know")
  4. Format context + recent history and call LLM
  5. Return (answer, sources_list)
"""

from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.vectorstores import FAISS

import config
from core.llm import get_llm


# ── Prompt builder ─────────────────────────────────────────────────────

def _build_messages(
    question: str,
    context: str,
    bot_name: str,
    system_prompt: str,
    history: list[dict],
) -> list:
    """Assemble the messages list for the LLM."""
    system_content = f"""Ты — {bot_name}.

{system_prompt}

═══════════════════════════════
ПРАВИЛА (обязательные):
1. Отвечай ТОЛЬКО на основе «Контекста из документов» ниже.
2. Если ответ не содержится в контексте — ответь ровно одной фразой:
   «Я не нашёл информации по этому вопросу в доступных документах.»
3. Не придумывай и не добавляй факты из своих общих знаний.
4. Отвечай коротко и по делу (1–4 предложения, если не просят подробнее).
5. Пиши на том же языке, что и вопрос пользователя.
═══════════════════════════════

Контекст из документов:
{context}
"""

    messages = [SystemMessage(content=system_content)]

    # Add recent conversation history
    for msg in history[-(config.MAX_HISTORY_PAIRS * 2):]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=question))
    return messages


# ── Source formatting ──────────────────────────────────────────────────

def _format_sources(scored_docs: list[tuple]) -> list[dict]:
    """Extract unique source metadata from retrieved docs."""
    seen = set()
    sources = []
    for doc, score in scored_docs:
        key = (
            doc.metadata.get("source_file", ""),
            doc.metadata.get("page", 0),
        )
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": doc.metadata.get("source_file", "неизвестный файл"),
                "page": int(doc.metadata.get("page", 0)) + 1,
                "score": round(float(score), 3),
                "snippet": doc.page_content[:200].replace("\n", " "),
            })
    return sorted(sources, key=lambda x: -x["score"])


# ── Main RAG function ──────────────────────────────────────────────────

def ask(
    question: str,
    vector_store: FAISS,
    session: dict,
    history: list[dict],
    threshold: float = config.RELEVANCE_THRESHOLD,
) -> tuple[str, list[dict]]:
    """
    Run RAG pipeline.

    Args:
        question:      User question
        vector_store:  Loaded FAISS index for this session
        session:       Session config dict (bot_name, system_prompt, llm_*)
        history:       List of previous messages [{role, content}]
        threshold:     Minimum relevance score (0–1)

    Returns:
        (answer_text, sources_list)
    """
    # 1. Retrieve with scores
    try:
        scored_docs = vector_store.similarity_search_with_relevance_scores(
            question, k=config.TOP_K_DOCS
        )
    except Exception as e:
        return f"⚠️ Ошибка поиска в векторном хранилище: {e}", []

    # 2. Filter by threshold
    relevant = [(doc, score) for doc, score in scored_docs if score >= threshold]

    if not relevant:
        return config.NO_CONTEXT_REPLY, []

    # 3. Build context string
    context_parts = []
    for i, (doc, score) in enumerate(relevant, 1):
        file_name = doc.metadata.get("source_file", "")
        page = int(doc.metadata.get("page", 0)) + 1
        context_parts.append(
            f"[{i}] Источник: {file_name}, стр. {page}\n{doc.page_content}"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 4. Build messages and call LLM
    llm = get_llm(
        provider=session.get("llm_provider", config.LLM_PROVIDER),
        model=session.get("llm_model", config.OPENAI_MODEL),
    )
    messages = _build_messages(
        question=question,
        context=context,
        bot_name=session.get("bot_name", config.DEFAULT_BOT_NAME),
        system_prompt=session.get("system_prompt", config.DEFAULT_SYSTEM_PROMPT),
        history=history,
    )

    try:
        response = llm.invoke(messages)
        answer = response.content
    except Exception as e:
        return f"⚠️ Ошибка LLM: {e}", []

    # 5. Format sources
    sources = _format_sources(relevant)

    return answer, sources
