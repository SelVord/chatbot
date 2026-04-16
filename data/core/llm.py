"""
core/llm.py — Factory for LLM instances.

Supported providers:
  "openai" — GPT-4o-mini (or any OpenAI model)
  "ollama" — local Ollama server (llama3.2, mistral, etc.)
"""

from functools import lru_cache

import config


@lru_cache(maxsize=4)
def get_llm(provider: str = config.LLM_PROVIDER, model: str = ""):
    """Return a cached LLM instance."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or config.OPENAI_MODEL,
            temperature=0.2,
            openai_api_key=config.OPENAI_API_KEY,
            streaming=True,
        )
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model or config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            temperature=0.2,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
