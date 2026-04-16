"""
core/embeddings.py — Factory for embedding models.

Supported providers:
  "openai" — OpenAI text-embedding-3-small (needs API key)
  "local"  — sentence-transformers via HuggingFace (free, runs locally)
"""

from functools import lru_cache

import config


@lru_cache(maxsize=4)
def get_embeddings(provider: str = config.EMBEDDING_PROVIDER):
    """Return a cached embedding instance for the given provider."""
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=config.OPENAI_API_KEY,
        )
    elif provider == "local":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=config.LOCAL_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        raise ValueError(f"Unknown embedding provider: {provider!r}")
